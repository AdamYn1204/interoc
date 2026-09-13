"""由 keypoint（加上深度）算出距離。

這一層不碰模型、不吃權重、不下載任何東西，純粹是幾何。所以它和 stage 1/2/3
不同，**不是可選的**：有什麼資料就算什麼距離，沒有需要分段降級的失效模式，
也就不需要 Protocol 注入。

    只有 keypoint        → distance_px
    keypoint + 三維座標  → 再加上 distance_m

像素距離永遠會被算出來。它不依賴深度與內參，而且是驗證公尺數的對照組——
metric 路徑靜默出錯時，兩者的比值（`meters_per_pixel`）是最早露出馬腳的
地方。

焦距的角色
----------
`focal_px` 只在 :func:`unproject` 用到，但它決定了**所有橫向距離**的尺度：
X = (u - cx) · Z / fx。填錯焦距，深度再準，算出來的距離仍然等比例地錯，
而且畫面看起來完全正常。這不是可以用預設值帶過的參數。
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Sequence

import numpy as np

from kernel.schemas import (
    LEFT_EYE,
    RIGHT_EYE,
    CoordinateFrame,
    DepthMap,
    Instance,
    InterObjectEyeDistance,
    InterocularDistance,
    Keypoint,
    MeasurementResult,
    Point3D,
)

#: 深度取樣的視窗半徑（像素）。單點取值不可靠——眼睛常落在物體邊緣，
#: 而單目深度在邊界最不穩，很容易讀到背景的深度。
DEPTH_SAMPLE_RADIUS = 5

#: 視窗內多接近中位數才算「一致」。有效像素中落在這個相對誤差內的比例即為
#: 深度取樣的信心值，直接反映「這一小塊的深度到底穩不穩」。
DEPTH_CONSISTENCY_TOLERANCE = 0.10


# -- 深度取樣與反投影 -------------------------------------------------------


def sample_depth(
    depth_map: DepthMap,
    keypoint: Keypoint,
    instance: Instance,
    radius: int = DEPTH_SAMPLE_RADIUS,
) -> tuple[float, float] | None:
    """在 keypoint 周圍穩健地取一個深度值，回傳 ``(公尺, 信心)``。

    取不到有效像素時回傳 None。

    三個必要的動作，少一個結果就會明顯變差：

    1. **取一個視窗而非單一像素**——單目深度的逐像素雜訊很大。
    2. **和 instance mask 取交集**——眼睛靠近輪廓邊緣，視窗一定會蓋到背景，
       不濾掉的話中位數會被拉向背景的深度。跨物體量測特別吃這一項：讀到
       背景的那一顆眼睛會讓兩隻動物的距離差出好幾公尺。
    3. **用中位數而非平均**——邊界處的深度會整片「飛」到背景值，那是離群
       而不是雜訊，平均擋不住。

    信心值是「視窗內有多少比例的有效像素落在中位數的
    :data:`DEPTH_CONSISTENCY_TOLERANCE` 之內」，可以直接解讀成這一小塊的
    深度有多一致。
    """
    u = int(round(keypoint.point.u))
    v = int(round(keypoint.point.v))
    height, width = depth_map.shape

    u0, u1 = max(0, u - radius), min(width, u + radius + 1)
    v0, v1 = max(0, v - radius), min(height, v + radius + 1)
    if u0 >= u1 or v0 >= v1:
        return None

    window = depth_map.data[v0:v1, u0:u1]
    valid = depth_map.validity()[v0:v1, u0:u1]
    if instance.mask is not None:
        valid = valid & instance.mask.data[v0:v1, u0:u1]

    values = window[valid]
    if values.size == 0:
        return None

    median = float(np.median(values))
    if median <= 0.0:
        return None

    agree = np.abs(values - median) <= median * DEPTH_CONSISTENCY_TOLERANCE
    return median, float(agree.mean())


def unproject(
    u: float, v: float, depth: float, focal_px: float, cx: float, cy: float
) -> Point3D:
    """把一個像素座標加深度反投影成相機三維座標，單位公尺。

    `depth` 必須是 z-depth（沿光軸），不是到光心的射線長度。

    這裡假設像素為正方形（fx == fy）。現代感光元件基本上都成立；真要處理
    非方形像素時，這個簽名要拆成 fx 與 fy 兩個參數。
    """
    return Point3D(
        x=(u - cx) * depth / focal_px,
        y=(v - cy) * depth / focal_px,
        z=depth,
    )


def attach_3d(
    instances: Sequence[Instance],
    depth_map: DepthMap,
    image_size: tuple[int, int],
    focal_px: float,
    principal_point: tuple[float, float] | None = None,
) -> tuple[Instance, ...]:
    """替每顆 keypoint 取深度並反投影，回傳掛上三維座標的 instances。

    取不到有效深度的 keypoint 會**原樣保留**而不是被丟掉：像素距離仍然
    算得出來，把整顆點刪掉只會連帶讓 2D 結果一起消失。
    """
    depth_map.require_metric()
    height, width = image_size
    cx, cy = principal_point or (width / 2.0, height / 2.0)

    enriched: list[Instance] = []
    for instance in instances:
        keypoints: list[Keypoint] = []
        for kp in instance.keypoints:
            sample = sample_depth(depth_map, kp, instance)
            if sample is None:
                keypoints.append(kp)
                continue
            depth, confidence = sample
            position = unproject(kp.point.u, kp.point.v, depth, focal_px, cx, cy)
            keypoints.append(
                kp.with_depth(depth, confidence).with_position_3d(position)
            )
        enriched.append(instance.with_keypoints(tuple(keypoints)))
    return tuple(enriched)


# -- 量測 -------------------------------------------------------------------


def measure(
    instances: Sequence[Instance], cross_keypoint: str = RIGHT_EYE
) -> MeasurementResult:
    """算出這一格的全部距離。

    `cross_keypoint` 決定跨物體配對要用哪顆眼睛，預設右眼對右眼——挑定一顆
    是必要的：左右眼在不同動物身上各差一個眼距，混著配會多出一個與物種
    體型相關的系統性偏差。
    """
    return MeasurementResult(
        interocular=_interocular(instances),
        inter_object=_inter_object(instances, cross_keypoint),
    )


def _interocular(
    instances: Sequence[Instance],
) -> tuple[InterocularDistance, ...]:
    """每隻動物的雙眼間距。兩顆眼睛都在才算得出來。"""
    results: list[InterocularDistance] = []
    for instance in instances:
        left = instance.keypoint(LEFT_EYE)
        right = instance.keypoint(RIGHT_EYE)
        if left is None or right is None:
            continue
        if not _usable(left, right):
            continue

        metric = _metric_distance(left, right)
        results.append(
            InterocularDistance(
                instance_id=instance.instance_id,
                label=instance.label,
                confidence=_combine_confidence(left, right),
                distance_px=_pixel_distance(left, right),
                distance_m=metric,
                left_eye=left.position_3d if metric is not None else None,
                right_eye=right.position_3d if metric is not None else None,
            )
        )
    return tuple(results)


def _inter_object(
    instances: Sequence[Instance], cross_keypoint: str
) -> tuple[InterObjectEyeDistance, ...]:
    """跨動物的眼對眼距離，N 隻共 N(N-1)/2 組。

    不同物種之間的配對沒有特別待遇——物種只是 metadata。要只看跨物種的
    結果請用 :meth:`MeasurementResult.cross_species`，在這裡先篩掉的話，
    同物種的那些就沒得當對照組了。
    """
    results: list[InterObjectEyeDistance] = []
    for a, b in combinations(instances, 2):
        kp_a = a.keypoint(cross_keypoint)
        kp_b = b.keypoint(cross_keypoint)
        if kp_a is None or kp_b is None:
            continue
        if not _usable(kp_a, kp_b):
            continue

        metric = _metric_distance(kp_a, kp_b)
        results.append(
            InterObjectEyeDistance(
                instance_a=a.instance_id,
                label_a=a.label,
                keypoint_a=cross_keypoint,
                instance_b=b.instance_id,
                label_b=b.label,
                keypoint_b=cross_keypoint,
                confidence=_combine_confidence(kp_a, kp_b),
                distance_px=_pixel_distance(kp_a, kp_b),
                distance_m=metric,
                point_a=kp_a.position_3d if metric is not None else None,
                point_b=kp_b.position_3d if metric is not None else None,
            )
        )
    return tuple(results)


# -- 量測的共用小工具 -------------------------------------------------------


def _usable(a: Keypoint, b: Keypoint) -> bool:
    """兩顆 keypoint 是否足以產生一筆量測。

    座標系不符直接拋錯而不是略過：那是組裝錯誤，不是資料問題。非有限值
    則略過——模型失敗時會吐 nan，而 nan 在後續的平均與比較裡是靜默傳染的。
    """
    for kp in (a, b):
        kp.point.require_frame(CoordinateFrame.IMAGE)
    return a.point.is_finite and b.point.is_finite


def _pixel_distance(a: Keypoint, b: Keypoint) -> float:
    """兩顆眼睛在原圖上的像素距離。不需要深度或內參，所以永遠算得出來。"""
    return math.hypot(a.point.u - b.point.u, a.point.v - b.point.v)


def _metric_distance(a: Keypoint, b: Keypoint) -> float | None:
    """兩顆眼睛的三維距離（公尺）；任一端缺三維座標時回傳 None。

    反投影出非正深度的點也視同缺失：z <= 0 不是「很近」，是算錯了。
    """
    if a.position_3d is None or b.position_3d is None:
        return None
    if not (a.position_3d.is_in_front and b.position_3d.is_in_front):
        return None
    return float(np.linalg.norm(a.position_3d.as_array() - b.position_3d.as_array()))


def _combine_confidence(a: Keypoint, b: Keypoint) -> float:
    """把兩端的定位信心與深度信心合成一個值。

    用相乘而不是取平均：任一端不可靠，整筆量測就不可靠，這個性質要保留。
    平均會讓一顆 0.95 的眼睛把一顆 0.05 的眼睛救起來，但那筆量測其實是廢的。
    """
    score = a.score * b.score
    for kp in (a, b):
        if kp.depth_confidence is not None:
            score *= kp.depth_confidence
    return float(min(max(score, 0.0), 1.0))
