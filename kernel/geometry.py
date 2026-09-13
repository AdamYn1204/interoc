"""由 keypoint 算出距離。

這一層不碰模型、不吃權重、不需要下載任何東西，純粹是幾何。所以它和
stage 1/2 不同，**不是可選的**：兩顆眼睛都在就一定算得出來，沒有需要
分段降級的失效模式，也就不需要 Protocol 注入。

目前只做像素距離。反投影成三維、以及由此得到的公尺數，要等 depth 階段
連同相機內參一起進來才有辦法算——理由見 :mod:`kernel.schemas.measurement`。
"""

from __future__ import annotations

import math
from typing import Sequence

from kernel.schemas import (
    LEFT_EYE,
    RIGHT_EYE,
    CoordinateFrame,
    Instance,
)
from kernel.schemas.measurement import InterocularDistance, MeasurementResult


def measure_interocular(instances: Sequence[Instance]) -> MeasurementResult:
    """為每隻湊齊雙眼的動物算一組雙眼像素距離。

    只定位到一顆眼睛的動物**不會**產生量測，也不會產生佔位用的 None。
    側臉時被遮住的那顆眼睛會被 ``min_score`` 濾掉（見
    :mod:`kernel.models.keypoint`），此時「量不到」是正確結果而不是缺漏，
    硬塞一筆進去只會讓評測時的分母失去意義。

    座標系不符會直接拋錯而不是略過：那是組裝錯誤，不是資料問題。
    """
    measurements: list[InterocularDistance] = []

    for inst in instances:
        left = inst.keypoint(LEFT_EYE)
        right = inst.keypoint(RIGHT_EYE)
        if left is None or right is None:
            continue

        # 兩顆眼睛都必須已經換算回原圖，否則算出來的距離是拿兩個不同尺度
        # 的數字相減，看起來完全正常。
        left.point.require_frame(CoordinateFrame.IMAGE)
        right.point.require_frame(CoordinateFrame.IMAGE)

        # 模型失敗時可能吐出 nan/inf。讓它流下去的話距離會是 nan，而 nan
        # 在後續的平均與比較裡是靜默傳染的。
        if not (left.point.is_finite and right.point.is_finite):
            continue

        measurements.append(
            InterocularDistance(
                instance_id=inst.instance_id,
                label=inst.label,
                distance_px=math.hypot(
                    right.point.u - left.point.u,
                    right.point.v - left.point.v,
                ),
                confidence=min(left.score, right.score),
            )
        )

    return MeasurementResult(tuple(measurements))
