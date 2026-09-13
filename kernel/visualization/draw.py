"""每個 stage 一張疊圖，供人工查驗。

    output/
    ├── 000000000285_1_masks.jpg    輪廓疊色 + bbox + 物種標籤
    ├── 000000000285_2_eyes.jpg     疊在 1 之上，加上眼睛位置與分數
    ├── 000000000285_3_depth.jpg    深度偽彩圖 + 輪廓 + 取樣視窗
    └── 000000000285_4_measure.jpg  兩種距離的連線與數字

這條 pipeline 上的失效幾乎都是**靜默**的——不丟例外、數值看起來也合理，
只是錯的：

    stage 1   多邊形柵格化歪掉、錯把背景納入輪廓
    stage 2   關鍵點索引錯位（鼻子被當成眼睛）、側臉時抓到被遮住的那顆眼
    stage 3   取樣視窗跨過輪廓邊界，深度讀到背景
    stage 4   連線跨到隔壁動物身上（keypoint 歸屬配錯）

其中關鍵點索引錯位那次，就是靠肉眼看疊圖才抓到的（見
:mod:`kernel.models.keypoint` 模組開頭）。所以疊圖在這個專案不是加分項，
而是主要的除錯手段。

1 與 2 是層層疊上去的，這樣才看得出「這顆眼睛屬於哪一塊輪廓」。3 換成深度
偽彩底圖、4 換成調暗的原圖，因為那兩層要看的是別的東西——疊在輪廓上反而
會被遮住。同一隻動物在各張圖上顏色固定，方便交叉比對。

所有繪圖都在 **RGB** 空間進行，只有 :func:`save` 會轉成 BGR——cv2 只有在
編碼寫檔時才在意通道順序。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from kernel.core import Scene
from kernel.geometry import DEPTH_SAMPLE_RADIUS
from kernel.schemas import LEFT_EYE, RIGHT_EYE, DepthMap, Keypoint
from kernel.visualization.palette import (
    CROSS_OBJECT_COLOR,
    FAILURE_COLOR,
    SAMPLE_WINDOW_COLOR,
    color_for,
    contrast_color,
)

MASK_ALPHA = 0.40

#: stage 4 底圖調暗的比例。不調暗的話，細線和數字會被原圖的細節吃掉。
DIM_FACTOR = 0.55

STAGE_FILENAMES = {
    1: "1_masks.jpg",
    2: "2_eyes.jpg",
    3: "3_depth.jpg",
    4: "4_measure.jpg",
}

ALL_STAGES = (1, 2, 3, 4)


def render_scene(
    image: np.ndarray,
    scene: Scene,
    outdir: Path,
    stages: Sequence[int] = ALL_STAGES,
) -> list[Path]:
    """把指定 stage 的疊圖平鋪寫進 ``outdir/``，回傳實際寫出的路徑。

    檔名是 ``<原始檔名>_<stage>.jpg``，不另開子目錄——一張圖最多幾個 stage，
    為此各建一個資料夾只是讓人多點幾層才看得到圖。

    renderer 回傳 None 代表這個 stage 沒有資料可畫（例如沒跑 stage 2，或跑了
    但一顆眼睛都沒過門檻），跳過而不是報錯——分段降級的原則在這裡同樣適用。
    """
    stem = scene.source.stem if scene.source else "scene"
    outdir.mkdir(parents=True, exist_ok=True)

    renderers = {
        1: draw_masks,
        2: draw_eyes,
        3: draw_depth,
        4: draw_measurements,
    }

    written: list[Path] = []
    for stage in stages:
        canvas = renderers[stage](image, scene)
        if canvas is None:
            continue
        path = outdir / f"{stem}_{STAGE_FILENAMES[stage]}"
        save(canvas, path)
        written.append(path)
    return written


# -- stage 1：切割 ----------------------------------------------------------


def draw_masks(
    image: np.ndarray, scene: Scene, label: LabelPlacer | None = None
) -> np.ndarray:
    """輪廓疊色 + bbox + 物種標籤。

    遮罩同時畫了半透明填色**和**實線邊界。只有填色的話看不出輪廓到底貼不貼
    合物體，而多邊形柵格化出錯時正是邊界會歪掉。

    `label` 可由上層的 stage 傳入共用，好讓兩層的標籤彼此避讓。
    """
    label = label or LabelPlacer()
    canvas = image.copy()
    thickness = _line_thickness(image)

    overlay = canvas.copy()
    for inst in scene.instances:
        if inst.mask is None:
            continue
        overlay[inst.mask.data] = color_for(inst.instance_id)
    cv2.addWeighted(overlay, MASK_ALPHA, canvas, 1 - MASK_ALPHA, 0, canvas)

    for inst in scene.instances:
        color = color_for(inst.instance_id)

        if inst.mask is not None:
            contours, _ = cv2.findContours(
                inst.mask.data.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(canvas, contours, -1, color, thickness)

        x1, y1, x2, y2 = (int(round(v)) for v in inst.bbox.as_xyxy())
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, max(1, thickness - 1))

        label(
            canvas,
            f"#{inst.instance_id} {inst.label} {inst.score:.2f}",
            (x1, y1),
            color,
            image,
        )

    return canvas


# -- stage 2：眼睛 ----------------------------------------------------------


def draw_eyes(
    image: np.ndarray, scene: Scene, label: LabelPlacer | None = None
) -> np.ndarray | None:
    """在 stage 1 之上標出眼睛。沒有任何眼睛時回傳 None。

    疊在輪廓上而不是畫在原圖，是為了讓「這顆眼睛屬於哪隻動物」一眼看得出來
    ——歸屬配錯是這一層最典型的失效。

    十字準星的中心是關鍵點的真實子像素位置。畫圓圈的話中心會被圓心取整
    掩蓋掉，而子像素精度直接影響最終的距離誤差。

    `label` 可由上層的 stage 傳入共用，理由同 :func:`draw_masks`。
    """
    if scene.n_eyes == 0:
        return None

    label = label or LabelPlacer()
    canvas = draw_masks(image, scene, label)
    thickness = _line_thickness(image)
    arm = max(6, int(_scale(image) * 8))

    for inst in scene.instances:
        color = color_for(inst.instance_id)
        for kp in inst.eyes:
            u, v = int(round(kp.point.u)), int(round(kp.point.v))
            cv2.line(canvas, (u - arm, v), (u + arm, v), color, thickness, cv2.LINE_AA)
            cv2.line(canvas, (u, v - arm), (u, v + arm), color, thickness, cv2.LINE_AA)
            cv2.circle(canvas, (u, v), arm // 2, color, thickness, cv2.LINE_AA)
            label(
                canvas,
                f"#{inst.instance_id} {kp.name} {kp.score:.2f}",
                (u + arm, v + arm),
                color,
                image,
            )

    return canvas


# -- stage 3：深度 ----------------------------------------------------------


def draw_depth(image: np.ndarray, scene: Scene) -> np.ndarray | None:
    """深度圖上色，並畫出每顆眼睛的**取樣視窗**。沒跑 stage 3 時回傳 None。

    畫取樣視窗是這張圖的重點，不是附帶資訊。深度取樣讀到背景是整條 pipeline
    最難察覺的失效：眼睛靠近輪廓邊緣，視窗必然會蓋到背景，而單目深度在邊界
    又最不穩。旁邊的 ``c=`` 直接告訴你視窗內的深度有多一致——低於 0.5 就代表
    那個視窗裡有兩群差很多的深度值，量出來的數字不能信。

    instance 輪廓也畫在深度圖上。少了它，這張圖就回答不了它唯一要回答的
    問題：取樣視窗到底有沒有跨過物體邊界。深度圖本身是看不出動物在哪的。
    """
    if scene.depth is None:
        return None

    label = LabelPlacer()
    canvas = _colorize_depth(scene.depth)
    thickness = _line_thickness(image)

    for inst in scene.instances:
        if inst.mask is None:
            continue
        contours, _ = cv2.findContours(
            inst.mask.data.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            canvas, contours, -1, color_for(inst.instance_id), max(1, thickness - 1)
        )

    for inst in scene.instances:
        for kp in inst.eyes:
            u, v = _xy(kp)
            r = DEPTH_SAMPLE_RADIUS
            color = SAMPLE_WINDOW_COLOR if kp.has_depth else FAILURE_COLOR
            cv2.rectangle(
                canvas, (u - r, v - r), (u + r, v + r), color, max(1, thickness - 1)
            )
            text = (
                f"#{inst.instance_id} {kp.depth:.2f}m c={kp.depth_confidence:.2f}"
                if kp.has_depth
                else f"#{inst.instance_id} no depth"
            )
            label(canvas, text, (u + r, v - r), color, image)

    return canvas


def _colorize_depth(depth_map: DepthMap) -> np.ndarray:
    """把公尺值轉成可看的偽彩圖。

    正規化用 2/98 百分位而不是最小/最大值：天空或反光面常有極端值，用極值
    正規化會把整個動物壓成同一個色調，那張圖就什麼也看不出來。
    """
    valid = depth_map.validity()
    data = depth_map.data

    if not valid.any():
        return np.zeros((*depth_map.shape, 3), dtype=np.uint8)

    lo, hi = np.percentile(data[valid], (2, 98))
    if hi <= lo:
        hi = lo + 1e-6

    normalized = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    gray = (normalized * 255).astype(np.uint8)
    colored = np.ascontiguousarray(cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)[:, :, ::-1])

    # 無效像素塗黑，才不會被誤讀成「很近」
    colored[~valid] = 0
    return colored


# -- stage 4：量測 ----------------------------------------------------------


def draw_measurements(image: np.ndarray, scene: Scene) -> np.ndarray | None:
    """畫出兩種距離：同一隻動物的雙眼連線，以及跨動物的眼對眼連線。

    沒有任何量測時回傳 None。

    底圖調暗，讓連線與數字讀得出來。雙眼連線用該動物的顏色、跨物體連線用
    白色虛線——兩者的誤差來源完全不同（前者由 keypoint 定位主導，後者由
    深度主導），視覺上必須一眼分得開。中性的白色留給跨物體，因為那條線
    不屬於任何一隻動物。

    這一層要查的失效是連線跨到隔壁動物身上：實線的兩端如果落在兩塊不同
    顏色的輪廓上，就是 keypoint 歸屬配錯了。
    """
    if scene.measurements.is_empty:
        return None

    label = LabelPlacer()
    canvas = (image * DIM_FACTOR).astype(np.uint8)
    thickness = _line_thickness(image)
    by_id = {inst.instance_id: inst for inst in scene.instances}

    for d in scene.measurements.interocular:
        inst = by_id.get(d.instance_id)
        if inst is None:
            continue
        left, right = inst.keypoint(LEFT_EYE), inst.keypoint(RIGHT_EYE)
        if left is None or right is None:
            continue
        color = color_for(d.instance_id)
        p, q = _xy(left), _xy(right)
        cv2.line(canvas, p, q, color, thickness + 1, cv2.LINE_AA)
        label(
            canvas,
            f"#{d.instance_id} {d.label} {_distance_text(d)}",
            _midpoint(p, q),
            color,
            image,
        )

    for d in scene.measurements.inter_object:
        a, b = by_id.get(d.instance_a), by_id.get(d.instance_b)
        if a is None or b is None:
            continue
        kp_a, kp_b = a.keypoint(d.keypoint_a), b.keypoint(d.keypoint_b)
        if kp_a is None or kp_b is None:
            continue
        p, q = _xy(kp_a), _xy(kp_b)
        _dashed_line(canvas, p, q, CROSS_OBJECT_COLOR, thickness)
        # dz 遠大於零時，這筆量的其實是「深度差」而不是橫向距離，誤差會由
        # 深度模型主導。診斷離群值時先看這一欄。
        gap = "" if d.depth_gap_m is None else f" dz={d.depth_gap_m:.2f}m"
        label(
            canvas,
            f"#{d.instance_a}-#{d.instance_b} {d.label_a}/{d.label_b} "
            f"{_distance_text(d)}{gap}",
            _midpoint(p, q),
            CROSS_OBJECT_COLOR,
            image,
        )

    return canvas


def _distance_text(measurement) -> str:
    """像素距離一定有，公尺距離則看有沒有跑 stage 3。"""
    parts = []
    if measurement.distance_px is not None:
        parts.append(f"{measurement.distance_px:.0f}px")
    if measurement.distance_m is not None:
        parts.append(f"{measurement.distance_m:.2f}m")
    parts.append(f"c={measurement.confidence:.2f}")
    return " ".join(parts)


def _xy(keypoint: Keypoint) -> tuple[int, int]:
    return (int(round(keypoint.point.u)), int(round(keypoint.point.v)))


def _midpoint(p: tuple[int, int], q: tuple[int, int]) -> tuple[int, int]:
    return ((p[0] + q[0]) // 2, (p[1] + q[1]) // 2)


def _dashed_line(
    canvas: np.ndarray,
    p: tuple[int, int],
    q: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    dash: int = 12,
) -> None:
    """cv2 沒有虛線，自己沿線段切段畫。"""
    length = int(np.hypot(q[0] - p[0], q[1] - p[1]))
    if length == 0:
        return
    steps = max(1, length // dash)
    for i in range(0, steps, 2):
        t0, t1 = i / steps, min((i + 1) / steps, 1.0)
        a = (int(p[0] + (q[0] - p[0]) * t0), int(p[1] + (q[1] - p[1]) * t0))
        b = (int(p[0] + (q[0] - p[0]) * t1), int(p[1] + (q[1] - p[1]) * t1))
        cv2.line(canvas, a, b, color, thickness, cv2.LINE_AA)


# -- 繪圖小工具 -------------------------------------------------------------


def save(canvas: np.ndarray, path: Path) -> Path:
    """寫出疊圖。

    這裡是整個繪圖流程唯一需要 RGB→BGR 的地方——cv2 只有在編碼時才在意
    通道順序。少了這一步，圖看起來會是藍紅顛倒的。

    用 ``imencode`` + 自己寫檔，理由與 :func:`main.load_image` 相同：
    ``imwrite`` 在 Windows 碰到非 ASCII 路徑會靜靜地寫不出來，而輸出目錄是
    跟著輸入檔名走的。
    """
    ok, encoded = cv2.imencode(path.suffix, canvas[:, :, ::-1])
    if not ok:
        raise ValueError(f"無法編碼影像：{path}")
    encoded.tofile(path)
    return path


def _scale(image: np.ndarray) -> float:
    """以 640px 為基準的縮放係數，讓線寬與字級隨影像大小調整。

    沒有這個的話，同一組參數在 COCO 的 640px 圖上剛好，在 4K 圖上會細到
    看不見——而看不見的疊圖等於沒畫。
    """
    return max(1.0, max(image.shape[:2]) / 640.0)


def _line_thickness(image: np.ndarray) -> int:
    return max(1, int(round(_scale(image) * 2)))


class LabelPlacer:
    """畫有底色的文字標籤，並避免標籤互相覆蓋。

    一律加不透明底色：白字畫在淺色動物身上、黑字畫在深色背景上都會消失，
    而讀不到的標籤等於沒有標籤。底色用該 instance 的顏色，文字顏色再依底色
    亮度挑黑或白。

    避開重疊是必要的，不是美觀問題。兩個標籤預設會疊在一起、把其中一個整個
    蓋掉——而被蓋掉的那個往往正是你要查的那個。這裡記住已經畫過的方框，
    新標籤上下交錯找到第一個不重疊的位置才落筆。

    一張疊圖共用一個 placer。之後的 stage 疊在 stage 1 之上，所以那幾層也要
    共用同一個，否則新標籤會蓋掉物種標籤。
    """

    def __init__(self) -> None:
        self._boxes: list[tuple[int, int, int, int]] = []

    def __call__(
        self,
        canvas: np.ndarray,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
        reference: np.ndarray,
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45 * _scale(reference)
        thickness = max(1, int(_scale(reference)))
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

        height, width = canvas.shape[:2]
        box_w, box_h = tw + 6, th + baseline + 4
        x = min(max(int(origin[0]), 0), max(0, width - box_w))

        left, top, right, bottom = self._find_slot(
            x, int(origin[1]), box_w, box_h, height
        )
        self._boxes.append((left, top, right, bottom))

        cv2.rectangle(canvas, (left, top), (right, bottom), color, -1)
        cv2.putText(
            canvas,
            text,
            (left + 3, bottom - baseline - 2),
            font,
            scale,
            contrast_color(color),
            thickness,
            cv2.LINE_AA,
        )

    def _find_slot(
        self, x: int, y: int, box_w: int, box_h: int, height: int
    ) -> tuple[int, int, int, int]:
        """從理想位置往下、往上交錯搜尋第一個不重疊的位置。

        全部試完都還是撞到時就用原位。寧可疊上去，也不要把標籤丟到離它所指
        的點很遠的地方——那比看不到更容易誤讀成別的物體的標籤。
        """
        step = box_h + 2
        offsets = [0]
        for i in range(1, 8):
            offsets.extend((i * step, -i * step))

        for dy in offsets:
            bottom = min(max(y + dy, box_h), height)
            candidate = (x, bottom - box_h, x + box_w, bottom)
            if not any(_intersects(candidate, placed) for placed in self._boxes):
                return candidate

        bottom = min(max(y, box_h), height)
        return (x, bottom - box_h, x + box_w, bottom)


def _intersects(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])
