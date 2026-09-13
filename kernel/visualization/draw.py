"""每個 stage 一張疊圖，供人工查驗。

    output/
    ├── 000000000285_1_masks.jpg    輪廓疊色 + bbox + 物種標籤
    └── 000000000285_2_eyes.jpg     疊在 1 之上，加上眼睛位置與分數

這條 pipeline 上的失效幾乎都是**靜默**的——不丟例外、數值看起來也合理，
只是錯的：

    stage 1   多邊形柵格化歪掉、錯把背景納入輪廓
    stage 2   關鍵點索引錯位（鼻子被當成眼睛）、側臉時抓到被遮住的那顆眼

其中關鍵點索引錯位那次，就是靠肉眼看疊圖才抓到的（見
:mod:`kernel.models.keypoint` 模組開頭）。所以疊圖在這個專案不是加分項，
而是主要的除錯手段。

圖是層層疊上去的：stage 2 畫在 stage 1 之上，這樣才看得出「這顆眼睛屬於
哪一塊輪廓」。同一隻動物在各張圖上顏色固定，方便交叉比對。

所有繪圖都在 **RGB** 空間進行，只有 :func:`save` 會轉成 BGR——cv2 只有在
編碼寫檔時才在意通道順序。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from kernel.core import Scene
from kernel.visualization.palette import color_for, contrast_color

MASK_ALPHA = 0.40
STAGE_FILENAMES = {
    1: "1_masks.jpg",
    2: "2_eyes.jpg",
}

ALL_STAGES = (1, 2)


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


def draw_eyes(image: np.ndarray, scene: Scene) -> np.ndarray | None:
    """在 stage 1 之上標出眼睛。沒有任何眼睛時回傳 None。

    疊在輪廓上而不是畫在原圖，是為了讓「這顆眼睛屬於哪隻動物」一眼看得出來
    ——歸屬配錯是這一層最典型的失效。

    十字準星的中心是關鍵點的真實子像素位置。畫圓圈的話中心會被圓心取整
    掩蓋掉，而子像素精度直接影響最終的距離誤差。
    """
    if scene.n_eyes == 0:
        return None

    label = LabelPlacer()
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
