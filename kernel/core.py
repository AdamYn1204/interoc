"""pipeline 的編排層。

:meth:`InterocularCore.forward` 是整個專案的單一入口：吃一張 RGB 影像，
吐一個 :class:`Scene`。

分段降級
--------
三個階段都是可選的，缺哪一段就少一部分結果，而不是整條失敗：

    只有 stage 1        → 有輪廓，沒有眼睛，沒有距離
    stage 1 + 2         → 有眼睛，有 **像素** 距離
    stage 1 + 2 + 3     → 再加上 **公尺** 距離（需要焦距）

這樣設計是因為 stage 2 與 stage 3 各要下載數百 MB 的權重，但 stage 1 在
一般照片上馬上就能跑。像素距離不需要任何相機資訊，所以它永遠會被算出來，
同時也是驗證 metric 結果的對照組。

焦距從哪來
----------
公尺數需要焦距，但**不一定要使用者提供**：Depth Pro 會從影像自己估一個。
優先序是「呼叫端給的 > 模型估的」，實際用了哪一個記在
``Scene.depth.focal_estimated``——回頭看結果時，「這組公尺數是量的還是
估的」是判斷它能不能用的第一個問題。

兩者都沒有時（例如用不估焦距的模型又沒給焦距），深度照樣算、疊圖照樣畫，
只是不做反投影，結果退回像素距離。這比丟例外好：深度圖本身仍然有診斷價值。

量測（:mod:`kernel.geometry`）不在這個降級表裡：它是純幾何、不吃權重，
有資料就算得出來，沒有需要關掉的理由。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kernel.geometry import attach_3d, measure
from kernel.models.base import (
    DepthEstimator,
    EyeDetector,
    Segmenter,
    validate_image,
)
from kernel.schemas import (
    RIGHT_EYE,
    DepthMap,
    Instance,
    MeasurementResult,
)


@dataclass(frozen=True, slots=True)
class Scene:
    """一張影像跑完 pipeline 之後的全部結果。"""

    image_size: tuple[int, int]
    """``(height, width)``，原圖尺寸。"""

    instances: tuple[Instance, ...]

    model: str
    """實際跑了哪些模型，例如 ``"yolo11l-seg+vitpose-plus-base-ap10k"``。

    讓每筆結果自己標明它是什麼跑出來的——比對兩次實驗時不必回頭翻設定檔。
    """

    measurements: MeasurementResult = MeasurementResult()
    """由 `instances` 上的 keypoint 算出的距離。

    與 `instances` 分開存放，而不是把距離掛回 `Instance` 上：距離是**兩個
    點之間**的性質。雙眼間距的兩端剛好在同一隻動物身上，跨物體距離則不是，
    掛在單一物體上無處安放。
    """

    source: Path | None = None

    depth: DepthMap | None = None
    """完整深度圖。只在跑了 stage 3 時存在，主要供視覺化與除錯使用。"""

    @property
    def n_instances(self) -> int:
        return len(self.instances)

    @property
    def n_eyes(self) -> int:
        return sum(len(inst.eyes) for inst in self.instances)

    @property
    def is_metric(self) -> bool:
        """本次結果是否含有真實尺度的距離。

        跑了深度還不夠——沒有焦距就反投影不了，那種情況下只有像素距離。
        """
        return self.depth is not None and self.depth.focal_px is not None

    @property
    def focal_px(self) -> float | None:
        """本次使用的焦距（像素）；沒跑深度時為 None。"""
        return None if self.depth is None else self.depth.focal_px


class InterocularCore:
    """把 segmentation、eye detection、depth 三段串起來。

    三個階段都以 Protocol 注入，所以測試時可以塞假模型，不需要任何權重::

        core = InterocularCore(AnimalSegmenter("yolo11l-seg.pt"))
        core = InterocularCore(
            AnimalSegmenter("yolo11l-seg.pt"),
            eyes=ViTPoseEyes(),
            depth=DepthAnythingV2(),
            focal_px=1800.0,
        )
    """

    def __init__(
        self,
        segmenter: Segmenter,
        eyes: EyeDetector | None = None,
        depth: DepthEstimator | None = None,
        focal_px: float | None = None,
        principal_point: tuple[float, float] | None = None,
        cross_keypoint: str = RIGHT_EYE,
    ) -> None:
        if (
            depth is not None
            and focal_px is None
            and not getattr(depth, "estimates_focal", False)
        ):
            raise ValueError(
                f"{getattr(depth, 'name', depth)} 不會自己估焦距，而你也沒有給 "
                f"focal_px。反投影 (u, v, Z) → (X, Y, Z) 一定要 fx，橫向距離的"
                f"尺度完全錨定在它上面。\n"
                f"要嘛提供焦距，要嘛改用會估焦距的模型（DepthPro）。"
            )
        self.segmenter = segmenter
        self.eyes = eyes
        self.depth = depth
        self.focal_px = focal_px

        self.principal_point = principal_point
        """主點 ``(cx, cy)``；None 時取影像中心。

        影像中心是個好用的近似，一般鏡頭的主點偏移只有幾個像素，對距離的
        影響遠小於深度誤差。
        """

        self.cross_keypoint = cross_keypoint
        """跨物體配對要用哪顆眼睛，預設右眼對右眼。"""

        self.name = "+".join(
            stage.name for stage in (segmenter, eyes, depth) if stage is not None
        )

    def forward(self, image: np.ndarray, source: Path | None = None) -> Scene:
        """跑完整條 pipeline。`image` 為 HxWx3 uint8 RGB。"""
        validate_image(image)
        height, width = image.shape[:2]

        instances = tuple(self.segmenter.segment(image))

        # 沒有動物就不必叫 stage 2——ViTPose 是 top-down 的，沒有框可吃。
        if self.eyes is not None and instances:
            instances = tuple(self.eyes.detect(image, instances))

        depth_map: DepthMap | None = None
        if self.depth is not None and instances:
            # require_metric 在這裡就擋下來，而不是等到算完距離才發現單位
            # 是假的——相對深度算出來的公尺數看起來完全正常。
            depth_map = self.depth.estimate(image, self.focal_px).require_metric()

            # 焦距可能來自呼叫端，也可能是模型自己估的——一律以深度圖上帶著
            # 的那個為準，兩者才不會脫鉤（模型的絕對尺度正比於它估的焦距，
            # 拿另一個焦距去反投影會讓距離整體偏掉）。
            if depth_map.focal_px is not None:
                instances = attach_3d(
                    instances,
                    depth_map,
                    (height, width),
                    depth_map.focal_px,
                    self.principal_point,
                )

        return Scene(
            image_size=(height, width),
            instances=instances,
            model=self.name,
            measurements=measure(instances, self.cross_keypoint),
            source=source,
            depth=depth_map,
        )
