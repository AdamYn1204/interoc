"""pipeline 的編排層。

:meth:`InterocularCore.forward` 是整個專案的單一入口：吃一張 RGB 影像，
吐一個 :class:`Scene`。

分段降級
--------
stage 2 是可選的，缺它就少一部分結果，而不是整條失敗：

    只有 stage 1   → 有輪廓，沒有眼睛
    stage 1 + 2    → 再加上每隻動物的眼睛位置，以及湊齊雙眼者的眼距

這樣設計是因為 stage 2 要另外下載 ViTPose 權重，而 stage 1 在一般照片上馬上
就能跑；只想確認切割品質時不必把眼睛那一段也載進來。

量測（:mod:`kernel.geometry`）不在這個降級表裡：它是純幾何、不吃權重，
有眼睛就一定算得出來，沒有需要關掉的理由。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kernel.geometry import measure_interocular
from kernel.models.base import EyeDetector, Segmenter, validate_image
from kernel.schemas import Instance, MeasurementResult


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
    點之間**的性質，掛在單一物體上只是剛好目前這一種（雙眼間距）兩端都在
    同一隻動物身上。跨物體的距離進來時無處可掛，屆時再搬就會動到 schema。
    """

    source: Path | None = None

    @property
    def n_instances(self) -> int:
        return len(self.instances)

    @property
    def n_eyes(self) -> int:
        return sum(len(inst.eyes) for inst in self.instances)


class InterocularCore:
    """把各個推論階段串起來。

    階段以 Protocol 注入，所以測試時可以塞假模型，不需要任何權重::

        core = InterocularCore(AnimalSegmenter("yolo11l-seg.pt"))
        core = InterocularCore(AnimalSegmenter("yolo11l-seg.pt"), eyes=ViTPoseEyes())
    """

    def __init__(self, segmenter: Segmenter, eyes: EyeDetector | None = None) -> None:
        self.segmenter = segmenter
        self.eyes = eyes

        self.name = "+".join(
            stage.name for stage in (segmenter, eyes) if stage is not None
        )

    def forward(self, image: np.ndarray, source: Path | None = None) -> Scene:
        """跑完整條 pipeline。`image` 為 HxWx3 uint8 RGB。"""
        validate_image(image)
        height, width = image.shape[:2]

        instances = tuple(self.segmenter.segment(image))

        # 沒有動物就不必叫 stage 2——ViTPose 是 top-down 的，沒有框可吃。
        if self.eyes is not None and instances:
            instances = tuple(self.eyes.detect(image, instances))

        return Scene(
            image_size=(height, width),
            instances=instances,
            model=self.name,
            measurements=measure_interocular(instances),
            source=source,
        )
