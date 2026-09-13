"""pipeline 的編排層。

:meth:`InterocularCore.forward` 是整個專案的單一入口：吃一張 RGB 影像，
吐一個 :class:`Scene`。

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kernel.models.base import Segmenter, validate_image
from kernel.schemas import Instance


@dataclass(frozen=True, slots=True)
class Scene:
    """一張影像跑完 pipeline 之後的全部結果。"""

    image_size: tuple[int, int]
    """``(height, width)``，原圖尺寸。"""

    instances: tuple[Instance, ...]

    model: str
    """實際跑了哪些模型，例如 ``"yolo11l-seg"``。

    讓每筆結果自己標明它是什麼跑出來的——比對兩次實驗時不必回頭翻設定檔。
    """

    source: Path | None = None

    @property
    def n_instances(self) -> int:
        return len(self.instances)


class InterocularCore:
    """把各個推論階段串起來。

    階段以 Protocol 注入，所以測試時可以塞假模型，不需要任何權重::

        core = InterocularCore(AnimalSegmenter("yolo11l-seg.pt"))
    """

    def __init__(self, segmenter: Segmenter) -> None:
        self.segmenter = segmenter
        self.name = segmenter.name

    def forward(self, image: np.ndarray, source: Path | None = None) -> Scene:
        """跑完整條 pipeline。`image` 為 HxWx3 uint8 RGB。"""
        validate_image(image)
        height, width = image.shape[:2]

        instances = tuple(self.segmenter.segment(image))

        return Scene(
            image_size=(height, width),
            instances=instances,
            model=self.name,
            source=source,
        )
