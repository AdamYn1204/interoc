"""推論階段的模型 wrapper。

    stage 1  segmentation  AnimalSegmenter  YOLO*-seg  → Instance(bbox, mask)

介面（`Segmenter`）定義在 :mod:`kernel.models.base`；具體實作放在各自的
模組::

    from kernel.models.segmentation import AnimalSegmenter

權重都是延遲載入的：建構一個 wrapper 不會讀權重也不吃 VRAM，要到第一次
推論才載入。
"""

from kernel.models.base import Segmenter, validate_image

__all__ = [
    "Segmenter",
    "validate_image",
]
