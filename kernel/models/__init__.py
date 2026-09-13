"""推論階段的模型 wrapper。

    stage 1  segmentation  AnimalSegmenter  YOLO*-seg  → Instance(bbox, mask)
    stage 2  keypoint      ViTPoseEyes      ViTPose++  → Instance(+ 眼睛)

介面（`Segmenter` / `EyeDetector`）定義在 :mod:`kernel.models.base`；具體實作
放在各自的模組::

    from kernel.models.segmentation import AnimalSegmenter
    from kernel.models.keypoint import ViTPoseEyes

兩個實作分屬不同框架（ultralytics / transformers），所以各自一個模組，不要
合併——它們的權重來源、輸入格式與失效模式都不一樣，混在一起會讓除錯時很難
分辨問題出在哪一層。

權重都是延遲載入的：建構一個 wrapper 不會讀權重也不吃 VRAM，要到第一次
推論才載入。
"""

from kernel.models.base import (
    EyeDetector,
    Segmenter,
    resolve_device,
    validate_image,
)

__all__ = [
    "EyeDetector",
    "Segmenter",
    "resolve_device",
    "validate_image",
]
