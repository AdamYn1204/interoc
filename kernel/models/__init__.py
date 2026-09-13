"""推論階段的模型 wrapper。

    stage 1  segmentation  AnimalSegmenter  YOLO*-seg           → Instance(bbox, mask)
    stage 2  keypoint      ViTPoseEyes      ViTPose++           → Instance(+ 眼睛)
    stage 3  depth         DepthAnythingV2  Depth Anything V2   → DepthMap

介面（`Segmenter` / `EyeDetector` / `DepthEstimator`）定義在
:mod:`kernel.models.base`；具體實作放在各自的模組::

    from kernel.models.segmentation import AnimalSegmenter
    from kernel.models.keypoint import ViTPoseEyes
    from kernel.models.depth import DepthAnythingV2

三個實作的框架與失效模式都不同（ultralytics / transformers 的兩種任務），
所以各自一個模組，不要合併——權重來源、輸入格式、出錯的樣子都不一樣，混在
一起會讓除錯時很難分辨問題出在哪一層。

權重都是延遲載入的：建構一個 wrapper 不會讀權重也不吃 VRAM，要到第一次
推論才載入。
"""

from kernel.models.base import (
    DepthEstimator,
    EyeDetector,
    Segmenter,
    resolve_device,
    validate_image,
)

__all__ = [
    "DepthEstimator",
    "EyeDetector",
    "Segmenter",
    "resolve_device",
    "validate_image",
]
