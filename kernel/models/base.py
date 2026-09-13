"""模型層的共同介面與共用工具。

介面用 `Protocol` 而非抽象基底類別，是為了讓實作可以自由組合、也方便在測試
裡塞假模型
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from kernel.schemas import Instance


def validate_image(image: np.ndarray) -> np.ndarray:
    """檢查輸入影像符合本專案的約定並原樣回傳。

    約定為 **HxWx3、uint8、RGB 通道順序**。

    RGB 這一點要特別留意：OpenCV 的 ``imdecode`` / ``VideoCapture`` 和
    ultralytics 的 ``predict`` 吃的都是 BGR，通道順序錯了不會報錯，只會讓
    精度默默下降。整條 pipeline 內部一律走 RGB，要轉 BGR 是各模型 wrapper
    自己在餵進框架前的責任（見 :mod:`kernel.models.segmentation`）。
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"影像必須是 numpy 陣列，收到 {type(image).__name__}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"影像必須是 HxWx3，收到 shape={image.shape}。"
            f"灰階影像請先自行複製成三通道。"
        )
    if image.dtype != np.uint8:
        raise ValueError(
            f"影像必須是 uint8（0..255），收到 {image.dtype}。"
            f"正規化是各模型前處理的責任，不要在外面先做。"
        )
    return image


@runtime_checkable
class Segmenter(Protocol):
    """Stage 1：把影像切成一個個物體。"""

    name: str
    """模型識別名，會流進輸出讓結果自己標明是什麼跑出來的。"""

    def segment(self, image: np.ndarray) -> Sequence[Instance]:
        """回傳影像中的所有物體。

        `image` 為 HxWx3 uint8 RGB。回傳的 `Instance` 必須已帶 bbox 與 mask，
        且座標一律在 ``CoordinateFrame.IMAGE``——把模型輸入座標換算回原圖是
        實作自己的責任，外界不需要知道它內部縮放到多大。
        """
        ...
