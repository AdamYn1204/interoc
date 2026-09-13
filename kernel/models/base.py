"""模型層的共同介面與共用工具。

介面用 `Protocol` 而非抽象基底類別，是為了讓實作可以自由組合、也方便在測試
裡塞假模型
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import torch

from kernel.schemas import DepthMap, Instance


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


def resolve_device(device: str | None) -> str:
    """把 None 解析成實際可用的裝置。"""
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


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


@runtime_checkable
class EyeDetector(Protocol):
    """Stage 2：在既有的物體上找出眼睛。"""

    name: str

    def detect(
        self, image: np.ndarray, instances: Sequence[Instance]
    ) -> Sequence[Instance]:
        """回傳掛上眼睛 keypoint 的 instances。

        回傳的序列**必須與傳入的 instances 一一對應且順序相同**，沒偵測到
        眼睛的那些則原樣回傳。這樣呼叫端不需要另外做歸屬比對。

        眼睛與物體的歸屬關係是結構性的——keypoint 直接掛在它所屬的
        `Instance` 上。top-down 模型本來就知道每顆關鍵點是從哪個框推論出來
        的，這個資訊不應該在回傳時被攤平掉；一旦攤平就只能靠框中心距離
        重新猜，而兩隻動物的框一重疊就會猜錯（眼睛在框的頂端，比的卻是框
        中心，體型大的那隻會把鄰居的眼睛整組搶走）。
        """
        ...


@runtime_checkable
class DepthEstimator(Protocol):
    """Stage 3：估計每像素的深度。"""

    name: str

    estimates_focal: bool
    """這個模型會不會自己估出焦距。

    決定了「沒給 focal 時能不能跑」：不會估的模型少了焦距就反投影不了，
    應該在建構 pipeline 當下就報錯，而不是跑完一整輪才發現沒有公尺數。
    """

    def estimate(self, image: np.ndarray, focal_px: float | None) -> DepthMap:
        """回傳與輸入影像同尺寸的深度圖，焦距一併掛在 `DepthMap.focal_px`。

        `focal_px` 是呼叫端已知的水平焦距（像素），**語義是覆寫**：

        * 給了 → 用它，並據以修正模型的尺度（會估焦距的模型，其絕對尺度
          正比於焦距，換一個焦距就要等比例重算深度）。
        * 沒給 → 由模型自己估（`estimates_focal` 為真時），或回傳 None 焦距
          的深度圖，此時下游只能走像素路徑。

        回傳的 `DepthMap.is_metric` 必須誠實：拿相對深度去冒充公尺，下游的
        距離看起來完全正常但毫無物理意義。
        """
        ...
