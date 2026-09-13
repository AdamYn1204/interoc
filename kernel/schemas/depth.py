"""深度圖。

由 depth 模型產出，被 :func:`kernel.geometry.sample_depth` 消費。放在
schemas 而不是 models 底下，是為了讓下游不必 import 整個 models package
（那會把 torch 一起拉進來）。

`is_metric` 不是註記，是閘門
----------------------------
單目深度模型分兩種，輸出長得一模一樣：

    相對深度   Depth Anything V2（原版）、未經校正的 MiDaS
               → 無尺度的視差／倒數深度，數值大小只有相對意義
    metric     Depth Anything V2 Metric、Metric3D v2
               → 公尺

把相對深度餵進距離計算，會得到看似合理、實際上完全沒有物理意義的公尺數，
而且畫面上完全看不出異常。所以 :meth:`DepthMap.require_metric` 存在，而且
任何會產出公尺數的路徑都必須先過它。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kernel.schemas.frames import CoordinateFrame
from kernel.schemas.points import Point2D


@dataclass(frozen=True, slots=True, eq=False)
class DepthMap:
    """每像素深度值。`is_metric` 為真時單位為公尺。

    `data` 是 **z-depth**（沿光軸的距離），不是到光心的射線長度。兩者在畫面
    邊緣可以差到數個百分點，而反投影公式假設的是前者——模型 wrapper 有責任
    在這裡就轉換完畢。
    """

    data: np.ndarray
    frame: CoordinateFrame = CoordinateFrame.IMAGE
    is_metric: bool = True

    focal_px: float | None = None
    """這張深度圖所對應的水平焦距（像素）。

    焦距跟著深度圖走，而不是另外傳一個參數，因為對 Depth Pro 這類模型來說
    兩者是**同一次推論的產物**：它的絕對尺度直接正比於它估出來的焦距，拆開
    傳很容易變成「深度用 A 的、焦距用 B 的」——那算出來的距離會整體偏掉，
    而且看不出來。

    不估焦距的模型（Depth Anything V2）在這裡放呼叫端給的值，或 None。
    """

    focal_estimated: bool = False
    """`focal_px` 是模型估的（True）還是外部給的（False）。

    這個旗標要一路帶到 JSON：回頭看結果時，「這組公尺數是量的還是猜的」
    是判斷它能不能用的第一個問題。
    """

    valid_mask: np.ndarray | None = None
    """模型自行標記的有效區域；None 表示未提供，此時僅以深度值本身判斷。"""

    def __post_init__(self) -> None:
        if not self.frame.is_2d:
            raise ValueError(f"DepthMap 需要二維座標系，收到 {self.frame!r}")
        if self.data.ndim != 2:
            raise ValueError(
                f"depth map 必須是二維 HxW 陣列，收到 shape={self.data.shape}"
            )
        if not np.issubdtype(self.data.dtype, np.floating):
            raise ValueError(f"depth map 必須是浮點 dtype，收到 {self.data.dtype}")
        if self.valid_mask is not None:
            if self.valid_mask.shape != self.data.shape:
                raise ValueError(
                    f"valid_mask shape {self.valid_mask.shape} 與 depth "
                    f"shape {self.data.shape} 不符"
                )
            if self.valid_mask.dtype != np.bool_:
                raise ValueError(
                    f"valid_mask 必須是 bool dtype，收到 {self.valid_mask.dtype}"
                )

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def validity(self) -> np.ndarray:
        """實際可用的像素：深度為正、有限，且未被 valid_mask 排除。"""
        ok = np.isfinite(self.data) & (self.data > 0.0)
        if self.valid_mask is not None:
            ok &= self.valid_mask
        return ok

    def require_metric(self) -> DepthMap:
        """確認此深度圖具有真實尺度，否則拋錯。見模組開頭。"""
        if not self.is_metric:
            raise ValueError(
                "此深度圖為相對深度（is_metric=False），不可用於量測。"
                "請改用 metric 深度模型（例如 Depth Anything V2 的 Metric "
                "checkpoint），或先完成尺度校正。"
            )
        return self

    def raw_at(self, point: Point2D) -> float | None:
        """最近鄰讀取單一像素的深度值，超出範圍或無效時回傳 None。

        **不要用這個做量測。** 眼睛常落在物體邊緣，而單目深度在邊界最不穩，
        單點取值很容易讀到背景的深度。量測請走
        :func:`kernel.geometry.sample_depth`，它會用 instance mask 限制取樣
        範圍並做離群剔除。這個方法只適合除錯與視覺化。
        """
        point.require_frame(self.frame)
        u, v = int(round(point.u)), int(round(point.v))
        if not (0 <= u < self.width and 0 <= v < self.height):
            return None
        value = float(self.data[v, u])
        if not np.isfinite(value) or value <= 0.0:
            return None
        if self.valid_mask is not None and not self.valid_mask[v, u]:
            return None
        return value
