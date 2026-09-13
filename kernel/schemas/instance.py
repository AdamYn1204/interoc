"""偵測到的物體實例，以及組成它的 bbox 與 mask。

這些型別採「漸進填充」設計：pipeline 每跑完一個 stage，就用 `with_*` 方法
產生一個帶有新資訊的新物件（全部 frozen，不就地修改）。這樣任一階段的
中間結果都可以留下來除錯，而且不會有某個模組偷改到別人資料的問題。

    seg → Instance(bbox, mask)

目前只有 stage 1，所以 Instance 身上只掛得住 bbox 與 mask；keypoint 與
depth 的欄位等那兩個階段進來再加。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from kernel.schemas.frames import CoordinateFrame


@dataclass(frozen=True, slots=True)
class BBox:
    """軸對齊邊界框。

    格式固定為 **xyxy、絕對像素、未正規化**。整個 codebase 不接受其他格式，
    任何 xywh / cxcywh / normalized 的模型輸出都必須在 model wrapper 裡就
    轉換完畢。這個限制是刻意的：bbox 格式混用是這類專案最常見的錯誤來源。
    """

    x1: float
    y1: float
    x2: float
    y2: float
    frame: CoordinateFrame = CoordinateFrame.IMAGE

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(
                f"BBox 座標顛倒：({self.x1}, {self.y1}) → ({self.x2}, {self.y2})。"
                f"預期 x1 <= x2 且 y1 <= y2。"
            )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True, slots=True, eq=False)
class InstanceMask:
    """單一 instance 的二值分割遮罩。

    `data` 是與所屬座標系完整影像同尺寸的布林陣列（shape 為 HxW），而不是
    只涵蓋 bbox 的 ROI。這是為了讓下游可以直接拿它去索引整張圖，不必再
    處理 offset。物體數量很多時這樣較耗記憶體，屆時可另外加上 RLE 或 ROI
    表示法，但正確性優先。
    """

    data: np.ndarray
    frame: CoordinateFrame = CoordinateFrame.IMAGE

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError(
                f"mask 必須是二維 HxW 陣列，收到 shape={self.data.shape}"
            )
        if self.data.dtype != np.bool_:
            raise ValueError(
                f"mask 必須是 bool dtype，收到 {self.data.dtype}。"
                f"若手上是 logits 或 0/1 陣列，請改用 InstanceMask.from_array()。"
            )

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        threshold: float = 0.5,
        frame: CoordinateFrame = CoordinateFrame.IMAGE,
    ) -> InstanceMask:
        """由機率圖、logits 或 0/1 陣列建構，一律二值化成 bool。"""
        return cls(np.asarray(array) > threshold, frame)

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def area(self) -> int:
        """遮罩內的像素數。"""
        return int(self.data.sum())

    @property
    def is_empty(self) -> bool:
        return self.area == 0


@dataclass(frozen=True, slots=True)
class Instance:
    """一個被偵測到的物體，以及掛在它身上的所有資訊。"""

    instance_id: int
    label: str
    score: float
    bbox: BBox
    mask: InstanceMask | None = None

    def with_mask(self, mask: InstanceMask) -> Instance:
        return replace(self, mask=mask)
