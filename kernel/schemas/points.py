"""影像平面上的點。

刻意不在這裡放任何幾何運算（反投影、測距、座標系轉換）。這個模組只負責
「這個點是什麼、在哪個座標系」。

二維點來自模型輸出，三維點一律由 :func:`kernel.geometry.unproject` 反投影
產生——:class:`Point3D` 不應該由任何模型 wrapper 直接建構。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kernel.schemas.frames import CoordinateFrame


class FrameMismatchError(ValueError):
    """座標系不符。

    幾乎所有「數字看起來很合理但結果就是錯」的 bug 都會在這裡被攔下來。
    """


@dataclass(frozen=True, slots=True)
class Point2D:
    """影像平面上的一個點，單位為像素。

    允許非整數座標：heatmap 經過 soft-argmax 之後會得到子像素位置，
    而子像素精度對眼距這種小距離的影響不小，不應該在這一層被四捨五入掉。
    """

    u: float
    v: float
    frame: CoordinateFrame = CoordinateFrame.IMAGE

    def require_frame(self, expected: CoordinateFrame) -> Point2D:
        """確認此點位於 `expected` 座標系，否則丟出 FrameMismatchError。

        所有會消費座標的函式都應該在入口呼叫一次。這是本專案防止
        「模型輸入座標拿去查原圖」的主要機制。
        """
        if self.frame is not expected:
            raise FrameMismatchError(
                f"預期座標系為 {expected.value}，實際收到 {self.frame.value}。"
            )
        return self

    def as_tuple(self) -> tuple[float, float]:
        return (self.u, self.v)

    def as_array(self) -> np.ndarray:
        """回傳 shape (2,) 的 float64 陣列。"""
        return np.array([self.u, self.v], dtype=np.float64)

    @property
    def is_finite(self) -> bool:
        """座標是否為有限值（模型失敗時可能吐出 nan / inf）。"""
        return bool(np.isfinite([self.u, self.v]).all())


@dataclass(frozen=True, slots=True)
class Point3D:
    """相機座標系中的一個點，單位**公尺**。

    座標軸方向見 :attr:`CoordinateFrame.CAMERA`：+x 向右、+y 向下、+z 沿
    光軸向前，與影像的 u/v 同向。

    `frame` 只允許三維座標系。把它寫成欄位而不是寫死，是為了讓之後要加
    世界座標系時，型別會直接指出哪些呼叫端需要修改。
    """

    x: float
    y: float
    z: float
    frame: CoordinateFrame = CoordinateFrame.CAMERA

    def __post_init__(self) -> None:
        if not self.frame.is_3d:
            raise FrameMismatchError(
                f"Point3D 需要三維座標系，收到 {self.frame.value}。"
            )

    def require_frame(self, expected: CoordinateFrame) -> Point3D:
        if self.frame is not expected:
            raise FrameMismatchError(
                f"預期座標系為 {expected.value}，實際收到 {self.frame.value}。"
            )
        return self

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def as_array(self) -> np.ndarray:
        """回傳 shape (3,) 的 float64 陣列。"""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @property
    def is_finite(self) -> bool:
        return bool(np.isfinite([self.x, self.y, self.z]).all())

    @property
    def is_in_front(self) -> bool:
        """是否位於相機前方。

        z <= 0 代表反投影拿到了非正的深度，那不是「很近」而是**算錯了**。
        這種點不該進入任何距離計算。
        """
        return self.z > 0.0
