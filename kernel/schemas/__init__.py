"""跨階段共用的資料契約。

模型 wrapper 各有各的輸出格式，講的話都不一樣；先在這裡把型別、單位與
座標系約定死，上層才不必在每個交界重新確認一次格式。這個 package 不做
任何運算，只定義「資料長什麼樣子」。

涵蓋 stage 1（bbox、mask）、stage 2（keypoint）、stage 3（深度與三維座標），
以及由它們算出的量測。長度單位一律公尺，見 :data:`LENGTH_UNIT`。
"""

from kernel.schemas.depth import DepthMap
from kernel.schemas.frames import LENGTH_UNIT, CoordinateFrame
from kernel.schemas.instance import (
    LEFT_EYE,
    RIGHT_EYE,
    BBox,
    Instance,
    InstanceMask,
    Keypoint,
)
from kernel.schemas.measurement import (
    InterObjectEyeDistance,
    InterocularDistance,
    MeasurementResult,
)
from kernel.schemas.points import FrameMismatchError, Point2D, Point3D

__all__ = [
    "LEFT_EYE",
    "LENGTH_UNIT",
    "RIGHT_EYE",
    "BBox",
    "CoordinateFrame",
    "DepthMap",
    "FrameMismatchError",
    "Instance",
    "InstanceMask",
    "InterObjectEyeDistance",
    "InterocularDistance",
    "Keypoint",
    "MeasurementResult",
    "Point2D",
    "Point3D",
]
