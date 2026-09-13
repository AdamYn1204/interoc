"""跨階段共用的資料契約。

模型 wrapper 各有各的輸出格式，講的話都不一樣；先在這裡把型別、單位與
座標系約定死，上層才不必在每個交界重新確認一次格式。這個 package 不做
任何運算，只定義「資料長什麼樣子」。

目前只有 stage 1（segmentation），所以契約也只涵蓋 bbox 與 mask。
"""

from kernel.schemas.frames import CoordinateFrame
from kernel.schemas.instance import BBox, Instance, InstanceMask

__all__ = [
    "BBox",
    "CoordinateFrame",
    "Instance",
    "InstanceMask",
]
