"""最終量測輸出。

這條 pipeline 對外承諾兩種距離，兩者都是 KPI：

1. :class:`InterocularDistance` —— 同一隻動物的左右眼間距，每隻各一組。
2. :class:`InterObjectEyeDistance` —— 不同動物的眼睛之間的距離（預設為
   右眼對右眼），N 隻有眼睛的動物共 N(N-1)/2 組。

兩者的誤差來源不同，所以刻意分成兩個型別而不是共用一個「兩點距離」：
同一隻動物的兩顆眼睛深度幾乎相同，橫向分量主導，誤差主要來自 keypoint
定位；跨物體的兩顆眼睛則可能差好幾公尺，誤差由深度主導。混在一起會讓
評測數字失去意義——一個 5% 的整體誤差，在前者是「還行」，在後者可能代表
深度模型根本沒抓到其中一隻。

`distance_px` 兩者都保留。它不需要深度、不需要內參，只要眼睛定位成功就
一定算得出來，所以在 metric 路徑失敗時（深度無效、焦距未知、模型吐的是
相對深度）它是唯一還能回報的數字。它同時也是驗證 metric 結果的對照組：
:attr:`InterocularDistance.meters_per_pixel` 就是拿來做那個對照的。
"""

from __future__ import annotations

from dataclasses import dataclass

from kernel.schemas.points import Point3D


@dataclass(frozen=True, slots=True)
class InterocularDistance:
    """單一動物左右眼之間的距離。

    內部一律以公尺儲存，只在展示或報表邊界換算成 mm / cm。
    """

    instance_id: int

    label: str
    """物種名稱，例如 ``"cat"``。純 metadata，不參與計算。

    帶著它是為了評測時能直接按物種切分：bird 在 ViTPose 上是已知的失效
    類別，必須單獨列，不能混進總體平均。
    """

    confidence: float
    """0..1，由兩顆眼睛的定位信心與深度取樣信心合成。

    合成公式屬於 geometry 層，這個欄位只負責攜帶結果。低信心的量測不應該
    被靜靜丟掉，而是照常輸出並讓呼叫端自行設門檻——「這次量不準」本身
    就是重要資訊。
    """

    distance_px: float | None = None
    """兩眼在原圖上的像素距離，不依賴深度與內參。

    保留小數。keypoint 是子像素精度的，而這個距離本身可能只有個位數像素
    （三公尺外的貓，雙眼間距在畫面上約 7 px），四捨五入掉的量級跟訊號
    本身同級。
    """

    distance_m: float | None = None
    """三維距離（公尺）。沒跑深度階段、或深度取樣失敗時為 None。"""

    left_eye: Point3D | None = None
    right_eye: Point3D | None = None
    """反投影後的三維眼睛位置；`distance_m` 為 None 時這兩欄也是 None。"""

    def __post_init__(self) -> None:
        _validate_distance(self.distance_m, self.confidence, self.distance_px)
        if self.distance_m is not None and (
            self.left_eye is None or self.right_eye is None
        ):
            raise ValueError(
                f"instance {self.instance_id} 有 distance_m 卻缺少三維眼睛座標；"
                f"metric 距離必須附帶它是從哪兩個點算出來的。"
            )

    @property
    def is_metric(self) -> bool:
        """是否具有真實尺度的距離。False 代表只有像素距離可用。"""
        return self.distance_m is not None

    @property
    def distance_mm(self) -> float | None:
        return None if self.distance_m is None else self.distance_m * 1000.0

    @property
    def distance_cm(self) -> float | None:
        return None if self.distance_m is None else self.distance_m * 100.0

    @property
    def meters_per_pixel(self) -> float | None:
        """眼睛所在深度處的每像素實際長度，供健全性檢查用。

        這個值應該與「該深度下由內參推得的像素尺度」吻合。兩者差一個數量級
        通常代表焦距填錯，或深度圖其實是相對深度。
        """
        if self.distance_m is None or not self.distance_px:
            return None
        return self.distance_m / self.distance_px


@dataclass(frozen=True, slots=True)
class InterObjectEyeDistance:
    """兩隻不同動物的眼睛之間的三維距離。

    預設配對是右眼對右眼，但 `keypoint_a` / `keypoint_b` 是一般化的欄位，
    要改成左對左或交叉配對都不必動型別——決定配哪些點是 geometry 層的事。

    這組量測的精度幾乎完全由深度決定：兩隻動物可能相距數公尺，而單目
    metric depth 的相對誤差會直接乘上那個距離。`confidence` 與
    `distance_px` 在這裡特別重要，`depth_gap_m` 則是診斷離群值的第一站。
    """

    instance_a: int
    label_a: str
    keypoint_a: str

    instance_b: int
    label_b: str
    keypoint_b: str

    confidence: float
    distance_px: float | None = None
    """兩顆眼睛在原圖上的像素距離，不依賴深度與內參。"""

    distance_m: float | None = None
    """三維距離（公尺）。沒跑深度階段、或深度取樣失敗時為 None。"""

    point_a: Point3D | None = None
    point_b: Point3D | None = None

    def __post_init__(self) -> None:
        if self.instance_a == self.instance_b:
            raise ValueError(
                f"InterObjectEyeDistance 用於量測不同動物之間的距離，"
                f"但兩端都是 instance {self.instance_a}。"
                f"同一隻動物的雙眼間距請用 InterocularDistance。"
            )
        _validate_distance(self.distance_m, self.confidence, self.distance_px)
        if self.distance_m is not None and (
            self.point_a is None or self.point_b is None
        ):
            raise ValueError(
                f"instance {self.instance_a}/{self.instance_b} 有 distance_m "
                f"卻缺少三維座標；metric 距離必須附帶它是從哪兩個點算出來的。"
            )

    @property
    def is_metric(self) -> bool:
        return self.distance_m is not None

    @property
    def is_cross_species(self) -> bool:
        """兩端是否為不同物種。"""
        return self.label_a != self.label_b

    @property
    def distance_mm(self) -> float | None:
        return None if self.distance_m is None else self.distance_m * 1000.0

    @property
    def distance_cm(self) -> float | None:
        return None if self.distance_m is None else self.distance_m * 100.0

    @property
    def instance_pair(self) -> tuple[int, int]:
        """兩個 instance id，經排序，方便當作查表的 key。"""
        return (
            min(self.instance_a, self.instance_b),
            max(self.instance_a, self.instance_b),
        )

    @property
    def depth_gap_m(self) -> float | None:
        """兩顆眼睛沿光軸的深度差；非 metric 時為 None。

        這個值遠大於零時，該筆量測基本上是在量「深度差」而不是「橫向距離」，
        其誤差會由深度模型的相對誤差主導。診斷離群值時先看這一欄。
        """
        if self.point_a is None or self.point_b is None:
            return None
        return abs(self.point_a.z - self.point_b.z)


@dataclass(frozen=True, slots=True)
class MeasurementResult:
    """單一影格上的全部量測結果。"""

    interocular: tuple[InterocularDistance, ...] = ()
    """每隻動物的雙眼間距。"""

    inter_object: tuple[InterObjectEyeDistance, ...] = ()
    """跨動物的眼對眼距離。"""

    @property
    def is_empty(self) -> bool:
        return not self.interocular and not self.inter_object

    @property
    def total(self) -> int:
        """兩類量測的總筆數。"""
        return len(self.interocular) + len(self.inter_object)

    def for_instance(self, instance_id: int) -> InterocularDistance | None:
        """取指定動物的雙眼間距，不存在時回傳 None。

        每隻動物最多一組——牠只有一對眼睛。
        """
        for d in self.interocular:
            if d.instance_id == instance_id:
                return d
        return None

    def by_label(self, label: str) -> tuple[InterocularDistance, ...]:
        """取某個物種的所有雙眼間距。

        評測時 bird 必須單獨列（ViTPose 在鳥身上是已知失效），這個方法
        就是給那種切分用的。
        """
        return tuple(d for d in self.interocular if d.label == label)

    def between(
        self, instance_a: int, instance_b: int
    ) -> tuple[InterObjectEyeDistance, ...]:
        """取兩隻指定動物之間的所有跨物體量測，與傳入順序無關。"""
        key = (min(instance_a, instance_b), max(instance_a, instance_b))
        return tuple(d for d in self.inter_object if d.instance_pair == key)

    def cross_species(self) -> tuple[InterObjectEyeDistance, ...]:
        """只取兩端物種不同的跨物體量測。"""
        return tuple(d for d in self.inter_object if d.is_cross_species)

    def with_min_confidence(self, threshold: float) -> MeasurementResult:
        """兩類量測一併濾掉信心低於門檻者。"""
        return MeasurementResult(
            tuple(d for d in self.interocular if d.confidence >= threshold),
            tuple(d for d in self.inter_object if d.confidence >= threshold),
        )


def _validate_distance(
    distance_m: float | None, confidence: float, distance_px: float | None
) -> None:
    if distance_m is None and distance_px is None:
        raise ValueError(
            "一筆量測至少要有一種距離；distance_m 與 distance_px 不可同時為 None。"
        )
    if distance_m is not None and distance_m < 0.0:
        raise ValueError(f"距離不可為負：{distance_m}")
    if distance_px is not None and distance_px < 0.0:
        raise ValueError(f"像素距離不可為負：{distance_px}")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence 必須落在 0..1，收到 {confidence}")
