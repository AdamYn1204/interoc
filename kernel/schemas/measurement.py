"""最終量測輸出。

目前只有一種距離：:class:`InterocularDistance`——同一隻動物左右眼之間的
間距，每隻各一組。跨動物的眼對眼距離要等 depth 階段進來才有意義，屆時
另開一個型別，不要把兩者塞進同一個「兩點距離」：同一隻動物的兩顆眼睛
深度幾乎相同，誤差主要來自 keypoint 定位；跨物體的兩顆眼睛則可能差好幾
公尺，誤差由深度主導。混在一起會讓評測數字失去意義。

為什麼像素距離值得單獨存在
--------------------------
`distance_px` 不需要深度、也不需要相機內參，只要兩顆眼睛都定位成功就一定
算得出來。之後 metric 路徑上線了它也會保留：深度無效或焦距未知時它是唯一
還能回報的數字，同時也是驗證公尺數的對照組。

這個型別刻意不預留 `distance_m` 之類的 None 欄位。留了的話下游會寫出
「先讀 distance_m、是 None 再退回 distance_px」的分支，而在 depth 進來
之前那個分支永遠走後者——一段從來沒被執行過的程式碼，等到真的有深度了
也不會有人記得它需要驗證。等那個階段到了再加，屆時型別會直接告訴你哪些
呼叫端需要修改。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InterocularDistance:
    """單一動物左右眼之間的像素距離。

    每隻動物最多一組——牠只有一對眼睛。只定位到一顆眼睛的動物不會產生
    這個型別的實例（見 :func:`kernel.geometry.measure_interocular`）。
    """

    instance_id: int

    label: str
    """物種名稱，例如 ``"cat"``。純 metadata，不參與計算。

    帶著它是為了評測時能直接按物種切分：bird 在 ViTPose 上是已知的失效
    類別，必須單獨列，不能混進總體平均。
    """

    distance_px: float
    """兩眼在**原圖**座標上的歐氏距離，單位為像素。

    保留小數。keypoint 是子像素精度的，而這個距離本身可能只有個位數像素
    （3 公尺外的貓，雙眼間距在畫面上約 7 px），四捨五入掉的量級跟訊號
    本身同級。
    """

    confidence: float
    """0..1，目前為兩顆眼睛定位信心中**較低**的那個。

    取較低者而非平均：一組量測的好壞由較差的那個端點決定。取平均會讓
    「一顆 0.90、一顆 0.31」看起來像 0.60，掩蓋掉其中一顆其實只是剛好
    擦過門檻。合成公式屬於 geometry 層，這個欄位只負責攜帶結果。
    """

    def __post_init__(self) -> None:
        if self.distance_px < 0.0:
            raise ValueError(f"像素距離不可為負：{self.distance_px}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence 必須落在 0..1，收到 {self.confidence}")


@dataclass(frozen=True, slots=True)
class MeasurementResult:
    """單一影格上的全部量測結果。"""

    interocular: tuple[InterocularDistance, ...] = ()
    """每隻動物的雙眼間距。沒有任何動物湊齊雙眼時為空。"""

    @property
    def is_empty(self) -> bool:
        return not self.interocular

    @property
    def total(self) -> int:
        return len(self.interocular)

    def for_instance(self, instance_id: int) -> InterocularDistance | None:
        """取指定動物的雙眼間距，不存在時回傳 None。"""
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

    def with_min_confidence(self, threshold: float) -> MeasurementResult:
        """濾掉信心低於門檻的量測。

        刻意不在產生量測時就濾——「這次量不準」本身是重要資訊，尤其鳥類
        是已知失效類別。門檻留給呼叫端自己設。
        """
        return MeasurementResult(
            tuple(d for d in self.interocular if d.confidence >= threshold)
        )
