"""座標系定義。

這個檔案存在的理由只有一個：這條 pipeline 裡同一組 (u, v) 數字可能屬於好幾個
完全不同的座標系，而它們長得一模一樣。把座標系當成型別的一部分帶著走，
才能在組裝階段就把「拿模型輸入座標去查原圖」這類錯誤擋下來。

stage 2 雖然是 top-down、實際上會裁切，但 ViTPose 的 processor 已經把座標
換算回原圖才回傳，CROP 不會外流到這一層，所以仍然不定義。
"""

from __future__ import annotations

from enum import StrEnum

#: 本專案內部一律以公尺表示長度，只在展示與報表邊界換算成 mm / cm。
#:
#: 這是個文件用的常數，不參與計算。它存在是因為「單位在哪裡換算」這件事
#: 一旦沒寫死，就會有人在中間層偷偷乘 1000，而那種錯誤數字看起來完全正常。
LENGTH_UNIT = "meter"


class CoordinateFrame(StrEnum):
    """一個座標值所屬的參考系。"""

    IMAGE = "image"
    """原始輸入影像的像素座標。

    原點在左上角，+u 向右、+v 向下，單位為像素且未正規化。
    這是 pipeline 的正規座標系：所有模型輸出最終都要換算回這裡才能互相比對。
    """

    MODEL_INPUT = "model_input"
    """模型輸入張量的像素座標。

    影像餵進模型前會經過 resize / letterbox，此時座標與 IMAGE 之間差了一個
    縮放與位移。各個模型的輸入尺寸不同，所以「MODEL_INPUT」永遠要搭配是
    哪一個模型才有意義，不可跨模型直接比較。
    """

    CAMERA = "camera"
    """相機三維座標，單位公尺。

    原點在光心，+x 向右、+y 向下、+z 沿光軸向前——與 IMAGE 的 u/v 方向一致，
    所以反投影不需要翻轉任何軸。

    這是唯一的三維座標系。世界座標系（多視角或有外參時才有意義）目前不存在，
    也就沒有「這個三維點是相對誰」的歧義。
    """

    @property
    def is_2d(self) -> bool:
        return self in (CoordinateFrame.IMAGE, CoordinateFrame.MODEL_INPUT)

    @property
    def is_3d(self) -> bool:
        return self is CoordinateFrame.CAMERA
