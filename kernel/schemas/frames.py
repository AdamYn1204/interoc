"""座標系定義。

這個檔案存在的理由只有一個：這條 pipeline 裡同一組 (u, v) 數字可能屬於好幾個
完全不同的座標系，而它們長得一模一樣。把座標系當成型別的一部分帶著走，
才能在組裝階段就把「拿模型輸入座標去查原圖」這類錯誤擋下來。

目前只有 stage 1，所以只定義原圖與模型輸入兩個座標系。CROP（top-down
keypoint 的裁切區）與 CAMERA（三維）要等後面兩個階段進來才有意義。
"""

from __future__ import annotations

from enum import StrEnum


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
