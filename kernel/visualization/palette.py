"""疊圖用的配色。

顏色一律以 **RGB** 表示，與 pipeline 內部一致。cv2 的繪圖函式不在意通道
語義，只有 ``imwrite`` 在意——所以整個繪圖過程都走 RGB，只在最後寫檔那一步
轉成 BGR（見 :func:`kernel.visualization.draw.save`）。
"""

from __future__ import annotations

#: 每個 instance 一個顏色，依 ``instance_id`` 循環取用。
#:
#: 同一隻動物在各張疊圖上必須是同一個顏色——查驗時最常做的動作就是「這一層
#: 的東西，對應 stage 1 哪一塊輪廓」，顏色一致才對得起來。所以這裡用固定
#: 順序的調色盤而不是隨機色。
INSTANCE_COLORS: tuple[tuple[int, int, int], ...] = (
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (0, 128, 128),
)


#: 量測（雙眼連線與距離標籤）專用色，不跟著 instance 跑。
#:
#: 刻意用單一顏色：量測是**跨越兩個點**的東西，用其中一端的顏色畫它會讓人
#: 以為那條線也屬於某隻動物。固定成藍色之後，「藍色 = 這是算出來的，不是
#: 模型直接吐出來的」在整份輸出裡就是一個一眼可辨的分類。
#:
#: 這個藍刻意不在 :data:`INSTANCE_COLORS` 裡面，才不會跟某一隻動物撞色。
MEASUREMENT_COLOR: tuple[int, int, int] = (0, 120, 255)


def color_for(instance_id: int) -> tuple[int, int, int]:
    """取某個 instance 的固定配色。"""
    return INSTANCE_COLORS[instance_id % len(INSTANCE_COLORS)]


def contrast_color(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """在給定底色上讀得清楚的文字顏色（黑或白）。

    用 ITU-R BT.601 的亮度權重。黃色底配白字會完全看不見，這個判斷不能省。
    """
    r, g, b = rgb
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)
