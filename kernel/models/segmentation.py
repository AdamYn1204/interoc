"""Stage 1：用 YOLO*-seg checkpoint 切出畫面中的動物。

權重是建構子參數，所以同一個類別同時涵蓋 YOLO11-seg 與 YOLO26-seg——兩者
共用 ultralytics 介面，差別只在權重檔::

    AnimalSegmenter("yolo11l-seg.pt")
    AnimalSegmenter("yolo26l-seg.pt")

權重在第一次推論時才載入，所以建構一個 segmenter 本身不吃 VRAM。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from ultralytics import YOLO

from kernel.models.base import validate_image
from kernel.schemas import BBox, Instance, InstanceMask

# COCO 的十個動物類別。推論結果只帶類別名稱、不帶 supercategory，所以這裡
# 直接列出來；若資料集端的動物清單有變動，兩邊要一起改。
COCO_ANIMALS = frozenset(
    {
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    }
)


@lru_cache(maxsize=4)
def load_yolo(weights: str):
    """載入一次就共用。

    上層可能為了讓 ``conf`` / ``device`` 逐次不同而每個請求都建一個新的
    :class:`AnimalSegmenter`；沒有這層快取的話，每個請求都要從磁碟重讀一份
    數十 MB 的權重。
    """
    return YOLO(weights)


class AnimalSegmenter:
    """切出影像中所有的 COCO 動物，滿足 :class:`kernel.models.base.Segmenter`。

    權重在第一次呼叫 :meth:`segment` 時才載入，不是在建構時。
    """

    def __init__(
        self,
        weights: str = "yolo11l-seg.pt",
        conf: float = 0.25,
        device: str | int | None = None,
    ) -> None:
        self.weights = weights
        self.conf = conf
        self.device = device
        self.name = Path(weights).stem
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = load_yolo(self.weights)
        return self._model

    def segment(self, image: np.ndarray) -> Sequence[Instance]:
        validate_image(image)
        height, width = image.shape[:2]

        # ultralytics 把傳入的原始陣列當 BGR 解讀，而本專案內部一律走 RGB。
        bgr = np.ascontiguousarray(image[:, :, ::-1])
        result = self.model.predict(
            bgr, conf=self.conf, device=self.device, verbose=False
        )[0]

        if result.masks is None:
            return ()

        instances: list[Instance] = []
        for i, box in enumerate(result.boxes):
            label = result.names[int(box.cls)]
            if label not in COCO_ANIMALS:
                continue

            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            # masks.xy 已經是原圖像素座標的多邊形；masks.data 則還停在
            # letterbox 後的推論解析度，要自己反算補邊與縮放，容易出錯。
            polygon = result.masks.xy[i]
            instances.append(
                Instance(
                    # instance_id 依序重編，而不是沿用 YOLO 的列索引——被
                    # 過濾掉的非動物類別會讓原索引出現空洞。
                    instance_id=len(instances),
                    label=label,
                    score=float(box.conf),
                    bbox=BBox(x1, y1, x2, y2),
                    mask=_rasterize(polygon, height, width),
                )
            )
        return tuple(instances)


def _rasterize(polygon: np.ndarray, height: int, width: int) -> InstanceMask:
    """把多邊形輪廓填成全圖尺寸的布林遮罩。

    深度取樣要拿遮罩直接索引 depth map，用全圖尺寸的布林陣列最省事，不必
    再處理 offset。輪廓本身若之後要回傳給前端，用 ``cv2.findContours`` 從
    遮罩反推即可，不需要兩份表示法同時存在。
    """
    canvas = np.zeros((height, width), dtype=np.uint8)
    if len(polygon) >= 3:
        cv2.fillPoly(canvas, [np.asarray(polygon, dtype=np.int32)], color=1)
    return InstanceMask(canvas.astype(bool))
