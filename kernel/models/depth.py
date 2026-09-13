"""Stage 3：單目 metric 深度。

這一層要解決的是「像素距離換不成公尺」，而換算需要兩樣東西：**每像素的
深度**，以及**焦距**。少任何一個都算不出橫向距離。

兩個實作，差別在焦距
--------------------
    DepthPro          深度 + 焦距，一次推論都給   ← 預設
    DepthAnythingV2   只給深度，焦距要外部提供

:class:`DepthPro`（Apple）會從影像本身估出視角，換算成焦距，所以整條路
不需要 EXIF、不需要校正、也不需要猜。這對這個專案是關鍵：COCO 的影像
EXIF 被剝得一乾二淨（實測 `testsample/` 8 張全部 ``tags=0``），沒有任何
相機資訊可讀。

尺度正比於焦距，所以焦距能覆寫
------------------------------
Depth Pro 的後處理是（見 ``post_process_depth_estimation``）::

    focal = 0.5 * W / tan(FOV / 2)
    depth = focal / (inverse_depth_raw * W)

也就是輸出的公尺數**正比於焦距**。所以真實焦距已知時，不必重跑推論，
直接等比例修正即可::

    depth_true = depth_model * (focal_true / focal_model)

:meth:`DepthPro.estimate` 的 `focal_px` 參數走的就是這條路。這同時說明了
為什麼焦距估錯會讓所有距離整體偏掉，而畫面看起來完全正常。

metric 與相對，是兩個不同的 checkpoint
---------------------------------------
Depth Anything V2 有兩條產線，**架構相同、輸出語義完全不同**：

    相對   ``depth-anything/Depth-Anything-V2-Large-hf``
           → affine-invariant 的倒數深度，沒有單位
    metric ``depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf``
           → 公尺，在 VKITTI（戶外）或 Hypersim（室內）上微調過

兩者用同一個類別載入、輸出同一種 tensor，**不會有任何錯誤訊息告訴你載錯
了**。所以靠 :data:`METRIC_MODELS` 白名單判定，並把結論寫進
``DepthMap.is_metric`` 交給下游的閘門去擋。
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Final

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from kernel.models.base import resolve_device, validate_image
from kernel.schemas import DepthMap

#: Depth Pro 的權重。深度與焦距一次給齊。
DEPTH_PRO_MODEL: Final = "apple/DepthPro-hf"

#: Depth Anything V2 中輸出為公尺的 checkpoint。不在名單上的一律當成相對深度。
#:
#: 白名單而非黑名單：漏判成「相對」只會讓量測拒絕執行並明確報錯，漏判成
#: 「metric」則會靜靜輸出錯誤的公尺數。兩種錯的代價差很多。
METRIC_MODELS: Final = frozenset(
    {
        f"depth-anything/Depth-Anything-V2-Metric-{domain}-{size}-hf"
        for domain in ("Indoor", "Outdoor")
        for size in ("Small", "Base", "Large")
    }
)

#: 預設用 Depth Pro：它不需要任何相機資訊就能給出公尺數。
DEFAULT_MODEL: Final = DEPTH_PRO_MODEL

#: 輸出深度的上限（公尺）。超過這個距離的預測沒有實際意義，截掉可以避免
#: 天空等無窮遠區域的極端值污染後續統計。
MAX_DEPTH_M: Final = 300.0


@lru_cache(maxsize=2)
def load_depth_model(model_id: str, device: str):
    """載入一次就共用。理由同 :func:`kernel.models.segmentation.load_yolo`。

    ``device`` 也是快取鍵：同一份權重放在 CPU 和 CUDA 上是兩個不同的東西。
    """
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.to(device).eval()
    return processor, model


class DepthPro:
    """Apple Depth Pro：深度與焦距一次推論都給。

    滿足 :class:`kernel.models.base.DepthEstimator`。權重在第一次
    :meth:`estimate` 時才載入，建構本身不吃 VRAM。

    估出來的焦距是**估計值**，不是量測值。它比盲猜一個視角可靠得多，但仍然
    應該被驗證——最便宜的驗證方法是拿畫面中體型已知的動物反推焦距來對照
    （`fx = Z · d_px / D_true`）。
    """

    estimates_focal = True

    def __init__(
        self,
        model: str = DEPTH_PRO_MODEL,
        device: str | None = None,
    ) -> None:
        self.model_id = model
        self._device = device
        self._model = None
        self._processor = None
        self.name = model.rsplit("/", 1)[-1]

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = resolve_device(None)
        return self._device

    def _load(self):
        if self._model is None:
            self._processor, self._model = load_depth_model(self.model_id, self.device)
        return self._processor, self._model

    def estimate(self, image: np.ndarray, focal_px: float | None = None) -> DepthMap:
        """回傳 metric 深度圖，`focal_px` 掛在 `DepthMap.focal_px` 上。

        給了 `focal_px` 就用它覆寫模型的估計值，並等比例修正深度（推導見
        模組開頭）；沒給就用模型自己估的。
        """
        validate_image(image)
        processor, model = self._load()
        height, width = image.shape[:2]

        inputs = processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)

        # target_sizes 一定要給：post_process 只有在知道目標尺寸時才會把
        # FOV 換算成焦距，也才會據以還原尺度。少了它拿到的是未定標的數字。
        result = processor.post_process_depth_estimation(
            outputs, target_sizes=[(height, width)]
        )[0]

        estimated = result.get("focal_length")
        if estimated is None:
            raise ValueError(
                f"{self.model_id} 沒有回傳焦距（field_of_view 為 None）。"
                f"這個 checkpoint 的 FOV head 沒有啟用，拿不到尺度——"
                f"請改用會估焦距的權重，或改用 DepthAnythingV2 並自行提供 --focal。"
            )

        depth = result["predicted_depth"].cpu().numpy().astype(np.float32)
        focal_model = float(estimated)

        # 尺度正比於焦距，所以覆寫時不必重跑推論，等比例修正即可。
        if focal_px is not None:
            depth = depth * (focal_px / focal_model)

        return DepthMap(
            np.clip(depth, 0.0, MAX_DEPTH_M),
            is_metric=True,
            focal_px=focal_px if focal_px is not None else focal_model,
            focal_estimated=focal_px is None,
        )


class DepthAnythingV2:
    """Depth Anything V2，滿足 :class:`kernel.models.base.DepthEstimator`。

    **不估焦距**，所以呼叫端必須提供，否則下游只能走像素路徑。留著它是
    為了當 :class:`DepthPro` 的對照組：兩個模型對同一張圖給出的深度差多少，
    是判斷「這次的公尺數能不能信」最直接的證據。

    載入相對深度的 checkpoint 不會報錯——那是合法用途（疊圖、除錯、看誰前
    誰後）。擋下來的是拿它去算公尺數，那發生在
    :meth:`kernel.schemas.DepthMap.require_metric`。
    """

    estimates_focal = False

    def __init__(
        self,
        model: str = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        device: str | None = None,
    ) -> None:
        self.model_id = model
        self.is_metric = model in METRIC_MODELS
        self._device = device
        self._model = None
        self._processor = None
        self.name = model.rsplit("/", 1)[-1]

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = resolve_device(None)
        return self._device

    def _load(self):
        if self._model is None:
            self._processor, self._model = load_depth_model(self.model_id, self.device)
        return self._processor, self._model

    def estimate(self, image: np.ndarray, focal_px: float | None = None) -> DepthMap:
        """回傳深度圖。`focal_px` 原樣掛上去，這個模型自己用不到。"""
        validate_image(image)
        processor, model = self._load()
        height, width = image.shape[:2]

        inputs = processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)

        # processor 內部會 resize，所以預測結果不是原圖尺寸，一定要還原。
        # 少了這一步，keypoint 的座標會拿去索引一張尺寸不同的圖——不會報錯，
        # 只會讀到錯的位置。
        depth = torch.nn.functional.interpolate(
            outputs.predicted_depth[:, None],
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        data = depth.cpu().numpy().astype(np.float32)
        if self.is_metric:
            data = np.clip(data, 0.0, MAX_DEPTH_M)

        return DepthMap(
            data,
            is_metric=self.is_metric,
            focal_px=focal_px,
            focal_estimated=False,
        )


def focal_from_fov(fov_deg: float, width: int) -> float:
    """由水平視角推算焦距（像素）。

    這是**假設值**，不是量測值，只在沒有更好的來源時才用。真實鏡頭的水平
    視角從廣角的 100° 到望遠的 10° 都有，猜錯 10° 大約就是 20% 的距離誤差，
    而畫面上完全看不出來。
    """
    return (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
