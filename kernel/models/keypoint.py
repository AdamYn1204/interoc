"""Stage 2：從 stage 1 的動物上找出眼睛。

用 ViTPose++（HF ``transformers``）。它是 top-down 模型——吃的正是 stage 1
已經產出的 bbox，兩層天然對接，不需要自己裁切、也不需要自己把座標換算回
原圖。

輸出的是 COCO 的人體 17 點，不是 AP-10K
---------------------------------------
這裡踩過一次坑。``dataset_index`` 切換的是**骨幹的 MoE 專家**（這個
checkpoint 有 6 個），很容易以為「選 AP-10K 專家就會拿到 AP-10K 的關鍵點
順序」。**不會。** 匯出到 HF 的這個 checkpoint 只帶了 COCO 那顆輸出頭——
``config.id2label`` 是 ``0=Nose, 1=L_Eye, 2=R_Eye, 3=L_Ear …``，換專家改變的
是特徵、不是輸出通道的語義。

而 COCO-human 與 AP-10K 的前三點剛好錯開一位：

    index      COCO-human（實際拿到的）     AP-10K
      0        Nose                        L_Eye
      1        L_Eye                       R_Eye
      2        R_Eye                       Nose

照 AP-10K 的順序取 index 0、1 會得到「鼻子 + 一隻眼睛」，而且**不會報錯**，
只會靜默給出錯的座標——這個 bug 是靠肉眼看疊圖才抓到的。所以眼睛是
index 1 和 2，見 :data:`EYE_KEYPOINTS`：索引與名稱的對應只有那一份，疊圖與
JSON 都從 keypoint 自己的 ``name`` 拿，不要在別處另外寫死一份對照表。

鳥的限制
--------
實測過：鳥的關鍵點會擠成一團落在頭部以外，分數約 0.27–0.54。人體與四足的
骨架都沒有鳥，這是預期內的失效。``min_score`` 濾得掉大部分，但評測時 bird
層仍要單獨列——這個專案的資料裡 bird 有 396 個實例、是最大宗
（1694 個中的 23%）。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
import torch
from transformers import AutoProcessor, VitPoseForPoseEstimation

from kernel.models.base import resolve_device, validate_image
from kernel.schemas import (
    LEFT_EYE,
    RIGHT_EYE,
    CoordinateFrame,
    Instance,
    Keypoint,
    Point2D,
)

# ViTPose++ 的 MoE 專家索引：0=COCO, 1=AiC, 2=MPII, 3=AP-10K,
# 4=APT-36K, 5=COCO-WholeBody。只換骨幹專家，換不動輸出頭（見模組開頭）。
AP10K_DATASET_INDEX = 3

# 輸出頭是 COCO 的：0=Nose, 1=L_Eye, 2=R_Eye。眼睛在 1 和 2，不是 0 和 1。
EYE_KEYPOINTS = {1: LEFT_EYE, 2: RIGHT_EYE}


@lru_cache(maxsize=2)
def load_vitpose(model_id: str, device: str):
    """載入一次就共用。理由同 :func:`kernel.models.segmentation.load_yolo`。

    ``device`` 也是快取鍵：同一份權重放在 CPU 和 CUDA 上是兩個不同的東西。
    """
    processor = AutoProcessor.from_pretrained(model_id)
    model = VitPoseForPoseEstimation.from_pretrained(model_id)
    model.to(device).eval()
    return processor, model


class ViTPoseEyes:
    """在每隻動物上標出眼睛，滿足 :class:`kernel.models.base.EyeDetector`。

    權重在第一次 :meth:`detect` 時才載入，建構本身不吃 VRAM。

    ``min_score`` 是**必要**的，不是可選的調校項：top-down 模型會無條件吐出
    全部 17 個關鍵點，側臉時被遮住的那顆眼睛照樣有座標。不區分的話每隻動物
    都會拿到兩顆眼睛，看起來合理、實際上有一顆是憑空捏的——比漏檢嚴重得多。

    沒過門檻的眼睛**不會被丟掉**，而是標成 ``observed=False``（虛擬眼）留在
    instance 上：報告需要回答「這顆是不是虛擬的」。它們不參與距離計算。
    """

    def __init__(
        self,
        model: str = "usyd-community/vitpose-plus-base",
        min_score: float = 0.3,
        device: str | None = None,
    ) -> None:
        self.model_id = model
        self.min_score = min_score
        self._device = device
        self._model = None
        self._processor = None
        self.name = f"{model.rsplit('/', 1)[-1]}-ap10k"

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = resolve_device(None)
        return self._device

    def _load(self):
        if self._model is None:
            self._processor, self._model = load_vitpose(self.model_id, self.device)
        return self._processor, self._model

    def keypoints(
        self, image: np.ndarray, instances: Sequence[Instance]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """每隻動物的全部 17 個關鍵點：``(座標 (17, 2), 分數 (17,))``。

        座標已是原圖像素。:meth:`detect` 只取其中兩個索引，但完整結果留在這
        一層，之後要加鼻子等其他點來做診斷時不必動推論邏輯。

        一次餵進所有動物的 box，不是逐隻呼叫——ViTPose 的 batch 單位是「框」
        而不是「圖」，逐隻跑會慢好幾倍。
        """
        if not instances:
            return []

        processor, model = self._load()

        # processor 要的是 COCO 的 [x, y, w, h]，不是 BBox 內部存的 xyxy
        boxes = np.array(
            [inst.bbox.as_xywh() for inst in instances], dtype=np.float32
        )

        inputs = processor(image, boxes=[boxes], return_tensors="pt").to(self.device)
        # 每個 box 一個專家索引，形狀必須是 (batch_size,)
        dataset_index = torch.full(
            (len(boxes),), AP10K_DATASET_INDEX, dtype=torch.int64, device=self.device
        )
        with torch.no_grad():
            outputs = model(**inputs, dataset_index=dataset_index)

        poses = processor.post_process_pose_estimation(outputs, boxes=[boxes])[0]
        return [
            (pose["keypoints"].cpu().numpy(), pose["scores"].cpu().numpy())
            for pose in poses
        ]

    def detect(
        self, image: np.ndarray, instances: Sequence[Instance]
    ) -> Sequence[Instance]:
        """回傳掛上眼睛的 instances，順序與傳入時一致。

        眼睛直接掛在牠所屬的 `Instance` 上，歸屬關係因此是結構性的——
        top-down 模型本來就知道每顆關鍵點是從哪個框推論出來的，這個資訊
        不需要、也不應該在事後用框中心距離重新推導。
        """
        validate_image(image)
        if not instances:
            return ()

        enriched: list[Instance] = []
        for instance, (keypoints, scores) in zip(
            instances, self.keypoints(image, instances)
        ):
            eyes: list[Keypoint] = []
            for index, name in EYE_KEYPOINTS.items():
                score = float(scores[index])
                x, y = (float(v) for v in keypoints[index][:2])
                # 沒過門檻的眼睛**留下來並標記**，而不是丟掉。丟掉的話下游只
                # 看得到「少一顆眼睛」，分不出是側臉被擋住、出框、還是模型整組
                # 失效；報告要回答「這顆是不是虛擬的」，就得保留這個證據。
                # 距離計算由 kernel.geometry 擋掉 observed=False 的點。
                eyes.append(
                    Keypoint(
                        name=name,
                        point=Point2D(x, y, CoordinateFrame.IMAGE),
                        score=score,
                        observed=score >= self.min_score,
                    )
                )
            enriched.append(instance.with_keypoints(tuple(eyes)))
        return tuple(enriched)
