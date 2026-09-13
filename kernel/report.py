"""把一次執行的全部結果整理成給人讀的報告。

``results.json`` 給程式讀，這份給人讀。兩份內容同源，差別在這份會**替每筆
數字下判斷**：這隻動物的眼睛是看到的還是捏的、這兩隻之間為什麼量不到。

報告分兩張表：

    每隻動物     位置、雙眼座標、是否虛擬、雙眼距離
    任兩隻動物   右眼對右眼的距離；量不到的配對也列出來，並寫明原因

兩張表各有一份 CSV（:func:`animal_rows` / :func:`pair_rows`），與 Markdown
表格共用同一份列資料，判定和備註不會兩邊各說各話。CSV 裡的數字不帶單位、
不做四捨五入以外的格式化，方便丟進試算表篩選。

「任兩隻」是字面意思：N 隻動物就列 N(N-1)/2 列，量不到的不省略。省略的話
讀報告的人會以為那兩隻動物根本不存在，而不是其中一隻側臉。

「虛擬」是什麼
--------------
top-down 姿態模型會無條件吐滿全部關鍵點。側臉時被頭擋住的那顆眼睛照樣有
座標，數值看起來完全正常，但那是模型在看不到的地方補出來的。分數低於
``min_score`` 的點就是這種，報告稱為**虛擬眼**，它們不參與任何距離計算。

另一種虛擬比較隱晦：兩顆眼睛**都過了門檻**，卻擠在幾乎同一個位置。這是
模型把同一顆看得到的眼睛同時標成左眼和右眼，分數門檻完全抓不到。報告以
:data:`COLLAPSE_RATIO` 標記這種情況。
"""

from __future__ import annotations

import csv
import math
from itertools import combinations
from pathlib import Path
from typing import Sequence

from kernel.core import Scene
from kernel.schemas import (
    LEFT_EYE,
    RIGHT_EYE,
    Instance,
    InterocularDistance,
    Keypoint,
)

#: 雙眼像素距離低於 bbox 對角線的這個比例時，視為兩顆眼睛塌在同一點。
#:
#: **未經驗證的啟發式門檻**，只用來標記、不用來過濾。依據是 testsample 上
#: 實測的正常值：馬 58 px、狗 39 px，約為各自 bbox 對角線的 8–10%；真正塌掉
#: 的情況會落在 1% 以下。取 2% 留足餘裕。標註資料齊了之後應該回頭校準。
COLLAPSE_RATIO = 0.02

#: 每隻動物表的 CSV 欄位，順序即輸出順序。
ANIMAL_COLUMNS = (
    "影像", "#", "物種", "偵測信心", "x1", "y1", "x2", "y2",
    "左眼x", "左眼y", "左眼分數", "左眼虛擬",
    "右眼x", "右眼y", "右眼分數", "右眼虛擬",
    "虛擬", "判定", "像素距離(px)", "公尺距離(m)", "信心",
)  # fmt: skip

#: 任兩隻動物表的 CSV 欄位，順序即輸出順序。
PAIR_COLUMNS = (
    "影像", "A#", "A物種", "B#", "B物種", "眼睛", "跨物種",
    "像素距離(px)", "公尺距離(m)", "深度差dz(m)", "信心", "備註",
)  # fmt: skip


def build_report(
    scenes: Sequence[Scene],
    *,
    min_score: float | None = None,
    cross_keypoint: str = RIGHT_EYE,
) -> str:
    """回傳整份 Markdown 報告。

    `min_score` 只用來寫進報告開頭的定義說明，不參與判斷——判斷依據是每顆
    keypoint 身上已經算好的 `observed`。
    """
    parts = [
        _header(scenes, min_score),
        _animal_table(scenes),
        _pair_table(scenes, cross_keypoint),
        _footnotes(min_score),
    ]
    return "\n\n".join(parts) + "\n"


def write_csv(path: Path, rows: Sequence[dict], columns: Sequence[str]) -> None:
    """把列資料寫成 CSV。

    用 UTF-8 with BOM：沒有 BOM 的話 Excel 會用系統編碼開檔，中文全變亂碼。
    """
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: _csv_value(row[k]) for k in columns} for row in rows)


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return round(value, 4)
    return value


# -- 開頭摘要 ---------------------------------------------------------------


def _header(scenes: Sequence[Scene], min_score: float | None) -> str:
    animals = [inst for s in scenes for inst in s.instances]
    verdicts = [
        _verdict(inst, s.measurements.for_instance(inst.instance_id))
        for s in scenes
        for inst in s.instances
    ]
    virtual = sum(1 for v in verdicts if v.is_virtual)
    measured = sum(1 for v in verdicts if v.label == "實測")
    pairs = sum(math.comb(s.n_instances, 2) for s in scenes)
    pairs_measured = sum(len(s.measurements.inter_object) for s in scenes)

    model = scenes[0].model if scenes else "—"
    lines = [
        "# Interoc 量測報告",
        "",
        f"- 模型：`{model}`",
        f"- 影像：{len(scenes)} 張",
        f"- 動物：{len(animals)} 隻，其中雙眼實測 {measured} 隻、含虛擬眼 {virtual} 隻",
        f"- 任兩隻配對：{pairs} 組，其中量得到 {pairs_measured} 組",
        f"- 公尺距離：{_metric_summary(scenes)}",
    ]
    return "\n".join(lines)


def _metric_summary(scenes: Sequence[Scene]) -> str:
    """一句話講清楚這份報告的公尺數是量的、估的，還是根本沒有。"""
    metric = [s for s in scenes if s.is_metric]
    if not metric:
        return "無（沒有跑深度，或拿不到焦距）——以下只有像素距離"

    estimated = sum(1 for s in metric if s.depth.focal_estimated)
    if estimated == len(metric):
        source = "焦距全部由深度模型估計"
    elif estimated == 0:
        source = "焦距全部由外部提供"
    else:
        source = f"焦距 {estimated} 張為模型估計、{len(metric) - estimated} 張為外部提供"
    return f"{len(metric)}/{len(scenes)} 張有公尺數，{source}"


# -- 每隻動物 ---------------------------------------------------------------


def animal_rows(scenes: Sequence[Scene]) -> list[dict]:
    """每隻動物一列，欄位見 :data:`ANIMAL_COLUMNS`。值保持原始型別，不帶單位。"""
    rows = []
    for scene in scenes:
        for inst in scene.instances:
            measurement = scene.measurements.for_instance(inst.instance_id)
            verdict = _verdict(inst, measurement)
            x1, y1, x2, y2 = inst.bbox.as_xyxy()
            rows.append(
                {
                    "影像": _image_name(scene),
                    "#": inst.instance_id,
                    "物種": inst.label,
                    "偵測信心": inst.score,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    **_eye_fields("左眼", inst.keypoint(LEFT_EYE)),
                    **_eye_fields("右眼", inst.keypoint(RIGHT_EYE)),
                    "虛擬": verdict.is_virtual,
                    "判定": verdict.label,
                    "像素距離(px)": measurement and measurement.distance_px,
                    "公尺距離(m)": measurement and measurement.distance_m,
                    "信心": measurement and measurement.confidence,
                }
            )
    return rows


def _eye_fields(side: str, kp: Keypoint | None) -> dict:
    if kp is None:
        return {f"{side}x": None, f"{side}y": None, f"{side}分數": None, f"{side}虛擬": None}
    return {
        f"{side}x": kp.point.u,
        f"{side}y": kp.point.v,
        f"{side}分數": kp.score,
        f"{side}虛擬": not kp.observed,
    }


def _animal_table(scenes: Sequence[Scene]) -> str:
    rows = [
        "## 每隻動物",
        "",
        "| 影像 | # | 物種 | 位置 (x1, y1)–(x2, y2) | 左眼 | 右眼 | 虛擬 | 判定 | 雙眼距離 |",
        "|---|--:|---|---|---|---|:-:|---|--:|",
    ]
    for r in animal_rows(scenes):
        x1, y1, x2, y2 = (round(r[k]) for k in ("x1", "y1", "x2", "y2"))
        rows.append(
            "| "
            + " | ".join(
                [
                    r["影像"],
                    str(r["#"]),
                    r["物種"],
                    f"({x1}, {y1})–({x2}, {y2})",
                    _eye_cell(r, "左眼"),
                    _eye_cell(r, "右眼"),
                    "是" if r["虛擬"] else "否",
                    r["判定"],
                    _distance_cell(r),
                ]
            )
            + " |"
        )

    if len(rows) == 4:
        rows.append("| — | | 沒有偵測到任何動物 | | | | | | |")
    return "\n".join(rows)


class _Verdict:
    """一隻動物的眼睛可信度判定。"""

    __slots__ = ("label", "is_virtual")

    def __init__(self, label: str, is_virtual: bool) -> None:
        self.label = label
        self.is_virtual = is_virtual


def _verdict(inst: Instance, measurement: InterocularDistance | None) -> _Verdict:
    """判定這隻動物的雙眼是看到的、捏的，還是可疑的。

    判斷順序有意義：先看有沒有虛擬眼（分數門檻），再看兩顆都過門檻時是否
    塌在一起（幾何門檻）。後者只有在前者通過時才有意義。
    """
    left, right = inst.keypoint(LEFT_EYE), inst.keypoint(RIGHT_EYE)
    if left is None and right is None:
        return _Verdict("無眼睛資料", is_virtual=False)

    inferred = inst.inferred_eyes
    if len(inferred) == 2:
        return _Verdict("雙眼皆虛擬", is_virtual=True)
    if len(inferred) == 1:
        side = "左" if inferred[0].name == LEFT_EYE else "右"
        return _Verdict(f"{side}眼虛擬（可能側臉）", is_virtual=True)

    if measurement is None or measurement.distance_px is None:
        return _Verdict("雙眼可見但無法量測", is_virtual=False)

    diagonal = math.hypot(inst.bbox.width, inst.bbox.height)
    if diagonal > 0 and measurement.distance_px < COLLAPSE_RATIO * diagonal:
        return _Verdict("雙眼重疊（疑似同一顆）", is_virtual=True)

    return _Verdict("實測", is_virtual=False)


# -- 任兩隻動物 -------------------------------------------------------------


def pair_rows(scenes: Sequence[Scene], cross_keypoint: str = RIGHT_EYE) -> list[dict]:
    """任兩隻動物一列，欄位見 :data:`PAIR_COLUMNS`。

    量不到的配對也有一列，數字欄為 None、備註寫原因。
    """
    side = _side_name(cross_keypoint)
    rows = []
    for scene in scenes:
        verdicts = {
            inst.instance_id: _verdict(
                inst, scene.measurements.for_instance(inst.instance_id)
            )
            for inst in scene.instances
        }
        for a, b in combinations(scene.instances, 2):
            found = [
                d
                for d in scene.measurements.between(a.instance_id, b.instance_id)
                if d.keypoint_a == cross_keypoint
            ]
            d = found[0] if found else None
            if d is None:
                note = _why_unmeasured(a, b, cross_keypoint, side)
            else:
                # 動物表上判為「雙眼重疊」的，這一列也要跟著說——否則兩張表對
                # 同一顆眼睛給出矛盾的可信度。只認重疊，不認一般的虛擬：「左眼
                # 虛擬」的動物右眼是看到的，拿來配對沒有問題。
                other = "左眼" if side == "右眼" else "右眼"
                notes = [
                    f"#{inst.instance_id} 雙眼重疊，{side}可能是{other}"
                    for inst in (a, b)
                    if verdicts[inst.instance_id].label.startswith("雙眼重疊")
                ]
                depth_note = _pair_note(d.distance_m, d.depth_gap_m)
                if depth_note:
                    notes.append(depth_note)
                note = "；".join(notes)

            rows.append(
                {
                    "影像": _image_name(scene),
                    "A#": a.instance_id,
                    "A物種": a.label,
                    "B#": b.instance_id,
                    "B物種": b.label,
                    "眼睛": side,
                    "跨物種": a.label != b.label,
                    "像素距離(px)": d and d.distance_px,
                    "公尺距離(m)": d and d.distance_m,
                    "深度差dz(m)": d and d.depth_gap_m,
                    "信心": d and d.confidence,
                    "備註": note,
                }
            )
    return rows


def _pair_table(scenes: Sequence[Scene], cross_keypoint: str) -> str:
    side = _side_name(cross_keypoint)
    rows = [
        f"## 任兩隻動物（{side}對{side}）",
        "",
        "| 影像 | A | B | 跨物種 | 像素距離 | 公尺距離 | 深度差 dz | 信心 | 備註 |",
        "|---|---|---|:-:|--:|--:|--:|--:|---|",
    ]
    for r in pair_rows(scenes, cross_keypoint):
        confidence = "—" if r["信心"] is None else f"{r['信心']:.2f}"
        rows.append(
            "| "
            + " | ".join(
                [
                    r["影像"],
                    f"#{r['A#']} {r['A物種']}",
                    f"#{r['B#']} {r['B物種']}",
                    "是" if r["跨物種"] else "否",
                    _px(r["像素距離(px)"]),
                    _m(r["公尺距離(m)"]),
                    _m(r["深度差dz(m)"]),
                    confidence,
                    r["備註"],
                ]
            )
            + " |"
        )

    if len(rows) == 4:
        rows.append("| — | | | | | | | | 沒有任何一張影像超過一隻動物 |")
    return "\n".join(rows)


def _side_name(keypoint: str) -> str:
    return "右眼" if keypoint == RIGHT_EYE else "左眼"


def _why_unmeasured(a: Instance, b: Instance, keypoint: str, side: str) -> str:
    """把「量不到」翻成人話。報告的讀者不該需要自己去翻 JSON 找原因。"""
    reasons = []
    for inst in (a, b):
        kp = inst.keypoint(keypoint)
        if kp is None:
            reasons.append(f"#{inst.instance_id} 沒有{side}")
        elif not kp.observed:
            reasons.append(f"#{inst.instance_id} {side}為虛擬")
        elif not kp.point.is_finite:
            reasons.append(f"#{inst.instance_id} {side}座標無效")
    return "；".join(reasons) if reasons else "無法量測"


def _pair_note(distance_m: float | None, depth_gap_m: float | None) -> str:
    """深度差佔了距離的大半時提醒一句。

    那種配對量的其實是「誰前誰後」而不是橫向距離，誤差由深度模型主導，
    是最該懷疑的一種數字。
    """
    if distance_m and depth_gap_m is not None and depth_gap_m > 0.5 * distance_m:
        return "深度差為主，數字受深度誤差主導"
    return ""


# -- 格式化 -----------------------------------------------------------------


def _image_name(scene: Scene) -> str:
    return scene.source.name if scene.source else "<image>"


def _eye_cell(row: dict, side: str) -> str:
    if row[f"{side}x"] is None:
        return "—"
    coord = f"({row[f'{side}x']:.1f}, {row[f'{side}y']:.1f}) s={row[f'{side}分數']:.2f}"
    return f"[虛擬] {coord}" if row[f"{side}虛擬"] else coord


def _distance_cell(row: dict) -> str:
    if row["信心"] is None:
        return "—"
    parts = [_px(row["像素距離(px)"])]
    if row["公尺距離(m)"] is not None:
        parts.append(_m(row["公尺距離(m)"]))
    return " / ".join(parts)


def _px(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f} px"


def _m(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f} m"


def _footnotes(min_score: float | None) -> str:
    threshold = f"`min_score={min_score}`" if min_score is not None else "`min_score`"
    return "\n".join(
        [
            "## 判讀說明",
            "",
            f"- **虛擬眼**：分數低於 {threshold} 的眼睛。姿態模型會無條件吐出全部"
            "關鍵點，側臉時被擋住的那顆照樣有座標，但那是補出來的。虛擬眼不參與"
            "任何距離計算。",
            f"- **雙眼重疊**：兩顆眼睛都過了門檻，距離卻小於 bbox 對角線的 "
            f"{COLLAPSE_RATIO:.0%}，通常是同一顆眼睛被同時標成左右眼。"
            "這是未經驗證的啟發式門檻，只標記、不過濾——該列的雙眼距離仍會輸出，"
            "但不應採信。",
            "- **深度差 dz**：兩顆眼睛沿光軸的距離差。它佔了總距離的大半時，"
            "該筆量的其實是「誰前誰後」，誤差由深度模型主導。",
            "- **位置**：bbox 的左上與右下角，原圖像素座標。",
        ]
    )
