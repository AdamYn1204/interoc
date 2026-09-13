from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from kernel.core import InterocularCore, Scene
from kernel.models.depth import (
    DEFAULT_MODEL as DEFAULT_DEPTH_MODEL,
)
from kernel.models.depth import (
    DEPTH_PRO_MODEL,
    DepthAnythingV2,
    DepthPro,
    focal_from_fov,
)
from kernel.models.keypoint import ViTPoseEyes
from kernel.models.segmentation import AnimalSegmenter
from kernel.schemas import LEFT_EYE, RIGHT_EYE
from kernel.visualization import render_scene

DEFAULT_IMAGE_DIR = Path("testsample")

#: 疊圖與結果檔的預設輸出目錄。不必特別指定就會存檔——這條 pipeline 的失效
#: 幾乎都是靜默的，預設就把查驗用的圖留下來，比事後想查才回頭重跑划算。
DEFAULT_OUTPUT_DIR = Path("output")

#: 完整結果的檔名，寫在輸出目錄底下。
RESULTS_FILENAME = "results.json"

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def load_image(path: Path) -> np.ndarray:
    buffer = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"無法解碼影像：{path}")
    return np.ascontiguousarray(bgr[:, :, ::-1])


def collect_images(paths: list[str], limit: int | None) -> list[Path]:
    targets = [Path(p) for p in paths] if paths else [DEFAULT_IMAGE_DIR]

    found: list[Path] = []
    for target in targets:
        if target.is_dir():
            found.extend(
                sorted(
                    p
                    for p in target.iterdir()
                    if p.suffix.lower() in IMAGE_SUFFIXES
                )
            )
        elif target.is_file():
            found.append(target)
        else:
            raise FileNotFoundError(f"找不到路徑：{target}")

    return found[:limit] if limit else found


def resolve_focal(args: argparse.Namespace, width: int) -> float | None:
    """決定要覆寫給深度模型的焦距（像素），沒有就回 None。

    `--focal` 優先；其次由 `--fov` 推算。兩者都沒給就回 None，交由深度模型
    自己估——Depth Pro 會估，Depth Anything V2 不會。

    `--fov` 是**假設值，不是量測值**：真實鏡頭的水平視角從廣角的 100° 到
    望遠的 10° 都有，猜錯 10° 大約就是 20% 的距離誤差，而畫面上完全看不出來。
    有 Depth Pro 可用的話，它估的值幾乎一定比這個猜測好。
    """
    if args.focal is not None:
        return args.focal
    if args.fov is not None:
        return focal_from_fov(args.fov, width)
    return None


def build_core(
    args: argparse.Namespace, focal_px: float | None
) -> InterocularCore:
    """依命令列參數組出 pipeline。模型在這裡只建構，權重要到第一次推論才載入。"""
    eyes = None
    if not args.no_eyes:
        eyes = ViTPoseEyes(
            model=args.eye_model, min_score=args.min_score, device=args.device
        )

    depth = None
    if not args.no_depth and eyes is not None:
        # 沒有眼睛就沒有要反投影的點，跑深度只是白燒一次推論——所以
        # --no-eyes 會連帶關掉 stage 3，不必再多下一個旗標。
        factory = DepthPro if args.depth_model == DEPTH_PRO_MODEL else DepthAnythingV2
        depth = factory(model=args.depth_model, device=args.device)

    return InterocularCore(
        segmenter=AnimalSegmenter(
            weights=args.weights, conf=args.conf, device=args.device
        ),
        eyes=eyes,
        depth=depth,
        focal_px=focal_px,
        cross_keypoint=args.cross_eye,
    )


def format_scene(scene: Scene) -> str:
    """把一個 Scene 排成人看得懂的區塊。"""
    name = scene.source.name if scene.source else "<image>"
    height, width = scene.image_size
    focal = ""
    if scene.depth is not None and scene.depth.focal_px is not None:
        source = "估" if scene.depth.focal_estimated else "given"
        focal = f"  fx={scene.depth.focal_px:.0f}px({source})"

    lines = [
        f"{name}  {width}x{height}  "
        f"{scene.n_instances} 隻動物 / {scene.n_eyes} 顆眼睛 / "
        f"{scene.measurements.total} 筆量測{focal}"
    ]

    for inst in scene.instances:
        eyes = ", ".join(f"{kp.name}={kp.score:.2f}" for kp in inst.eyes) or "無眼睛"
        measurement = scene.measurements.for_instance(inst.instance_id)
        distance = f"  眼距={_distance_text(measurement)}" if measurement else ""
        lines.append(
            f"  #{inst.instance_id} {inst.label:<9} conf={inst.score:.2f}  "
            f"{eyes}{distance}"
        )

    if not scene.instances:
        lines.append("  （沒有偵測到動物）")

    for d in scene.measurements.inter_object:
        mark = "跨物種" if d.is_cross_species else "同物種"
        gap = "" if d.depth_gap_m is None else f" dz={d.depth_gap_m:.2f}m"
        lines.append(
            f"  #{d.instance_a}({d.label_a}) ↔ #{d.instance_b}({d.label_b}) "
            f"{mark}  {d.keypoint_a}  {_distance_text(d)}{gap}"
        )

    return "\n".join(lines)


def _distance_text(measurement) -> str:
    """像素距離一定有，公尺距離則看有沒有跑 stage 3。"""
    parts = []
    if measurement.distance_px is not None:
        parts.append(f"{measurement.distance_px:.1f}px")
    if measurement.distance_m is not None:
        parts.append(f"{measurement.distance_m:.3f}m")
    parts.append(f"c={measurement.confidence:.2f}")
    return " ".join(parts)


def scene_to_dict(scene: Scene) -> dict:
    """轉成可序列化的結構。遮罩不進 JSON，太大了；要看輪廓請看疊圖。"""

    def instance_to_dict(inst) -> dict:
        # 眼距掛在各自的 instance 底下而不是另開一個頂層區塊：這一版的量測
        # 兩端都在同一隻動物身上，讀 JSON 的人不必自己拿 id 去對照。跨物體
        # 的距離進來時無處可掛，屆時才需要一個平行的 measurements 區塊。
        measurement = scene.measurements.for_instance(inst.instance_id)
        return {
            "instance_id": inst.instance_id,
            "label": inst.label,
            "score": inst.score,
            "bbox_xyxy": list(inst.bbox.as_xyxy()),
            "eyes": {
                kp.name: {
                    "x": kp.point.u,
                    "y": kp.point.v,
                    "score": kp.score,
                }
                for kp in inst.eyes
            },
            "interocular": (
                None
                if measurement is None
                else {
                    "distance_px": measurement.distance_px,
                    "distance_m": measurement.distance_m,
                    "confidence": measurement.confidence,
                }
            ),
        }

    return {
        "source": str(scene.source) if scene.source else None,
        "image_size": {"height": scene.image_size[0], "width": scene.image_size[1]},
        "model": scene.model,
        "is_metric": scene.is_metric,
        # 焦距的來源要記下來：回頭看結果時，「這組公尺數是量的還是估的」
        # 是判斷它能不能用的第一個問題。
        "camera": (
            None
            if scene.depth is None
            else {
                "focal_px": scene.depth.focal_px,
                "focal_estimated": scene.depth.focal_estimated,
            }
        ),
        "instances": [instance_to_dict(inst) for inst in scene.instances],
        # 跨物體距離放在頂層而不是掛在某一隻動物底下：它的兩端分屬兩隻動物，
        # 掛在任何一邊都會讓另一邊看起來像是附屬品。
        "inter_object": [
            {
                "instance_a": d.instance_a,
                "label_a": d.label_a,
                "instance_b": d.instance_b,
                "label_b": d.label_b,
                "keypoint": d.keypoint_a,
                "cross_species": d.is_cross_species,
                "distance_px": d.distance_px,
                "distance_m": d.distance_m,
                "depth_gap_m": d.depth_gap_m,
                "confidence": d.confidence,
            }
            for d in scene.measurements.inter_object
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="切出畫面中的動物。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"不給路徑時預設跑 {DEFAULT_IMAGE_DIR}/ 底下的全部圖片。",
    )
    parser.add_argument(
        "images",
        nargs="*",
        help=f"圖片檔或資料夾（可多個）。省略時用 {DEFAULT_IMAGE_DIR}/",
    )

    stage1 = parser.add_argument_group("stage 1：切割")
    stage1.add_argument("--weights", default="yolo11l-seg.pt", help="YOLO*-seg 權重")
    stage1.add_argument("--conf", type=float, default=0.5, help="偵測信心門檻")

    stage2 = parser.add_argument_group("stage 2：眼睛")
    stage2.add_argument("--no-eyes", action="store_true", help="只做切割，不找眼睛")
    stage2.add_argument(
        "--eye-model", default="usyd-community/vitpose-plus-base", help="ViTPose 權重"
    )
    stage2.add_argument(
        "--min-score",
        type=float,
        default=0.3,
        help="眼睛信心門檻。調低會讓側臉動物拿到被遮住的那顆眼睛，"
        "座標是憑空捏的",
    )

    stage3 = parser.add_argument_group(
        "stage 3：深度（公尺距離）",
        "預設開啟。Depth Pro 會自己估焦距，所以不需要任何相機參數就有公尺數；"
        "已知焦距時用 --focal 覆寫會更準。",
    )
    stage3.add_argument(
        "--no-depth",
        action="store_true",
        help="不跑深度，只輸出像素距離（省下權重下載與一次推論）",
    )
    stage3.add_argument(
        "--depth-model",
        default=DEFAULT_DEPTH_MODEL,
        help=f"深度權重（預設 {DEFAULT_DEPTH_MODEL}，會自己估焦距）。"
        f"換成 Depth Anything V2 的 Metric checkpoint 則必須自行提供焦距",
    )
    stage3.add_argument(
        "--focal",
        type=float,
        metavar="PX",
        help="原圖的水平焦距（像素）。覆寫模型的估計值",
    )
    stage3.add_argument(
        "--fov",
        type=float,
        metavar="DEG",
        help="水平視角（度），用來推算焦距。這是假設值，會等比例影響所有距離",
    )
    stage3.add_argument(
        "--cross-eye",
        default=RIGHT_EYE,
        choices=(LEFT_EYE, RIGHT_EYE),
        help=f"跨物體配對用哪顆眼睛（預設 {RIGHT_EYE}）",
    )

    out = parser.add_argument_group("輸出")
    out.add_argument(
        "--json",
        metavar="PATH",
        type=Path,
        help=f"完整結果的 JSON 路徑（預設 <outdir>/{RESULTS_FILENAME}）",
    )
    out.add_argument(
        "--outdir",
        metavar="PATH",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"疊圖與 results.json 的輸出目錄（預設 {DEFAULT_OUTPUT_DIR}/）",
    )
    out.add_argument(
        "--no-save",
        action="store_true",
        help="不寫出任何檔案，只印在終端機",
    )
    out.add_argument("--limit", type=int, metavar="N", help="只跑前 N 張")
    out.add_argument("--device", help="cuda / cuda:0 / cpu；省略時自動偵測")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        images = collect_images(args.images, args.limit)
    except FileNotFoundError as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1

    if not images:
        print(
            f"錯誤：{DEFAULT_IMAGE_DIR}/ 底下沒有圖片"
            f"（支援 {', '.join(sorted(IMAGE_SUFFIXES))}）",
            file=sys.stderr,
        )
        return 1

    # --no-save 一票否決所有寫檔，讓「只想看一眼」的情況不會留下垃圾。
    outdir = None if args.no_save else args.outdir
    json_path = None if args.no_save else (args.json or outdir / RESULTS_FILENAME)

    if not args.no_depth and args.focal is None and args.fov is not None:
        print(
            f"注意：焦距是由 --fov {args.fov}° 推算的假設值，不是量測值。"
            f"所有公尺數都會等比例地跟著這個假設走；像素距離不受影響。\n"
        )

    print(f"共 {len(images)} 張影像\n")

    scenes: list[Scene] = []
    rendered = 0
    core: InterocularCore | None = None
    focal_px: float | None = None

    for path in images:
        image = load_image(path)

        # 焦距依 --fov 推算時會跟著影像寬度走，所以 core 不能在迴圈外建好就
        # 一路用到底——同一批圖裡混著不同解析度時，共用一個焦距會讓其中一部分
        # 的公尺數整體偏掉。寬度沒變就沿用，權重才不會被重複載入。
        width = image.shape[1]
        current_focal = resolve_focal(args, width)
        if core is None or current_focal != focal_px:
            focal_px = current_focal
            core = build_core(args, focal_px)

        scene = core.forward(image, source=path)
        scenes.append(scene)
        print(format_scene(scene))

        if outdir:
            written = render_scene(image, scene, outdir)
            rendered += len(written)
            if written:
                print(f"  → {', '.join(p.name for p in written)}")
        print()

    total_instances = sum(s.n_instances for s in scenes)
    total_eyes = sum(s.n_eyes for s in scenes)
    total_measured = sum(s.measurements.total for s in scenes)
    total_cross = sum(len(s.measurements.cross_species()) for s in scenes)
    print(
        f"合計：{total_instances} 隻動物，{total_eyes} 顆眼睛，"
        f"{total_measured} 筆量測（其中跨物種 {total_cross} 筆）"
    )
    if outdir:
        print(f"疊圖：{rendered} 張，寫入 {outdir}/")

    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps([scene_to_dict(s) for s in scenes], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"結果：{json_path}")

    return 0


if __name__ == "__main__":
    """Interoc 命令列入口。

    uv run main.py                          # 跑 testsample/ 底下全部圖片
    uv run main.py path/to/cat.jpg          # 跑單張
    uv run main.py imgs/ --limit 5          # 跑資料夾的前 5 張
    uv run main.py --focal 1800             # 已知焦距時覆寫 Depth Pro 的估計值
    uv run main.py --no-depth               # 只要像素距離，不跑深度
    uv run main.py --no-eyes                # 只切割（連帶不跑深度）
    uv run main.py --no-save                # 只印，不寫檔

    不給圖片路徑時就跑 :data:`DEFAULT_IMAGE_DIR`。

    輸出**預設就會存檔**，不必加任何參數::

        output/
        ├── results.json                完整結果
        ├── 000000000285_1_masks.jpg
        ├── 000000000285_2_eyes.jpg
        ├── 000000000285_3_depth.jpg
        ├── 000000000285_4_measure.jpg
        └── 000000020247_1_masks.jpg

    預設存檔是刻意的：這條 pipeline 的失效幾乎都是靜默的，跑的當下留下查驗用的
    圖，比事後起疑再回頭重跑一次划算得多。用 ``--outdir`` 換目錄、``--no-save``
    完全關掉。

    **預設三段全跑**，輸出每隻動物的雙眼間距與兩兩動物之間的右眼對右眼距離，
    像素與公尺都有。公尺數之所以不需要任何相機參數，是因為 Depth Pro 會自己
    估焦距——COCO 的影像 EXIF 被剝得一乾二淨，這是唯一能自動拿到尺度的路。
    已知真實焦距時用 ``--focal`` 覆寫會更準。

    第一次執行會下載三份權重（YOLO、ViTPose、Depth Pro），其中 Depth Pro 約
    1.9 GB。只想看切割或只要像素距離時，用 ``--no-depth`` 跳過它。

    像素距離任何情況下都會輸出，它是驗證公尺數的對照組。JSON 裡的
    ``camera.focal_estimated`` 記著這次的焦距是量的還是估的。
    """

    raise SystemExit(main())
