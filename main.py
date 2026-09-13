from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from kernel.core import InterocularCore, Scene
from kernel.models.keypoint import ViTPoseEyes
from kernel.models.segmentation import AnimalSegmenter
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


def build_core(args: argparse.Namespace) -> InterocularCore:
    """依命令列參數組出 pipeline。模型在這裡只建構，權重要到第一次推論才載入。"""
    eyes = None
    if not args.no_eyes:
        eyes = ViTPoseEyes(
            model=args.eye_model, min_score=args.min_score, device=args.device
        )

    return InterocularCore(
        segmenter=AnimalSegmenter(
            weights=args.weights, conf=args.conf, device=args.device
        ),
        eyes=eyes,
    )


def format_scene(scene: Scene) -> str:
    """把一個 Scene 排成人看得懂的區塊。"""
    name = scene.source.name if scene.source else "<image>"
    height, width = scene.image_size
    lines = [
        f"{name}  {width}x{height}  "
        f"{scene.n_instances} 隻動物 / {scene.n_eyes} 顆眼睛 / "
        f"{scene.measurements.total} 組眼距"
    ]

    for inst in scene.instances:
        eyes = ", ".join(f"{kp.name}={kp.score:.2f}" for kp in inst.eyes) or "無眼睛"
        measurement = scene.measurements.for_instance(inst.instance_id)
        distance = f"  眼距={measurement.distance_px:.1f}px" if measurement else ""
        lines.append(
            f"  #{inst.instance_id} {inst.label:<9} conf={inst.score:.2f}  "
            f"{eyes}{distance}"
        )

    if not scene.instances:
        lines.append("  （沒有偵測到動物）")

    return "\n".join(lines)


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
                    "confidence": measurement.confidence,
                }
            ),
        }

    return {
        "source": str(scene.source) if scene.source else None,
        "image_size": {"height": scene.image_size[0], "width": scene.image_size[1]},
        "model": scene.model,
        "instances": [instance_to_dict(inst) for inst in scene.instances],
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

    print(f"共 {len(images)} 張影像\n")

    core = build_core(args)
    scenes: list[Scene] = []
    rendered = 0
    for path in images:
        image = load_image(path)

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
    print(
        f"合計：{total_instances} 隻動物，{total_eyes} 顆眼睛，"
        f"{total_measured} 組眼距"
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
    uv run main.py --no-eyes                # 只切割，不找眼睛
    uv run main.py --no-save                # 只印，不寫檔

    不給圖片路徑時就跑 :data:`DEFAULT_IMAGE_DIR`。

    輸出**預設就會存檔**，不必加任何參數::

        output/
        ├── results.json                完整結果
        ├── 000000000285_1_masks.jpg
        ├── 000000000285_2_eyes.jpg
        ├── 000000000285_3_interocular.jpg
        └── 000000020247_1_masks.jpg

    預設存檔是刻意的：這條 pipeline 的失效幾乎都是靜默的，跑的當下留下查驗用的
    圖，比事後起疑再回頭重跑一次划算得多。用 ``--outdir`` 換目錄、``--no-save``
    完全關掉。

    預設跑 stage 1 + 2（切割 + 眼睛），並對雙眼俱全的動物算出**像素**眼距。
    ``--no-eyes`` 只跑切割，此時不會產生 2_eyes.jpg 與 3_interocular.jpg。

    眼距目前只有像素值，還不是公分——換算成真實尺度需要深度與相機焦距，
    兩者都還沒進來。
    """

    raise SystemExit(main())
