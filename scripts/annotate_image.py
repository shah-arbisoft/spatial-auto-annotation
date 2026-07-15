"""Annotate arbitrary images (not from the dataset) — the demo entry point.

Full-automation mode on any photo: GroundingDINO finds objects, SAM2 segments
them, Depth Anything estimates depth, and the shipped geometric rules emit the
spatial triplets. Writes a JSON per image plus an overlay PNG with the detected
boxes and the triplet list, and prints the triplets to the console.

    python scripts/annotate_image.py path\\to\\photo.jpg
    python scripts/annotate_image.py path\\to\\folder --out outputs/demo
    python scripts/annotate_image.py photo.jpg --prompts "cup,laptop,plant"

By default it looks for the dataset's six classes; --prompts adds or replaces
the objects to look for (comma-separated noun phrases; each becomes a label).
Runs on the GPU (~2-3 s per image after the models load).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import load_rgb
from src.pipeline import annotate_objects, load_config, objects_from
from src.contact import pair_contacts
from run_sgdet import PROMPTS, detect, COMMON_OBJECTS  # reuse the proven detection wrapper

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

COLOURS = ["#e6194b", "#4363d8", "#3cb44b", "#f58231", "#911eb4", "#46f0f0",
           "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324"]


def overlay(image, objs, triplets, out_png):
    """Draw detected boxes + the triplet list under the image."""
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415

    img = Image.fromarray(image)
    W, H = img.size
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    lines = [f"{s}  --{k}-->  {o}" for s, k, o in triplets] or ["(no triplets)"]
    strip = 22 * min(len(lines), 14) + 8
    canvas = Image.new("RGB", (W, H + strip), "black")
    canvas.paste(img, (0, 0))
    dr = ImageDraw.Draw(canvas)
    for i, o in enumerate(objs):
        c = COLOURS[i % len(COLOURS)]
        x1, y1, x2, y2 = [v * s for v, s in zip(o.box, (W, H, W, H))]
        dr.rectangle([x1, y1, x2, y2], outline=c, width=3)
        tag = f"{o.label}{o.idx}"
        ty = max(0, y1 - 20)
        dr.rectangle([x1, ty, x1 + dr.textlength(tag, font=font) + 8, ty + 18], fill=c)
        dr.text((x1 + 4, ty + 1), tag, fill="white", font=font)
    for j, line in enumerate(lines[:14]):
        dr.text((6, H + 4 + 22 * j), line, fill="white", font=font)
    if len(lines) > 14:
        dr.text((W - 160, H + strip - 22), f"(+{len(lines) - 14} more)",
                fill="gray", font=font)
    canvas.save(out_png)


def collect_images(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in EXTS)
    return [path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", help="an image file or a folder of images")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--prompts", default=None,
                    help="comma-separated extra objects to look for "
                         "(e.g. 'cup,laptop,plant'); added to the six defaults")
    ap.add_argument("--common-objects", action="store_true",
                    help="search a built-in list of ~60 everyday object types "
                         "instead of naming them (for unknown footage)")
    ap.add_argument("--only-prompts", action="store_true",
                    help="use --prompts INSTEAD of the six dataset classes")
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/demo")
    args = ap.parse_args()

    paths = collect_images(Path(args.images))
    if not paths:
        raise SystemExit(f"no images found at {args.images}")

    if args.common_objects:
        PROMPTS.clear()
        PROMPTS.update({p: p.replace(" ", "_") for p in COMMON_OBJECTS})
    if args.prompts:
        extra = {p.strip(): p.strip().replace(" ", "_")
                 for p in args.prompts.split(",") if p.strip()}
        if args.only_prompts:
            PROMPTS.clear()
        PROMPTS.update(extra)

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import pipeline as hf_pipeline  # noqa: PLC0415
    print("loading models (GroundingDINO + SAM2 + Depth Anything) ...")
    detector = hf_pipeline("zero-shot-object-detection",
                           model="IDEA-Research/grounding-dino-tiny",
                           device=0 if args.device == "cuda" else -1)
    from src.depth import DepthEstimator  # noqa: PLC0415
    from src.segment import Segmenter  # noqa: PLC0415
    depther = DepthEstimator(hf_model=cfg["depth"]["hf_model"], device=args.device).load()
    segmenter = Segmenter(hf_model=cfg["segmentation"]["hf_model"], device=args.device).load()

    for path in paths:
        image = load_rgb(path)          # EXIF-corrected load, same as the study
        h, w = image.shape[:2]
        dets = detect(detector, image, args.threshold)

        objs, named = [], []
        if dets:
            from PIL import Image  # noqa: PLC0415

            boxes_px = [d["box_px"] for d in dets]
            labels = [d["label"] for d in dets]
            masks = segmenter.masks_from_boxes(image, boxes_px)
            depth_map = depther.estimate(Image.fromarray(image))
            contact = pair_contacts(masks)
            objs = objects_from(boxes_px, labels, masks, depth_map, w, h,
                                cfg.get("geometry", {}).get("z_scale", 1.0))
            name = {o.idx: f"{o.label}{o.idx}" for o in objs}
            for p in annotate_objects(objs, cfg, contact):
                for k in p.predicates:
                    named.append([name[p.subject], k, name[p.object]])

        stem = path.stem
        (out / f"{stem}.json").write_text(
            json.dumps({"image": str(path), "detections": dets,
                        "triplets": named}, indent=1), encoding="utf-8")
        overlay(image, objs, named, out / f"{stem}.png")

        print(f"\n{path.name}: {len(dets)} objects, {len(named)} triplets")
        for s, k, o in named:
            print(f"  {s}  --{k}-->  {o}")

    print(f"\noverlays + json -> {out}")


if __name__ == "__main__":
    main()
