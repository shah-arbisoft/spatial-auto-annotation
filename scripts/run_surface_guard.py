"""Guard-only surface detection: elevation evidence for the ground-plane rule.

The plane fallback's audited failure mode is an object resting on a support
the six annotated classes cannot see (a flight case, a table edge, a tray):
the object's box bottom then locates its support, not itself, and the fallback
mis-orders it. This pass detects such surfaces with zero-shot prompts and
records, per image, which annotated objects rest on one. The surfaces are
never labelled and never enter any relation - they exist only to widen the
elevation guard.

Writes outputs/geometry/<group>/<stem>.surfelev.json:
    {"elevated": [obj_idx, ...], "surfaces": [{"label", "score"}, ...]}

scripts/reannotate_from_cache.py picks the files up automatically; re-run it
plus eval/fidelity.py afterwards to measure the effect.

    python scripts/run_surface_guard.py            # full dataset (GPU, ~25 min)
    python scripts/run_surface_guard.py --limit 20 # trial
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import SpatialDataset, load_rgb
from src.pipeline import load_config
from src.contact import contact_below

# Surfaces objects in this dataset actually rest on, beyond the floor and the
# six annotated classes. Guard-only: these are never emitted as objects.
SURFACE_PROMPTS = ["table", "desk", "chair", "shelf", "tray",
                   "flight case", "cardboard box lid"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--contact-min", type=float, default=0.60,
                    help="mask-contact fraction onto a surface that counts as elevated")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = SpatialDataset(cfg["dataset"]["root"])
    geo_root = Path("outputs/geometry")
    if not geo_root.exists():
        raise SystemExit("no geometry cache - run scripts/run_annotator.py first")

    from transformers import pipeline as hf_pipeline  # noqa: PLC0415
    print("loading GroundingDINO + SAM2 ...")
    detector = hf_pipeline("zero-shot-object-detection",
                           model="IDEA-Research/grounding-dino-tiny",
                           device=0 if args.device == "cuda" else -1)
    from src.segment import Segmenter  # noqa: PLC0415
    segmenter = Segmenter(hf_model=cfg["segmentation"]["hf_model"], device=args.device).load()
    from PIL import Image  # noqa: PLC0415

    images = list(ds)
    if args.limit:
        images = images[: args.limit]

    n_img = n_elev = 0
    for gt in tqdm(images, desc="surface guard"):
        group, stem = gt.image_id.split("/")
        gpath = geo_root / group / f"{stem}.json"
        if not gpath.exists() or not gt.image_path.exists():
            continue
        image = load_rgb(gt.image_path)
        h, w = image.shape[:2]

        results = detector(Image.fromarray(image),
                           candidate_labels=[f"{p}." for p in SURFACE_PROMPTS],
                           threshold=args.threshold)
        surfaces = []
        surf_boxes = []
        for r in results:
            b = r["box"]
            surf_boxes.append([b["xmin"], b["ymin"], b["xmax"], b["ymax"]])
            surfaces.append({"label": r["label"].rstrip("."),
                             "score": round(float(r["score"]), 4)})

        elevated: list[int] = []
        if surf_boxes:
            surf_masks = segmenter.masks_from_boxes(image, surf_boxes)
            # object masks: SAM2 on the annotated ground-truth boxes
            obj_boxes_px = [[o.box[0] * w, o.box[1] * h, o.box[2] * w, o.box[3] * h]
                            for o in gt.objects]
            obj_masks = segmenter.masks_from_boxes(image, obj_boxes_px)
            for oi, om in enumerate(obj_masks):
                for sm in surf_masks:
                    # ignore a "surface" that is essentially the same region
                    inter = np.logical_and(om, sm).sum()
                    if inter > 0.8 * max(om.sum(), 1):
                        continue
                    if contact_below(om, sm) >= args.contact_min:
                        elevated.append(oi)
                        break
        out = gpath.with_suffix("").with_suffix(".surfelev.json")
        out.write_text(json.dumps({"elevated": sorted(set(elevated)),
                                   "surfaces": surfaces}), encoding="utf-8")
        n_img += 1
        n_elev += len(set(elevated))

    print(f"done: {n_img} images, {n_elev} object-on-surface findings")
    print("next: python scripts/reannotate_from_cache.py && python eval/fidelity.py")


if __name__ == "__main__":
    main()
