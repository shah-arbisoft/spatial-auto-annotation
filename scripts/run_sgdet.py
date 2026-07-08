"""Detector-in-the-loop (SGDet) pass: no ground-truth boxes anywhere.

GroundingDINO (zero-shot, text-prompted) finds the objects, SAM2 segments
them, Depth Anything provides depth, and the shipped geometric rules emit the
triplets — the full-automation deployment mode, as opposed to the PredCls
setting used to isolate the relation rules in RQ1.

Per image this caches detections + triplets to outputs/sgdet/, which
eval/sgdet_eval.py scores offline against the human labels (greedy IoU
matching of detected boxes to gold objects).

    python scripts/run_sgdet.py            # full dataset (GPU, ~15-25 min)
    python scripts/run_sgdet.py --limit 20 # trial
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import SpatialDataset, load_rgb
from src.pipeline import annotate_objects, load_config, objects_from
from src.contact import pair_contacts

# text prompts -> dataset class names. Short noun phrases work best with
# GroundingDINO; tuned once on the 20-image trial (see outputs/tables/sgdet.md).
PROMPTS = {
    "book": "book",
    "bottle": "bottle",
    "box": "box",
    "cube": "cube",
    "person": "human",
    "remote control": "remote",
}


def detect(detector, image, threshold):
    from PIL import Image  # noqa: PLC0415

    results = detector(Image.fromarray(image),
                       candidate_labels=[f"{p}." for p in PROMPTS],
                       threshold=threshold)
    dets = []
    for r in results:
        label = PROMPTS.get(r["label"].rstrip(".").strip())
        if label is None:
            continue
        b = r["box"]
        dets.append({"label": label, "score": round(float(r["score"]), 4),
                     "box_px": [b["xmin"], b["ymin"], b["xmax"], b["ymax"]]})
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/sgdet")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = SpatialDataset(cfg["dataset"]["root"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import pipeline as hf_pipeline  # noqa: PLC0415
    print("loading GroundingDINO (zero-shot detection) ...")
    detector = hf_pipeline("zero-shot-object-detection",
                           model="IDEA-Research/grounding-dino-tiny",
                           device=0 if args.device == "cuda" else -1)
    from src.depth import DepthEstimator  # noqa: PLC0415
    from src.segment import Segmenter  # noqa: PLC0415
    depther = DepthEstimator(hf_model=cfg["depth"]["hf_model"], device=args.device).load()
    segmenter = Segmenter(hf_model=cfg["segmentation"]["hf_model"], device=args.device).load()

    images = list(ds)
    if args.limit:
        images = images[: args.limit]

    n = 0
    for gt in tqdm(images, desc="sgdet"):
        if not gt.image_path.exists() or len(gt.objects) == 0:
            continue
        image = load_rgb(gt.image_path)
        h, w = image.shape[:2]
        dets = detect(detector, image, args.threshold)

        triplets = []
        if dets:
            from PIL import Image  # noqa: PLC0415

            boxes_px = [d["box_px"] for d in dets]
            labels = [d["label"] for d in dets]
            masks = segmenter.masks_from_boxes(image, boxes_px)
            depth_map = depther.estimate(Image.fromarray(image))
            contact = pair_contacts(masks)
            objs = objects_from(boxes_px, labels, masks, depth_map, w, h,
                                cfg.get("geometry", {}).get("z_scale", 1.0))
            for p in annotate_objects(objs, cfg, contact):
                for k in p.predicates:
                    triplets.append([p.subject, k, p.object])

        group, stem = gt.image_id.split("/")
        (out / group).mkdir(exist_ok=True)
        (out / group / f"{stem}.json").write_text(
            json.dumps({"detections": dets, "triplets": triplets}),
            encoding="utf-8")
        n += 1

    print(f"done: {n} images -> {out}")
    print("score it offline:  python eval/sgdet_eval.py")


if __name__ == "__main__":
    main()
