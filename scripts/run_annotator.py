"""Run the automatic annotator over the dataset.

For each image: take the ground-truth boxes/labels (PredCls setting), run SAM2 +
Depth Anything, lift to 3D, compute the seven predicates, and write the scene
graph in all three dataset formats. Also caches per-image object geometry and a
per-pair record table used for near-threshold fitting (eval/fit_near.py) and the
RQ1 fidelity study.

    python scripts/run_annotator.py --limit 20            # quick trial
    python scripts/run_annotator.py                       # full dataset
    python scripts/run_annotator.py --no-sam2             # box-geometry only (fast)

Perception needs the GPU; outputs land under outputs/.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import SpatialDataset, load_rgb
from src.pipeline import annotate_image, load_config
from src.writers import write_vg_json, write_yolo_txt, write_h5


def px_boxes(objects, w, h):
    """Normalised (x1,y1,x2,y2) -> pixel boxes."""
    return [(o.box[0] * w, o.box[1] * h, o.box[2] * w, o.box[3] * h) for o in objects]


def gold_maps(gt_image):
    """Map (subj_idx, obj_idx) -> set of gold predicate names."""
    m: dict[tuple[int, int], set[str]] = {}
    for r in gt_image.relations:
        m.setdefault((r.subject, r.object), set()).add(r.predicate)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N images")
    ap.add_argument("--group", default=None, help="restrict to one group, e.g. group_0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--no-sam2", action="store_true", help="use box geometry, skip SAM2")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = SpatialDataset(cfg["dataset"]["root"], target_only=cfg["dataset"]["target_only"])

    out = Path(args.out)
    ann_dir = out / "annotations"
    geo_dir = out / "geometry"
    ann_dir.mkdir(parents=True, exist_ok=True)
    geo_dir.mkdir(parents=True, exist_ok=True)

    # perception models (loaded once); skipped in --no-sam2 for the segmenter
    depther = _load_depth(cfg, args.device)
    segmenter = None if args.no_sam2 else _load_sam2(cfg, args.device)

    pairs_csv = open(out / "pairs.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(pairs_csv)
    # near_metric = size-relative box gap (the `near` rule's input);
    # dist3d kept for the classifier features / ablations.
    writer.writerow(["image_id", "subj", "obj", "near_metric", "dist3d",
                     "gold_near", "gold_any", "gold_contact", "pred", "gold"])

    images = list(ds)
    if args.group:
        images = [im for im in images if im.image_id.startswith(args.group)]
    if args.limit:
        images = images[: args.limit]

    n_done = 0
    for gt in tqdm(images, desc="annotating"):
        if not gt.image_path.exists() or len(gt.objects) == 0:
            continue
        image = load_rgb(gt.image_path)  # EXIF-corrected: annotations are in the upright frame
        h, w = image.shape[:2]
        boxes = px_boxes(gt.objects, w, h)
        labels = [o.label for o in gt.objects]

        objs, pairs, _ = annotate_image(
            image, boxes, labels, segmenter, depther, cfg, use_sam2=not args.no_sam2
        )

        group, stem = gt.image_id.split("/")
        (ann_dir / group).mkdir(exist_ok=True)
        (geo_dir / group).mkdir(exist_ok=True)
        obj_dicts = [{"label": o.label, "box": list(o.box)} for o in objs]
        base = ann_dir / group / stem
        write_vg_json(gt.image_id, w, h, obj_dicts, pairs, f"{base}.json")
        write_yolo_txt(obj_dicts, f"{base}.txt")
        try:
            write_h5(gt.image_id, w, h, obj_dicts, pairs, f"{base}.h5")
        except Exception as e:  # h5py missing / schema issue — don't kill the run
            if n_done == 0:
                tqdm.write(f"[warn] h5 export skipped: {e}")

        # cache object geometry for offline near-fitting / re-evaluation
        geo = [{"idx": o.idx, "label": o.label, "box": list(o.box),
                "cx": o.cx, "cy": o.cy, "depth": o.depth, "pos3d": o.pos3d.tolist()}
               for o in objs]
        (geo_dir / group / f"{stem}.json").write_text(json.dumps(geo), encoding="utf-8")

        # per-pair records
        from src.predicates import box_gap_rel
        gmap = gold_maps(gt)
        pred_map = {(p.subject, p.object): p.predicates for p in pairs}
        for a in objs:
            for b in objs:
                if a.idx == b.idx:
                    continue
                gap = box_gap_rel(a, b)
                dist = float(np.linalg.norm(a.pos3d - b.pos3d))
                gold = gmap.get((a.idx, b.idx), set())
                gold_rev = gmap.get((b.idx, a.idx), set())
                contact = bool({"on", "under"} & (gold | gold_rev))
                pred = pred_map.get((a.idx, b.idx), [])
                writer.writerow([gt.image_id, a.idx, b.idx, f"{gap:.6f}", f"{dist:.6f}",
                                 int("near" in gold), int(bool(gold or gold_rev)),
                                 int(contact), ";".join(pred), ";".join(sorted(gold))])
        n_done += 1

    pairs_csv.close()
    print(f"done: {n_done} images -> {ann_dir} ; records -> {out/'pairs.csv'}")


def _load_depth(cfg, device):
    from src.depth import DepthEstimator
    return DepthEstimator(hf_model=cfg["depth"]["hf_model"], device=device).load()


def _load_sam2(cfg, device):
    from src.segment import Segmenter
    return Segmenter(hf_model=cfg["segmentation"]["hf_model"], device=device).load()


if __name__ == "__main__":
    main()
