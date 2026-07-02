"""Re-run the predicate rules from the cached geometry — no GPU needed.

scripts/run_annotator.py caches each object's lifted geometry (box, centroid,
depth, 3D position) under outputs/geometry/. Threshold or rule changes only
affect the predicate stage, so this script re-emits every annotation file and
pairs.csv from that cache in seconds instead of re-running SAM2/depth.

    python scripts/reannotate_from_cache.py
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
from src.dataset import SpatialDataset
from src.pipeline import annotate_objects, load_config
from src.predicates import Obj, box_gap_rel
from src.writers import write_vg_json, write_yolo_txt, write_h5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = SpatialDataset(cfg["dataset"]["root"], target_only=cfg["dataset"]["target_only"])
    out = Path(args.out)
    geo_root = out / "geometry"
    ann_dir = out / "annotations"
    if not geo_root.exists():
        raise SystemExit("no geometry cache — run scripts/run_annotator.py once first")

    pairs_csv = open(out / "pairs.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(pairs_csv)
    writer.writerow(["image_id", "subj", "obj", "near_metric", "dist3d",
                     "gold_near", "gold_any", "gold_contact", "pred", "gold"])

    n = 0
    for gt in tqdm(list(ds), desc="re-annotating"):
        group, stem = gt.image_id.split("/")
        gpath = geo_root / group / f"{stem}.json"
        if not gpath.exists():
            continue
        geo = json.loads(gpath.read_text(encoding="utf-8"))
        objs = [Obj(o["idx"], o["label"], tuple(o["box"]), o["cx"], o["cy"],
                    o["depth"], np.array(o["pos3d"])) for o in geo]
        pairs = annotate_objects(objs, cfg)

        (ann_dir / group).mkdir(parents=True, exist_ok=True)
        obj_dicts = [{"label": o.label, "box": list(o.box)} for o in objs]
        base = ann_dir / group / stem
        write_vg_json(gt.image_id, gt.width, gt.height, obj_dicts, pairs, f"{base}.json")
        write_yolo_txt(obj_dicts, f"{base}.txt")
        try:
            write_h5(gt.image_id, gt.width, gt.height, obj_dicts, pairs, f"{base}.h5")
        except Exception as e:
            if n == 0:
                tqdm.write(f"[warn] h5 export skipped: {e}")

        gmap: dict[tuple[int, int], set] = {}
        for r in gt.relations:
            gmap.setdefault((r.subject, r.object), set()).add(r.predicate)
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
        n += 1

    pairs_csv.close()
    print(f"re-annotated {n} images from cache -> {ann_dir}")


if __name__ == "__main__":
    main()
