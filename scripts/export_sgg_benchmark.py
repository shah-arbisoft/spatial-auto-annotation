"""Export both label sources to SGG-Benchmark's COCO-relation format.

Builds datasets/spatial_sgg/{train,val,test}/ with the images plus TWO
annotation variants per split:

    _annotations.human.coco.json   relations = the dataset's human labels
    _annotations.auto.coco.json    relations = this tool's labels (pairs.csv)

Copy the chosen variant to `_annotations.coco.json` before training — the
only difference between the two experiment arms is that file. Boxes and
classes are the ground-truth objects in both variants (the comparison
isolates the relation-label source, exactly like RQ2), and the test split
always carries the HUMAN relations, whichever arm is trained.

Split follows the calibration protocol: train = groups 0-4, val = group_5,
test = groups 6-8. Runs offline (no GPU) from the dataset + outputs/pairs.csv.

    python scripts/export_sgg_benchmark.py
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import SpatialDataset
from src.pipeline import load_config
from src.predicates import PREDICATES

SPLITS = {
    "train": {f"group_{i}" for i in range(5)},
    "val": {"group_5"},
    "test": {"group_6", "group_7", "group_8"},
}


def split_of(group: str) -> str:
    for s, gs in SPLITS.items():
        if group in gs:
            return s
    raise KeyError(group)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm-replies", default=None, dest="vlm_replies",
                    help="a VLM replies file; emits a third annotation "
                         "variant, _annotations.vlm.coco.json, so the "
                         "benchmark can be trained on that source too")
    args = ap.parse_args()
    cfg = load_config("configs/default.yaml")
    ds = SpatialDataset(cfg["dataset"]["root"])
    out_root = Path("datasets/spatial_sgg")

    # auto labels for every ordered pair, from the shipped-rules re-annotation
    auto = defaultdict(list)  # image_id -> [(subj, obj, predicate), ...]
    with open("outputs/pairs.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["pred"]:
                for k in r["pred"].split(";"):
                    auto[r["image_id"]].append((int(r["subj"]), int(r["obj"]), k))

    # a vision-language model's labels, same shape as the auto dict
    vlm = defaultdict(list)
    variants = ["human", "auto"]
    if args.vlm_replies:
        variants.append("vlm")
        for line in Path(args.vlm_replies).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            for rel in r.get("relations", []):
                s_, pred_, o_ = rel   # stored as [subject, predicate, object]
                if pred_ in PREDICATES:
                    vlm[r["image_id"]].append((int(s_), int(o_), pred_))
        covered = len(vlm)
        print(f"VLM labels: {sum(len(v) for v in vlm.values())} relations "
              f"over {covered} images")

    pred_to_id = {k: i for i, k in enumerate(PREDICATES)}
    # index 0 = __no_relation__; predicates occupy 1..7 (see predicate_id +1)
    rel_categories = [{"id": 0, "name": "__no_relation__"}]
    rel_categories += [{"id": i + 1, "name": k} for i, k in enumerate(PREDICATES)]

    label_ids: dict[str, int] = {}
    data = {s: {"images": [], "annotations": [], **{v: [] for v in variants}}
            for s in SPLITS}
    counters = {s: {"img": 0, "ann": 0, **{v: 0 for v in variants}}
                for s in SPLITS}
    next_img = {s: 0 for s in SPLITS}
    next_ann = {s: 0 for s in SPLITS}
    next_rel = {s: {v: 0 for v in variants} for s in SPLITS}

    for s in SPLITS:
        (out_root / s).mkdir(parents=True, exist_ok=True)

    n_copied = 0
    for gt in ds:
        group, stem = gt.image_id.split("/")
        s = split_of(group)
        if not gt.image_path.exists() or len(gt.objects) == 0:
            continue

        img_id = next_img[s]
        next_img[s] += 1
        file_name = f"{group}_{stem}.jpg"
        dst = out_root / s / file_name
        if not dst.exists():
            shutil.copyfile(gt.image_path, dst)
            n_copied += 1
        data[s]["images"].append({"id": img_id, "file_name": file_name,
                                  "width": gt.width, "height": gt.height})
        counters[s]["img"] += 1

        # objects -> global annotation ids for this split
        obj2ann: dict[int, int] = {}
        for oi, o in enumerate(gt.objects):
            label_ids.setdefault(o.label, len(label_ids))
            x1, y1, x2, y2 = o.box
            bx, by = x1 * gt.width, y1 * gt.height
            bw, bh = (x2 - x1) * gt.width, (y2 - y1) * gt.height
            aid = next_ann[s]
            next_ann[s] += 1
            obj2ann[oi] = aid
            data[s]["annotations"].append({
                "id": aid, "image_id": img_id,
                # +1: SGG-Benchmark reserves object index 0 for __background__
                "category_id": label_ids[o.label] + 1,
                "bbox": [round(bx, 1), round(by, 1), round(bw, 1), round(bh, 1)],
                "area": round(bw * bh, 1), "iscrowd": 0,
            })
            counters[s]["ann"] += 1

        def add(variant: str, subj: int, obj: int, pred: str):
            if pred not in pred_to_id or subj not in obj2ann or obj not in obj2ann:
                return
            rid = next_rel[s][variant]
            next_rel[s][variant] += 1
            data[s][variant].append({
                "id": rid, "image_id": img_id,
                "subject_id": obj2ann[subj], "object_id": obj2ann[obj],
                # +1: relation index 0 is reserved for __no_relation__
                "predicate_id": pred_to_id[pred] + 1,
            })
            counters[s][variant] += 1

        for r in gt.relations:                 # human arm: gold everywhere
            add("human", r.subject, r.object, r.predicate)
        if s == "test":                        # auto arm: gold on TEST only
            for r in gt.relations:
                add("auto", r.subject, r.object, r.predicate)
        else:                                  # train/val: the tool's labels
            for subj, obj, pred in auto.get(gt.image_id, []):
                add("auto", subj, obj, pred)
        if "vlm" in variants:
            if s == "test":                    # test is human gold for every arm
                for r in gt.relations:
                    add("vlm", r.subject, r.object, r.predicate)
            else:
                for subj, obj, pred in vlm.get(gt.image_id, []):
                    add("vlm", subj, obj, pred)

    # index 0 = __background__; object classes occupy 1..6 (see category_id +1)
    categories = [{"id": 0, "name": "__background__", "supercategory": "none"}]
    categories += [{"id": i + 1, "name": n, "supercategory": "none"}
                   for n, i in sorted(label_ids.items(), key=lambda kv: kv[1])]
    for s in SPLITS:
        for variant in variants:
            doc = {"images": data[s]["images"],
                   "annotations": data[s]["annotations"],
                   "categories": categories,
                   "rel_categories": rel_categories,
                   "rel_annotations": data[s][variant]}
            p = out_root / s / f"_annotations.{variant}.coco.json"
            p.write_text(json.dumps(doc), encoding="utf-8")

    print(f"images copied: {n_copied}")
    head = f"{'split':<6} {'images':>7} {'objects':>8}" +            "".join(f"{v + ' rels':>12}" for v in variants)
    print(head)
    for s in SPLITS:
        c = counters[s]
        print(f"{s:<6} {c['img']:>7} {c['ann']:>8}"
              + "".join(f"{c[v]:>12}" for v in variants))
    print(f"\nclasses: {[c['name'] for c in categories]}")
    print(f"predicates: {list(pred_to_id)}")
    print(f"-> {out_root}/  (copy the chosen variant to _annotations.coco.json)")


if __name__ == "__main__":
    main()
