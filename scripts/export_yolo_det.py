"""Export the ground-truth boxes as an Ultralytics YOLO detection dataset.

REACT++ (SGG-Benchmark) uses a YOLO backbone pretrained on the dataset's own
object classes; this exports datasets/spatial_sgg_yolo/ so that detector can
be trained once and shared by both experiment arms (the arms differ only in
relation labels, never in boxes).

Layout (Ultralytics convention):
    datasets/spatial_sgg_yolo/
        images/{train,val,test}/group_stem.jpg
        labels/{train,val,test}/group_stem.txt   (class cx cy w h, normalised)
        data.yaml

Same split as export_sgg_benchmark.py: train 0-4, val 5, test 6-8. Class ids
follow the same alphabetical-by-first-seen order as the COCO export (the two
exports print their class lists; they must match).

    python scripts/export_yolo_det.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import SpatialDataset
from src.pipeline import load_config
from export_sgg_benchmark import SPLITS, split_of


def main():
    cfg = load_config("configs/default.yaml")
    ds = SpatialDataset(cfg["dataset"]["root"])
    root = Path("datasets/spatial_sgg_yolo")
    for s in SPLITS:
        (root / "images" / s).mkdir(parents=True, exist_ok=True)
        (root / "labels" / s).mkdir(parents=True, exist_ok=True)

    label_ids: dict[str, int] = {}
    counts = {s: 0 for s in SPLITS}
    for gt in ds:
        group, stem = gt.image_id.split("/")
        s = split_of(group)
        if not gt.image_path.exists() or len(gt.objects) == 0:
            continue
        name = f"{group}_{stem}"
        dst = root / "images" / s / f"{name}.jpg"
        if not dst.exists():
            shutil.copyfile(gt.image_path, dst)
        lines = []
        for o in gt.objects:
            label_ids.setdefault(o.label, len(label_ids))
            x1, y1, x2, y2 = o.box  # already normalised
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            lines.append(f"{label_ids[o.label]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (root / "labels" / s / f"{name}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        counts[s] += 1

    names = [n for n, _ in sorted(label_ids.items(), key=lambda kv: kv[1])]
    (root / "data.yaml").write_text(
        "path: .\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n", encoding="utf-8")
    print({s: c for s, c in counts.items()}, "->", root)
    print("classes:", names)


if __name__ == "__main__":
    main()
