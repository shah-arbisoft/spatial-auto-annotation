"""Build the manual-audit pack for RQ1's true-precision estimate.

Human annotations cover ~10% of pairs, so an automatic label absent from the
gold is not necessarily wrong. This script samples the tool's EXTRA predictions
(predicate emitted, human did not label it on that ordered pair), stratified
per predicate, renders each as an image with the subject/object boxes drawn,
and writes a verdict sheet. A human marks each sampled triplet correct/wrong;
the per-predicate fraction correct estimates the tool's true precision.

    python scripts/make_audit_pack.py --per-predicate 15 --seed 42

Outputs: outputs/audit/audit_sheet.csv + outputs/audit/img/NNN.png
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import SpatialDataset, load_rgb
from src.pipeline import load_config
from src.predicates import PREDICATES


def _hits(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _place(w, h, box, W, H, taken):
    """Find a home for a w x h tag that touches its own box but nothing placed.

    Overlapping objects are the norm here rather than the exception -- support
    claims are about things in contact -- so anchoring both tags to the top-left
    of their boxes hid one behind the other on exactly the items where knowing
    which is which matters most. Eight positions around the box are tried in
    preference order, all of them adjacent to the box so the tag stays readable
    as belonging to it; only if every one collides does the tag step downwards.
    """
    x1, y1, x2, y2 = box
    cands = [(ax, ay)
             for ay in (y1 - h - 2, y2 + 2, y1 + 2, y2 - h - 2)
             for ax in (x1, x2 - w)]
    for cx, cy in cands:
        cx = min(max(0.0, cx), W - w)
        cy = min(max(0.0, cy), H - h)
        r = (cx, cy, cx + w, cy + h)
        if not any(_hits(r, t) for t in taken):
            return r
    cx = min(max(0.0, x1), W - w)
    cy = min(max(0.0, y1 - h - 2), H - h)
    while any(_hits((cx, cy, cx + w, cy + h), t) for t in taken) and cy + h < H:
        cy += h + 2
    return (cx, cy, cx + w, cy + h)


def render(sample_id, image_path, geo, subj, obj, predicate, out_png):
    img = Image.fromarray(load_rgb(image_path))
    W, H = img.size
    dr = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    by_idx = {o["idx"]: o for o in geo}
    # the caption strip is reserved first, so no tag can be pushed under it
    taken = [(0.0, H - 24.0, float(W), float(H))]
    for idx, colour, role in ((subj, "#e6194b", "SUBJECT"), (obj, "#4363d8", "OBJECT")):
        o = by_idx[idx]
        x1, y1, x2, y2 = [v * s for v, s in zip(o["box"], (W, H, W, H))]
        dr.rectangle([x1, y1, x2, y2], outline=colour, width=4)
        tag = f"{role}: {o['label']}{idx}"
        w, h = dr.textlength(tag, font=font) + 8, 18.0
        r = _place(w, h, (x1, y1, x2, y2), W, H, taken)
        taken.append(r)
        dr.rectangle(list(r), fill=colour)
        dr.text((r[0] + 4, r[1] + 1), tag, fill="white", font=font)
    caption = (f"#{sample_id}:  {by_idx[subj]['label']}{subj} "
               f"is {predicate} {by_idx[obj]['label']}{obj}")
    dr.rectangle([0, H - 24, W, H], fill="black")
    dr.text((6, H - 21), caption, fill="white", font=font)
    img.save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="outputs/pairs.csv")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--per-predicate", type=int, default=15)
    ap.add_argument("--predicates", default=None,
                    help="comma-separated subset, e.g. 'on,under' (default: all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/audit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rng = random.Random(args.seed)

    # collect extra predictions per predicate
    extras = {k: [] for k in PREDICATES}
    with open(args.pairs, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pred = set(r["pred"].split(";")) if r["pred"] else set()
            gold = set(r["gold"].split(";")) if r["gold"] else set()
            for k in pred - gold:
                extras[k].append((r["image_id"], int(r["subj"]), int(r["obj"])))

    ds = SpatialDataset(cfg["dataset"]["root"])
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)

    wanted = (args.predicates.split(",") if args.predicates else list(PREDICATES))
    rows = []
    sample_id = 0
    for k in [k for k in PREDICATES if k in wanted]:
        pool = extras[k]
        take = rng.sample(pool, min(args.per_predicate, len(pool)))
        for image_id, subj, obj in take:
            sample_id += 1
            group, stem = image_id.split("/")
            geo = json.loads(Path(f"outputs/geometry/{group}/{stem}.json")
                             .read_text(encoding="utf-8"))
            image_path = Path(cfg["dataset"]["root"]) / "img_data" / group / f"{stem}.jpg"
            png = out / "img" / f"{sample_id:03d}.png"
            render(sample_id, image_path, geo, subj, obj, k, png)
            by_idx = {o["idx"]: o for o in geo}
            rows.append({
                "id": sample_id, "image": png.name, "image_id": image_id,
                "subject": f"{by_idx[subj]['label']}{subj}",
                "predicate": k,
                "object": f"{by_idx[obj]['label']}{obj}",
                "verdict (y/n)": "", "notes": "",
            })

    with open(out / "audit_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"audit pack: {len(rows)} samples -> {out}/audit_sheet.csv + {out}/img/")
    print("Mark each triplet y (correct) or n (wrong) in the sheet; "
          "per-predicate %y estimates true precision.")


if __name__ == "__main__":
    main()
