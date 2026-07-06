"""Build the per-predicate failure gallery with causes tagged automatically.

For every human triplet the tool misses, the cause is diagnosed by re-checking
the rule's individual conditions against the cached geometry — not by eye:

  front/behind : abstained (|dz| inside depth_eps) | annotator convention
                 inverted (groups 6/8) | depth ordering error
  on/under     : depth gate suppressed | containment (nested boxes) |
                 vertical gap too large | insufficient horizontal overlap |
                 centroid order
  near         : contact-exclusion conflict | gap beyond fitted threshold
  left/right   : abstained (centre band) | centre flip

Outputs: cause-frequency table (outputs/tables/failure_gallery.md), a verdict
sheet (outputs/failure_gallery/gallery.csv), and up to N rendered examples per
(predicate, cause) under outputs/failure_gallery/img/.

    python scripts/make_failure_gallery.py --per-cause 3
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import SpatialDataset, load_rgb
from src.pipeline import load_config, thresholds_from_config
from src.predicates import (Obj, box_gap_rel, evaluate_scene,
                            _vertical_gap, _x_extent_overlap)

INVERTED_GROUPS = {"group_6", "group_8"}
FLIP = {"in front of": "behind", "behind": "in front of"}


def diagnose(k, a, b, pset, group, t):
    """Return the cause tag for a missed gold triplet (a --k--> b)."""
    if k in ("in front of", "behind"):
        if abs(a.depth - b.depth) <= t.depth_eps:
            return "abstained: depths within ambiguity band"
        if group in INVERTED_GROUPS:
            return "annotator convention inverted (measured, groups 6/8)"
        return "depth ordering error"
    if k in ("on", "under"):
        top, bottom = (a, b) if k == "on" else (b, a)
        gap = _vertical_gap(top=top, bottom=bottom)
        touching = -t.on_vertical_gap <= gap <= t.on_vertical_gap
        overlap = _x_extent_overlap(top, bottom) >= t.on_horizontal_overlap
        above = top.cy < bottom.cy
        co = abs(a.depth - b.depth) <= t.on_depth_eps
        if above and touching and overlap and not co:
            return "depth gate suppressed (no co-location)"
        if gap < -t.on_vertical_gap:
            return "containment: boxes nested (shallow view angle)"
        if gap > t.on_vertical_gap:
            return "vertical gap too large"
        if not overlap:
            return "insufficient horizontal overlap"
        if not above:
            return "centroid order (subject not above)"
        return "other"
    if k == "near":
        if pset & {"on", "under"}:
            return "contact-exclusion conflict (tool called it support)"
        if box_gap_rel(a, b) > t.near_T:
            return "gap beyond fitted threshold"
        return "other"
    # lateral
    if abs(a.cx - b.cx) <= t.lateral_center_eps:
        return "abstained: centres within band"
    return "centre flip vs human judgement"


def render(sample_id, image_path, objs, s_idx, o_idx, k, cause, out_png):
    img = Image.fromarray(load_rgb(image_path))
    W, H = img.size
    dr = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    by = {o.idx: o for o in objs}
    for idx, colour, role in ((s_idx, "#e6194b", "SUBJ"), (o_idx, "#4363d8", "OBJ")):
        o = by[idx]
        x1, y1, x2, y2 = [v * s for v, s in zip(o.box, (W, H, W, H))]
        dr.rectangle([x1, y1, x2, y2], outline=colour, width=4)
        tag = f"{role}: {o.label}{idx}"
        ty = max(0, y1 - 19)
        dr.rectangle([x1, ty, x1 + dr.textlength(tag, font=font) + 8, ty + 17], fill=colour)
        dr.text((x1 + 4, ty + 1), tag, fill="white", font=font)
    dr.rectangle([0, H - 40, W, H], fill="black")
    dr.text((6, H - 38), f"#{sample_id} missed: {by[s_idx].label}{s_idx} --{k}--> {by[o_idx].label}{o_idx}",
            fill="white", font=font)
    dr.text((6, H - 20), f"cause: {cause}", fill="#ffd166", font=font)
    img.save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cause", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/failure_gallery")
    args = ap.parse_args()

    cfg = load_config("configs/default.yaml")
    t = thresholds_from_config(cfg)
    ds = SpatialDataset(cfg["dataset"]["root"])
    rng = random.Random(args.seed)

    misses = []  # (predicate, cause, image_id, image_path, objs, s, o)
    counts = collections.Counter()
    for gt in ds:
        group, stem = gt.image_id.split("/")
        gp = Path("outputs/geometry") / group / f"{stem}.json"
        if not gp.exists():
            continue
        geo = json.loads(gp.read_text(encoding="utf-8"))
        objs = [Obj(o["idx"], o["label"], tuple(o["box"]), o["cx"], o["cy"],
                    o["depth"], np.array(o["pos3d"])) for o in geo]
        by = {o.idx: o for o in objs}
        pred = {(p.subject, p.object): set(p.predicates)
                for p in evaluate_scene(objs, t)}
        for r in gt.relations:
            pset = pred.get((r.subject, r.object), set())
            if r.predicate in pset:
                continue
            cause = diagnose(r.predicate, by[r.subject], by[r.object], pset, group, t)
            counts[(r.predicate, cause)] += 1
            misses.append((r.predicate, cause, gt.image_id, gt.image_path,
                           objs, r.subject, r.object))

    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)

    # sample renders per (predicate, cause)
    by_pc = collections.defaultdict(list)
    for m in misses:
        by_pc[(m[0], m[1])].append(m)
    rows, sid = [], 0
    for (k, cause), items in sorted(by_pc.items()):
        for m in rng.sample(items, min(args.per_cause, len(items))):
            sid += 1
            png = out / "img" / f"{sid:03d}.png"
            render(sid, m[3], m[4], m[5], m[6], k, cause, png)
            rows.append({"id": sid, "image": png.name, "image_id": m[2],
                         "predicate": k, "cause": cause,
                         "subject": f"{[o for o in m[4] if o.idx==m[5]][0].label}{m[5]}",
                         "object": f"{[o for o in m[4] if o.idx==m[6]][0].label}{m[6]}"})
    with open(out / "gallery.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # cause-frequency table
    md = ["# Failure gallery — misses by predicate and diagnosed cause\n",
          "| Predicate | Cause | Misses | Share of predicate's misses |",
          "|---|---|---|---|"]
    per_pred = collections.Counter()
    for (k, cause), n in counts.items():
        per_pred[k] += n
    for (k, cause), n in sorted(counts.items(), key=lambda x: (x[0][0], -x[1])):
        md.append(f"| {k} | {cause} | {n} | {100*n/per_pred[k]:.0f}% |")
    md.append(f"\ntotal misses: {sum(counts.values())}; rendered examples: {len(rows)} "
              f"(seeded sample, {args.per_cause} per cause) in `{out}/img/`")
    Path("outputs/tables/failure_gallery.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
