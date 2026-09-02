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
from src.anonymise import face_regions, overlap_frac, pixelate
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


PANEL_W, PANEL_H, MAX_ZOOM = 208, 156, 14


def _inset(img, boxes, colours, W, H):
    """A magnified view of the two claim objects, placed clear of both.

    The median claim object is about 30 px across and the smallest under 10, so
    a full-frame view alone asks the auditor to judge a relation between things
    they cannot resolve. Marking those "not sure" is not neutral: the
    instruction sheet converts uncertainty to a wrong verdict, so unreadable
    items push the precision estimate down rather than widening it.

    The crop covers both objects with margin, magnified up to MAX_ZOOM, and is
    dropped in whichever corner is furthest from them so it hides nothing that
    matters. Returns the panel rectangle, for the label placer to avoid.
    """
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    pad = max(12.0, 0.35 * max(x2 - x1, y2 - y1))
    cx1, cy1 = max(0.0, x1 - pad), max(0.0, y1 - pad)
    cx2, cy2 = min(float(W), x2 + pad), min(float(H), y2 + pad)
    # widen the crop to the panel's aspect so the magnified view is not squashed
    cw, ch = max(1.0, cx2 - cx1), max(1.0, cy2 - cy1)
    zoom = min(MAX_ZOOM, PANEL_W / cw, PANEL_H / ch)
    if zoom <= 1.05:
        return None                       # already legible; a panel would only cover things
    want_w, want_h = PANEL_W / zoom, PANEL_H / zoom
    mx, my = (cx1 + cx2) / 2, (cy1 + cy2) / 2
    cx1 = min(max(0.0, mx - want_w / 2), W - want_w)
    cy1 = min(max(0.0, my - want_h / 2), H - want_h)
    crop = img.crop((int(cx1), int(cy1), int(cx1 + want_w), int(cy1 + want_h)))
    panel = crop.resize((PANEL_W, PANEL_H), Image.LANCZOS)
    pd = ImageDraw.Draw(panel)
    for b, c in zip(boxes, colours):
        pd.rectangle([(b[0] - cx1) * zoom - 2, (b[1] - cy1) * zoom - 2,
                      (b[2] - cx1) * zoom + 2, (b[3] - cy1) * zoom + 2],
                     outline=c, width=2)

    # the corner furthest from the objects, so the panel covers nothing relevant
    corners = [(0.0, 0.0), (W - PANEL_W, 0.0),
               (0.0, H - PANEL_H - 24), (W - PANEL_W, H - PANEL_H - 24)]
    px, py = max(corners, key=lambda c: (c[0] + PANEL_W / 2 - mx) ** 2
                                        + (c[1] + PANEL_H / 2 - my) ** 2)
    img.paste(panel, (int(px), int(py)))
    d = ImageDraw.Draw(img)
    d.rectangle([px, py, px + PANEL_W, py + PANEL_H], outline="#ffd400", width=2)
    return (px, py, px + PANEL_W + 2, py + PANEL_H + 2)


def render(sample_id, image_path, geo, subj, obj, predicate, out_png,
           anonymise=True, zoom=True):
    """Draw one claim. Returns (faces_masked, claim_object_obscured).

    Anonymisation is on by default and happens before anything is drawn, so the
    saved file never holds an unmasked face even transiently. An audit pack is
    sent to a third-party API and read by a person, which is the same disclosure
    Section 3.12 covers for the validation website, and 35% of these frames carry
    an annotated human. The caller is told when a mask lands on a claim object,
    because that item is no longer judgeable and the pack has to account for it.
    """
    img = Image.fromarray(load_rgb(image_path))
    W, H = img.size
    obscured = False
    if anonymise:
        regions = face_regions((W, H), geo)
        by = {o["idx"]: o for o in geo}
        for idx in (subj, obj):
            if by[idx].get("label") == "human":
                continue          # the claim is about the person; masking the head is fine
            b = [v * s for v, s in zip(by[idx]["box"], (W, H, W, H))]
            if any(overlap_frac(b, r) > 0.25 for r in regions):
                obscured = True
        pixelate(img, regions)
    by_idx = {o["idx"]: o for o in geo}
    boxes = [[v * s for v, s in zip(by_idx[i]["box"], (W, H, W, H))]
             for i in (subj, obj)]
    # the magnified view is taken from the clean frame, before the full-size
    # markers are drawn, so the panel shows the objects rather than the boxes
    panel = _inset(img, boxes, ("#e6194b", "#4363d8"), W, H) if zoom else None
    dr = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    # the caption strip is reserved first, so no tag can be pushed under it
    taken = [(0.0, H - 24.0, float(W), float(H))]
    if panel:
        taken.append(panel)
    for idx, colour, role in ((subj, "#e6194b", "SUBJECT"), (obj, "#4363d8", "OBJECT")):
        o = by_idx[idx]
        x1, y1, x2, y2 = [v * s for v, s in zip(o["box"], (W, H, W, H))]
        # The objects here are small -- the median is about 30x30 px and the
        # smallest are under 10 -- so a fixed 4 px outline drawn inward covered
        # more than half of the median claim object and all of the smallest. The
        # marker has to sit OUTSIDE the box, and scale with it, or it hides the
        # thing the auditor is being asked to look at.
        lw = max(1, min(3, int(min(x2 - x1, y2 - y1) // 8)))
        dr.rectangle([max(0, x1 - lw), max(0, y1 - lw),
                      min(W - 1, x2 + lw), min(H - 1, y2 + lw)],
                     outline=colour, width=lw)
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
    return (len(face_regions((W, H), geo)) if anonymise else 0), obscured


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
