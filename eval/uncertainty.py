"""Confidence intervals for the headline recall numbers.

The RQ1 recalls in chapter 4 are computed over every human triplet in the
dataset, so they carry no *sampling* error in the usual sense: they are the
population value for these 836 images. They do carry generalisation
uncertainty, which is the question a reader actually has - would another
batch of images from the same process give the same number? - and that is
what this script estimates.

Method: a CLUSTER bootstrap over images. Triplets inside one image are not
independent (they share objects, a scene layout, a depth map and an
annotator), so resampling triplets would understate the interval. Resampling
whole images with replacement preserves that clustering and is the standard
remedy. Intervals are the 2.5th and 97.5th percentiles over the resamples.

Reported for the pooled set and for the held-out annotator groups (6-8)
separately, because the held-out numbers are the ones that carry the
generalisation claim.

    python eval/uncertainty.py --iters 2000
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PREDICATES = ["on", "under", "to the left of", "to the right of",
              "in front of", "behind", "near"]
HELD_OUT = {"group_6", "group_7", "group_8"}
PAIRS = Path("outputs/pairs.csv")
OUT_JSON = Path("outputs/uncertainty.json")
OUT_MD = Path("outputs/tables/uncertainty.md")


def load_hits():
    """image_id -> list of (predicate, recovered?) for every human triplet."""
    by_image = defaultdict(list)
    with open(PAIRS, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gold = set(r["gold"].split(";")) if r["gold"] else set()
            if not gold:
                continue
            pred = set(r["pred"].split(";")) if r["pred"] else set()
            for k in gold:
                if k in PREDICATES:
                    by_image[r["image_id"]].append((k, k in pred))
    return by_image


def bootstrap(by_image, images, iters, rng):
    """Cluster bootstrap -> {predicate: (recall, lo, hi, n_gold)}."""
    per_img = {}
    for img in images:
        counts = defaultdict(lambda: [0, 0])  # predicate -> [hits, total]
        for k, hit in by_image[img]:
            counts[k][1] += 1
            counts[k][0] += int(hit)
        per_img[img] = {k: tuple(v) for k, v in counts.items()}

    point, draws = {}, {k: [] for k in PREDICATES}
    for k in PREDICATES:
        h = sum(per_img[i].get(k, (0, 0))[0] for i in images)
        n = sum(per_img[i].get(k, (0, 0))[1] for i in images)
        point[k] = (h / n if n else float("nan"), n)

    idx = np.arange(len(images))
    for _ in range(iters):
        pick = rng.choice(idx, size=len(idx), replace=True)
        agg = {k: [0, 0] for k in PREDICATES}
        for j in pick:
            for k, (h, n) in per_img[images[j]].items():
                agg[k][0] += h
                agg[k][1] += n
        for k in PREDICATES:
            h, n = agg[k]
            if n:
                draws[k].append(h / n)

    out = {}
    for k in PREDICATES:
        r, n = point[k]
        if draws[k]:
            lo, hi = np.percentile(draws[k], [2.5, 97.5])
        else:
            lo = hi = float("nan")
        out[k] = {"recall": r, "lo": float(lo), "hi": float(hi), "n_gold": n}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not PAIRS.exists():
        raise SystemExit(f"missing {PAIRS}; run scripts/run_annotator.py first")
    rng = np.random.default_rng(args.seed)
    by_image = load_hits()
    all_imgs = sorted(by_image)
    held = [i for i in all_imgs if i.split("/")[0] in HELD_OUT]
    print(f"images with gold: {len(all_imgs)} (held-out {len(held)}); "
          f"{args.iters} bootstrap resamples")

    pooled = bootstrap(by_image, all_imgs, args.iters, rng)
    heldout = bootstrap(by_image, held, args.iters, rng)

    report = {"iters": args.iters, "seed": args.seed,
              "method": "cluster bootstrap over images (percentile interval)",
              "pooled": pooled, "held_out": heldout}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = ["# Recall with 95% cluster-bootstrap intervals\n",
          f"Resampling whole images with replacement ({args.iters} draws, "
          "seed {}), which preserves the correlation between triplets that "
          "share an image.\n".format(args.seed),
          "| predicate | pooled recall (95% CI) | n | held-out recall (95% CI) | n |",
          "|---|---|---|---|---|"]
    for k in PREDICATES:
        p, h = pooled[k], heldout[k]
        md.append(f"| {k} | {p['recall']:.3f} ({p['lo']:.3f}-{p['hi']:.3f}) | "
                  f"{p['n_gold']} | {h['recall']:.3f} ({h['lo']:.3f}-{h['hi']:.3f}) | "
                  f"{h['n_gold']} |")
    pm = np.mean([pooled[k]["recall"] for k in PREDICATES])
    hm = np.mean([heldout[k]["recall"] for k in PREDICATES])
    md.append(f"| **mean** | **{pm:.3f}** | | **{hm:.3f}** | |")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nreport -> {OUT_JSON} ; table -> {OUT_MD}")


if __name__ == "__main__":
    main()
