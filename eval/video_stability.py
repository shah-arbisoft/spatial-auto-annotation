"""Temporal persistence of the emitted triplets on the demonstration clips.

Chapter 4's video section reports how often a predicate, once emitted for a
pair of tracked objects, keeps being emitted while both objects stay visible.
`scripts/run_video.py` writes the per-frame records but only prints
frame-to-frame Jaccard; this script derives the persistence figures from
those records so the reported numbers come from a command rather than from a
one-off calculation.

Definitions, stated because the figure depends on them:

  co-visible   a frame in which both track ids appear in the frame's own
               track map, so the pair could have been labelled at all
  persistence  for a (pair, predicate) emitted in at least one co-visible
               frame, the fraction of its co-visible frames that carry it
  eligible     pairs with at least `--min-frames` co-visible frames, below
               which the ratio is too coarse to mean anything

Raw and smoothed are both reported: the gap between them is what the
temporal majority vote actually buys.

    python eval/video_stability.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PREDICATES = ["on", "under", "to the left of", "to the right of",
              "in front of", "behind", "near"]


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def persistence(recs: list[dict], key: str, min_frames: int) -> dict:
    covis: dict = defaultdict(int)
    emitted: dict = defaultdict(int)
    for r in recs:
        # "tracks" maps detection index -> track id, and the triplets are
        # written in track ids, so the values are what must be compared.
        ids = sorted(set(r["tracks"].values()))
        present = {tuple(t) for t in r.get(key, [])}
        for a in ids:
            for b in ids:
                if a == b:
                    continue
                for p in PREDICATES:
                    covis[(a, b, p)] += 1
                    if (a, p, b) in present:
                        emitted[(a, b, p)] += 1

    elig = [k for k in covis if covis[k] >= min_frames and emitted[k] > 0]
    if not elig:
        return {"eligible": 0}
    ratios = np.array([emitted[k] / covis[k] for k in elig])
    return {
        "eligible": len(elig),
        "mean_persistence": float(ratios.mean()),
        "share_above_90": float((ratios >= 0.90).mean()),
        "median": float(np.median(ratios)),
    }


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/video")
    ap.add_argument("--min-frames", type=int, default=20)
    ap.add_argument("--out", default="outputs/video_stability.json")
    args = ap.parse_args()

    root = Path(args.dir)
    clips = sorted(p for p in root.iterdir()
                   if p.is_dir() and (p / "frames.jsonl").exists())
    result = {}
    for c in clips:
        recs = load(c / "frames.jsonl")
        raw = persistence(recs, "triplets_raw", args.min_frames)
        sm = persistence(recs, "triplets_smoothed", args.min_frames)
        rs = [{tuple(t) for t in r["triplets_raw"]} for r in recs]
        ss = [{tuple(t) for t in r["triplets_smoothed"]} for r in recs]
        entry = {
            "frames": len(recs),
            "raw": raw,
            "smoothed": sm,
            "jaccard_raw": float(np.mean([jaccard(rs[i], rs[i + 1])
                                          for i in range(len(rs) - 1)])),
            "jaccard_smoothed": float(np.mean([jaccard(ss[i], ss[i + 1])
                                               for i in range(len(ss) - 1)])),
        }
        result[c.name] = entry
        print(f"\n{c.name}: {entry['frames']} frames, "
              f"{sm['eligible']} eligible pair-predicates "
              f"(co-visible >= {args.min_frames} frames)")
        print(f"  persistence   raw {raw['mean_persistence']:.3f}  "
              f"smoothed {sm['mean_persistence']:.3f}")
        print(f"  share >= 0.90 raw {raw['share_above_90']:.2f}  "
              f"smoothed {sm['share_above_90']:.2f}")
        print(f"  frame-to-frame Jaccard  raw {entry['jaccard_raw']:.3f}  "
              f"smoothed {entry['jaccard_smoothed']:.3f}")

    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
