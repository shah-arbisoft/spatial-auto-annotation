"""Does annotating keyframes only cost anything?

A robot camera oversamples: consecutive frames of a slow walk carry almost
identical content, so a per-frame pipeline pays a full perception pass to
recompute relations that have not changed. `src/keyframes.py` segments a
sequence by content drift and nominates one keyframe per segment. This script
measures what that costs, on the one sequence where the answer can be checked
against human annotation.

Two measurements, one pass:

  STABILITY   Propagate the keyframe's computed predicates to every other
              frame in its segment (objects matched by label and box overlap)
              and compare against what the pipeline computes on that frame
              directly. No human labels involved, so this also runs on
              unannotated footage. A predicate that survives a viewpoint
              change was geometrically determined; one that flips was a coin
              toss, which makes this an unsupervised reliability signal for
              the depth pair specifically.

  FIDELITY    Score the propagated predicates against the frame's own human
              triplets and compare with the per-frame pipeline's recall. This
              is the number that decides whether keyframe selection is free:
              if recall is unchanged, the skipped frames held no information
              the keyframe did not already carry.

    python eval/keyframe_propagation.py --tau 10
    python eval/keyframe_propagation.py --sweep 5,10,15,20,25
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import SpatialDataset
from src.keyframes import thumbnail, segment_sequence

PREDICATES = ["on", "under", "to the left of", "to the right of",
              "in front of", "behind", "near"]
MIN_IOU = 0.50


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_pairs(path: str) -> dict:
    """{image_id: {(subj, obj): (pred_set, gold_set)}} from pairs.csv."""
    out: dict = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = (int(r["subj"]), int(r["obj"]))
            pred = {p for p in r["pred"].split(";") if p}
            gold = {p for p in r["gold"].split(";") if p}
            out[r["image_id"]][key] = (pred, gold)
    return dict(out)


def match_objects(src_objs, dst_objs) -> dict[int, int]:
    """Greedy same-label, highest-overlap mapping dst index -> src index.

    Object indices are per-frame and unstable: the annotators recorded
    different subsets of the scene from frame to frame (group_0 alone has 43
    distinct label orderings), so propagation cannot go by position.
    """
    cand = []
    for di, d in enumerate(dst_objs):
        for si, s in enumerate(src_objs):
            if d.label != s.label:
                continue
            v = iou(d.box, s.box)
            if v >= MIN_IOU:
                cand.append((v, di, si))
    cand.sort(reverse=True)
    used_d, used_s, out = set(), set(), {}
    for v, di, si in cand:
        if di in used_d or si in used_s:
            continue
        used_d.add(di)
        used_s.add(si)
        out[di] = si
    return out


def run(ds, pairs, thumbs, tau, verbose=True) -> dict:
    frames_by_group: dict[str, list[str]] = defaultdict(list)
    for image_id in sorted(pairs):
        frames_by_group[image_id.split("/")[0]].append(image_id)

    # stability: does the keyframe's verdict survive the viewpoint change?
    agree = defaultdict(int)
    support = defaultdict(int)
    # fidelity: recall of human triplets, propagated vs computed per frame
    hit_prop = defaultdict(int)
    hit_frame = defaultdict(int)
    gold_n = defaultdict(int)

    n_frames = n_keys = 0
    unmatched = matched = 0

    for group, ids in sorted(frames_by_group.items()):
        th = [thumbs[i] for i in ids]
        segs = segment_sequence(th, tau)
        n_frames += len(ids)
        n_keys += len(segs)
        for seg in segs:
            kid = ids[seg.keyframe]
            kobjs = ds[kid].objects
            for fi in seg.frames:
                fid = ids[fi]
                if fid == kid:
                    continue
                fobjs = ds[fid].objects
                m = match_objects(kobjs, fobjs)
                for (a, b), (pred_f, gold) in pairs[fid].items():
                    if a not in m or b not in m:
                        unmatched += 1
                        continue
                    matched += 1
                    pred_k = pairs.get(kid, {}).get((m[a], m[b]), (set(), set()))[0]
                    for p in PREDICATES:
                        if p in pred_f:
                            support[p] += 1
                            if p in pred_k:
                                agree[p] += 1
                        if p in gold:
                            gold_n[p] += 1
                            if p in pred_k:
                                hit_prop[p] += 1
                            if p in pred_f:
                                hit_frame[p] += 1

    def table(hit, tot):
        return {p: (hit[p] / tot[p]) if tot[p] else None for p in PREDICATES}

    stab = table(agree, support)
    rec_p = table(hit_prop, gold_n)
    rec_f = table(hit_frame, gold_n)
    mean = lambda d: float(np.mean([v for v in d.values() if v is not None]))

    return {
        "tau": tau,
        "frames": n_frames,
        "keyframes": n_keys,
        "compression": n_frames / n_keys if n_keys else None,
        "pairs_matched": matched,
        "pairs_unmatched": unmatched,
        "support": dict(support),
        "gold": dict(gold_n),
        "stability": stab,
        "recall_propagated": rec_p,
        "recall_per_frame": rec_f,
        "mean_stability": mean(stab),
        "mean_recall_propagated": mean(rec_p),
        "mean_recall_per_frame": mean(rec_f),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="dataset root")
    ap.add_argument("--pairs", default="outputs/pairs.csv")
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--sweep", default=None, help="comma-separated taus")
    ap.add_argument("--out", default="outputs/keyframe_propagation.json")
    args = ap.parse_args()

    root = args.root
    if root is None:
        cfg = Path("configs/default.yaml").read_text(encoding="utf-8")
        for line in cfg.splitlines():
            if "root:" in line:
                root = line.split("root:", 1)[1].strip().strip('"\'')
                break

    ds_iter = SpatialDataset(root)
    ds = {im.image_id: im for im in ds_iter}
    pairs = load_pairs(args.pairs)
    pairs = {k: v for k, v in pairs.items() if k in ds}
    print(f"{len(pairs)} annotated frames with pairs")

    thumbs = {}
    for image_id, im in ds.items():
        if image_id in pairs:
            img = ImageOps.exif_transpose(Image.open(im.image_path))
            thumbs[image_id] = thumbnail(np.asarray(img.convert("L")))

    taus = ([float(t) for t in args.sweep.split(",")] if args.sweep
            else [args.tau])
    results = []
    for tau in taus:
        r = run(ds, pairs, thumbs, tau)
        results.append(r)
        print(f"\ntau={tau:g}  {r['keyframes']}/{r['frames']} keyframes "
              f"({r['compression']:.1f}x)  matched pairs {r['pairs_matched']} "
              f"(unmatched {r['pairs_unmatched']})")
        print(f"{'predicate':16s} {'stability':>10s} {'rec(prop)':>10s} "
              f"{'rec(frame)':>11s} {'support':>8s}")
        for p in PREDICATES:
            s, rp, rf = r["stability"][p], r["recall_propagated"][p], r["recall_per_frame"][p]
            fmt = lambda v: f"{v:.3f}" if v is not None else "    —"
            print(f"{p:16s} {fmt(s):>10s} {fmt(rp):>10s} {fmt(rf):>11s} "
                  f"{r['support'].get(p, 0):8d}")
        print(f"{'MEAN':16s} {r['mean_stability']:10.3f} "
              f"{r['mean_recall_propagated']:10.3f} "
              f"{r['mean_recall_per_frame']:11.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
