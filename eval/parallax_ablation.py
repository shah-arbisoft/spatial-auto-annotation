"""A9: can multi-frame parallax order front/behind better than monocular depth?

The released images are consecutive frames of one robot walk (§4.14), so for
any annotated image there exist neighbouring frames taken from a different
camera position. That makes a second, independent depth cue available, and
one that needs no learned model and no calibration: under camera translation
a nearer object sweeps further across the image than a far one, so residual
image displacement orders objects by depth directly.

This is worth testing because it is the obvious objection to §4.14's reading.
If the front/behind shortfall were caused by weak monocular depth, then a
geometric cue should beat a learned one and recall should rise. If §4.14 is
right and the shortfall is mostly definitional, better depth should change
very little.

Method, per annotated image:

  1. take the frame `gap` positions later in the raw capture
  2. track each ground-truth box with Lucas-Kanade on features inside it
  3. subtract the median displacement over all objects, which removes the
     camera's rotation, under which everything moves alike regardless of depth
  4. read the residual magnitude as an inverse depth: bigger means nearer

Scored only against annotator groups that use the standard front/behind
convention, since the two inverted groups (§4.5) would penalise any method
that is geometrically right, and the question here is about geometry.

    python eval/parallax_ablation.py --gap 20
    python eval/parallax_ablation.py --sweep 10,20,30,40
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

INVERTED = {"group_6", "group_8"}
MAX_FRAME = 2649


def load_frame(raw_dir: Path, n: int) -> np.ndarray:
    return np.asarray(ImageOps.exif_transpose(
        Image.open(raw_dir / f"{n:06d}.jpg")).convert("L"))


def box_shift(cv2, a, b, box_px):
    """Median displacement of tracked corners inside a box, or None."""
    x1, y1, x2, y2 = (int(v) for v in box_px)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(a.shape[1] - 1, x2), min(a.shape[0] - 1, y2)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    mask = np.zeros_like(a)
    mask[y1:y2, x1:x2] = 255
    pts = cv2.goodFeaturesToTrack(a, maxCorners=60, qualityLevel=0.01,
                                  minDistance=3, mask=mask)
    if pts is None or len(pts) < 4:
        return None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, winSize=(21, 21),
                                          maxLevel=3)
    ok = st.ravel() == 1
    if ok.sum() < 4:
        return None
    return np.median((nxt[ok] - pts[ok]).reshape(-1, 2), axis=0)


def triangulated_depths(cv2, a, b, boxes_px, focal_mult=0.9):
    """Per-object depth from two views, by essential matrix + triangulation.

    The residual-magnitude reading below assumes displacement falls off with
    depth alone. It does not: under forward motion the flow is radial from
    the focus of expansion, so an object near that point barely moves however
    close it is. Recovering the pose and triangulating handles any camera
    motion, at the cost of needing intrinsics. No calibration exists for this
    capture, so focal length is assumed at `focal_mult * W`. Whether that
    matters is not asserted here but measured: `--focal-sweep` re-runs the
    ablation across a range of assumptions and reports whether the ordering
    accuracy moves (Appendix D.6).

    Returns {object index: depth} for objects whose points triangulate.
    """
    H, W = a.shape
    f = focal_mult * W
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float64)

    # correspondences over the whole image drive the pose estimate
    pts = cv2.goodFeaturesToTrack(a, maxCorners=1200, qualityLevel=0.01,
                                  minDistance=6)
    if pts is None or len(pts) < 30:
        return {}
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, winSize=(21, 21),
                                          maxLevel=3)
    ok = st.ravel() == 1
    p0, p1 = pts[ok].reshape(-1, 2), nxt[ok].reshape(-1, 2)
    if len(p0) < 30:
        return {}
    E, inl = cv2.findEssentialMat(p0, p1, K, method=cv2.RANSAC,
                                  prob=0.999, threshold=1.0)
    if E is None or E.shape != (3, 3):
        return {}
    _, R, t, _ = cv2.recoverPose(E, p0, p1, K)

    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([R, t])

    out = {}
    for i, (x1, y1, x2, y2) in boxes_px.items():
        inside = ((p0[:, 0] >= x1) & (p0[:, 0] <= x2) &
                  (p0[:, 1] >= y1) & (p0[:, 1] <= y2))
        if inside.sum() < 5:
            continue
        X = cv2.triangulatePoints(P0, P1, p0[inside].T, p1[inside].T)
        X = (X[:3] / X[3]).T
        z = X[:, 2]
        z = z[np.isfinite(z) & (z > 0)]
        if z.size >= 3:
            out[i] = float(np.median(z))
    return out


def gold_pairs(pairs_csv: str) -> dict:
    """{image_id: {(subj, obj): gold_predicates}} for depth-labelled pairs."""
    out: dict = defaultdict(dict)
    with open(pairs_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            gold = {p for p in r["gold"].split(";") if p}
            pred = {p for p in r["pred"].split(";") if p}
            if gold & {"in front of", "behind"}:
                out[r["image_id"]][(int(r["subj"]), int(r["obj"]))] = (gold, pred)
    return dict(out)


def run(ds, gold, raw_dir, gap, method="residual", verbose=True,
        focal_mult=0.9) -> dict:
    import cv2  # noqa: PLC0415

    n_img = n_pair = 0
    par_hit = mono_hit = both = 0
    agree = 0
    skipped_img = 0

    for image_id, pairs in sorted(gold.items()):
        group, stem = image_id.split("/")
        if group in INVERTED:
            continue
        n = int(stem)
        if n + gap > MAX_FRAME:
            continue
        im = ds.get(image_id)
        if im is None:
            continue
        H, W = im.height, im.width
        try:
            a, b = load_frame(raw_dir, n), load_frame(raw_dir, n + gap)
        except FileNotFoundError:
            continue

        boxes = {i: (o.box[0] * W, o.box[1] * H, o.box[2] * W, o.box[3] * H)
                 for i, o in enumerate(im.objects)}

        if method == "triangulate":
            depths = triangulated_depths(cv2, a, b, boxes, focal_mult)
            if len(depths) < 3:
                skipped_img += 1
                continue
            # nearer means smaller depth, so invert to keep "bigger = nearer"
            resid = {i: -z for i, z in depths.items()}
        else:
            shifts = {}
            for i, bx in boxes.items():
                s = box_shift(cv2, a, b, bx)
                if s is not None:
                    shifts[i] = s
            if len(shifts) < 3:
                skipped_img += 1
                continue
            # remove the rotational component: under pure rotation every
            # object moves alike, so the median carries no depth
            med = np.median(np.stack(list(shifts.values())), axis=0)
            resid = {i: float(np.linalg.norm(v - med)) for i, v in shifts.items()}

        n_img += 1
        for (si, oi), (g, pred) in pairs.items():
            if si not in resid or oi not in resid:
                continue
            n_pair += 1
            truth = "in front of" if "in front of" in g else "behind"
            # bigger residual = nearer the camera = in front of
            par = "in front of" if resid[si] > resid[oi] else "behind"
            mono = ("in front of" if "in front of" in pred
                    else "behind" if "behind" in pred else None)
            if par == truth:
                par_hit += 1
            if mono == truth:
                mono_hit += 1
            if mono is not None and par == mono:
                agree += 1
            if mono is not None:
                both += 1

    return {
        "method": method,
        "gap": gap,
        "focal_mult": focal_mult,
        "images": n_img,
        "images_skipped": skipped_img,
        "pairs": n_pair,
        "parallax_accuracy": par_hit / n_pair if n_pair else None,
        "monocular_accuracy": mono_hit / n_pair if n_pair else None,
        "pairs_monocular_committed": both,
        "agreement_parallax_vs_monocular": agree / both if both else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=r"D:\uni_project\extended_raw_data\rightimg")
    ap.add_argument("--root", default=None)
    ap.add_argument("--pairs", default="outputs/pairs.csv")
    ap.add_argument("--gap", type=int, default=20)
    ap.add_argument("--sweep", default=None)
    ap.add_argument("--method", default="residual",
                    choices=["residual", "triangulate"])
    ap.add_argument("--limit", type=int, default=0, help="cap images, for a quick run")
    ap.add_argument("--focal-sweep", default=None,
                    help="comma-separated focal multipliers of image width, "
                         "e.g. 0.5,0.7,0.9,1.2,1.6,2.0; triangulate only")
    ap.add_argument("--out", default="outputs/parallax_ablation.json")
    args = ap.parse_args()

    root = args.root
    if root is None:
        for line in Path("configs/default.yaml").read_text(encoding="utf-8").splitlines():
            if "root:" in line:
                root = line.split("root:", 1)[1].strip().strip("\"'")
                break

    ds = {im.image_id: im for im in SpatialDataset(root)}
    gold = gold_pairs(args.pairs)
    if args.limit:
        gold = dict(sorted(gold.items())[:args.limit])
    print(f"{len(gold)} annotated images carry a front/behind label "
          f"({sum(len(v) for v in gold.values())} such pairs)")

    gaps = [int(g) for g in args.sweep.split(",")] if args.sweep else [args.gap]
    focals = ([float(f) for f in args.focal_sweep.split(",")]
              if args.focal_sweep else [0.9])
    if len(focals) > 1 and args.method != "triangulate":
        raise SystemExit("--focal-sweep only affects the triangulate method")

    rows = []
    for g in gaps:
      for fm in focals:
        r = run(ds, gold, Path(args.raw), g, args.method, focal_mult=fm)
        rows.append(r)
        if len(focals) > 1:
            print(f"  focal {fm:.2f}xW: ordering accuracy "
                  f"{r['parallax_accuracy']:.3f} on {r['pairs']} pairs")
            continue
        print(f"\n[{args.method}] gap {g:3d} frames: {r['images']} images, {r['pairs']} pairs "
              f"({r['images_skipped']} images skipped, too few trackable objects)")
        print(f"  multi-frame ordering accuracy {r['parallax_accuracy']:.3f}")
        print(f"  monocular cascade accuracy   {r['monocular_accuracy']:.3f}")
        print(f"  the two agree on             "
              f"{r['agreement_parallax_vs_monocular']:.3f} of pairs where the "
              f"cascade commits")

    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
