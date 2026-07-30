"""Content-adaptive frame selection: annotate viewpoints, not frames.

A robot camera emits frames far faster than the scene changes. Annotating
every frame spends a full perception pass (detector + SAM2 + depth) to
recompute relations that have not moved, and inflates any per-frame average
with near-duplicates. The fix is to segment the sequence by content and
annotate one representative frame per segment.

The obvious method does not work here. Classic shot detection thresholds the
difference between *consecutive* frames, which assumes cuts. A robot walking
through a room produces no cuts: on the 2,650-frame capture behind this
project's dataset, consecutive-frame differencing finds exactly one boundary
in the whole sequence, because each frame differs from its predecessor by
about 0.08 px of motion. The change is real but arrives gradually.

`segment_sequence` therefore measures drift from the *anchor* of the current
segment rather than from the previous frame, and opens a new segment when
that drift crosses `tau`. Slow motion accumulates instead of being repeatedly
rounded away, and a hard cut still triggers a boundary immediately, so the
one method covers both regimes.

Two uses, one parameter:

  small tau  near-duplicate removal. Each segment is one viewpoint; keep its
             representative and skip the rest (the cheap mode).
  large tau  scene grouping. Each segment holds several viewpoints of the
             same arrangement, which is what the cross-view consistency check
             in `eval/viewpoint_stability.py` consumes.

Distances are computed on 64x48 mean-subtracted greyscale: mean subtraction
discards global exposure shifts, which are otherwise the largest signal in an
auto-exposing camera and would fire spurious boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

THUMB_W, THUMB_H = 64, 48


def thumbnail(img: np.ndarray) -> np.ndarray:
    """Downsample to a mean-subtracted 64x48 greyscale signature.

    Accepts HxW or HxWx3 uint8. Subsampling by stride rather than area
    averaging keeps this cheap enough to run on every frame of a long capture
    before any GPU work is scheduled.
    """
    a = np.asarray(img)
    if a.ndim == 3:
        a = a.mean(axis=2)
    h, w = a.shape
    rows = np.linspace(0, h - 1, THUMB_H).astype(int)
    cols = np.linspace(0, w - 1, THUMB_W).astype(int)
    t = a[np.ix_(rows, cols)].astype(np.float32)
    return t - t.mean()


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference between two thumbnails, in grey levels."""
    return float(np.abs(a - b).mean())


@dataclass
class Segment:
    """A run of frames judged to show the same content."""

    start: int
    end: int                                   # inclusive
    keyframe: int                              # representative frame index
    frames: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frames)


def segment_sequence(thumbs: list[np.ndarray], tau: float,
                     min_len: int = 1) -> list[Segment]:
    """Split a frame sequence into segments of near-constant content.

    A new segment opens when the current frame's distance from the active
    segment's anchor exceeds `tau`. `min_len` suppresses single-frame
    segments caused by a transient (a person crossing the view, an exposure
    step) by folding them into the previous segment.

    The keyframe is the segment's most typical view rather than its first,
    which on a moving camera is usually mid-transition.
    """
    if not thumbs:
        return []
    bounds: list[list[int]] = [[0]]
    anchor = thumbs[0]
    for i in range(1, len(thumbs)):
        if distance(thumbs[i], anchor) > tau:
            bounds.append([i])
            anchor = thumbs[i]
        else:
            bounds[-1].append(i)

    if min_len > 1:
        merged: list[list[int]] = []
        for grp in bounds:
            if merged and len(grp) < min_len:
                merged[-1].extend(grp)
            else:
                merged.append(grp)
        bounds = merged

    return [Segment(start=g[0], end=g[-1], keyframe=_medoid(thumbs, g), frames=g)
            for g in bounds]


def _medoid(thumbs: list[np.ndarray], idxs: list[int]) -> int:
    """Index of the most central frame in a group.

    The frame nearest the group's mean signature. The exact L1 medoid needs
    every pairwise distance, which is quadratic in the group size and becomes
    the dominant cost on long static runs; the distance-to-mean choice is
    linear and picks the same frame on all but the most irregular groups.
    """
    if len(idxs) <= 2:
        return idxs[0]
    sub = np.stack([thumbs[i].ravel() for i in idxs])
    d = np.abs(sub - sub.mean(axis=0)).mean(axis=1)
    return idxs[int(d.argmin())]


def sweep(thumbs: list[np.ndarray], taus) -> list[dict]:
    """Segment count and compression at a range of thresholds.

    Used to choose an operating point from the data instead of assuming one:
    the useful tau is where near-duplicates have collapsed but genuinely
    different arrangements have not yet been merged.
    """
    out = []
    n = len(thumbs)
    for tau in taus:
        segs = segment_sequence(thumbs, tau)
        lens = [len(s) for s in segs]
        out.append({
            "tau": float(tau),
            "segments": len(segs),
            "compression": n / len(segs) if segs else float("nan"),
            "median_len": float(np.median(lens)) if lens else 0.0,
            "max_len": int(max(lens)) if lens else 0,
        })
    return out


def boundary_agreement(segs: list[Segment], truth: list[int],
                       tol: int = 3) -> dict:
    """Score detected boundaries against known ones (precision/recall).

    `truth` holds the first frame index of each true group. A detected
    boundary counts as a hit if some true boundary lies within `tol` frames.
    """
    det = [s.start for s in segs if s.start > 0]
    ref = [t for t in truth if t > 0]
    hit = sum(1 for d in det if any(abs(d - t) <= tol for t in ref))
    found = sum(1 for t in ref if any(abs(d - t) <= tol for d in det))
    prec = hit / len(det) if det else 0.0
    rec = found / len(ref) if ref else 0.0
    return {
        "detected": len(det),
        "true": len(ref),
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
    }
