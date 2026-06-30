"""RQ2 — a controlled lightweight relation classifier.

Pair features (relative position, depth difference, box geometry) + a small MLP,
with class weighting for rare predicates. Deliberately simple so the human-vs-auto
comparison isolates the LABEL SOURCE rather than a complex model's own biases.
A few hundred lines; trains on the 2060 in minutes.

Scaffold (Week 5). Feature extraction is real and shared by both training runs.
"""

from __future__ import annotations

import numpy as np

from src.predicates import Obj


def pair_features(a: Obj, b: Obj) -> np.ndarray:
    """Geometric features for ordered pair (A, B). Pure geometry, no labels.

    Kept aligned with the predicate rules so the classifier sees the same
    signal the rules use, plus a little extra box geometry.
    """
    dx = b.cx - a.cx                      # +ve => B right of A
    dy = b.cy - a.cy                      # +ve => B below A
    ddepth = b.depth - a.depth            # +ve => B farther than A
    ax1, ay1, ax2, ay2 = a.box
    bx1, by1, bx2, by2 = b.box
    aw, ah = ax2 - ax1, ay2 - ay1
    bw, bh = bx2 - bx1, by2 - by1
    dist = float(np.linalg.norm(a.pos3d - b.pos3d)) if a.pos3d is not None and b.pos3d is not None else 0.0
    return np.array([dx, dy, ddepth, aw, ah, bw, bh, aw * ah, bw * bh, dist], dtype=float)


def build_mlp(in_dim: int, n_classes: int):
    """Small MLP. Implemented Week 5 (sklearn MLPClassifier or a tiny torch net)."""
    raise NotImplementedError("MLP build — Week 5")
