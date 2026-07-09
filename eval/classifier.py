"""RQ2 — features and model for the controlled relation classifier.

Deliberately lightweight: geometric pair features plus a small MLP, so the
human-vs-auto comparison isolates the LABEL SOURCE rather than a complex
model's own biases. Features are pure measured geometry — identical whichever
label source trains the model — and come entirely from the cached
geometry/contact maps, so the experiment is offline and exactly reproducible.
"""

from __future__ import annotations

import math

import numpy as np

from src.predicates import Obj, box_gap_rel, _vertical_gap, _x_extent_overlap

FEATURE_NAMES = [
    "dx", "dy", "ddepth", "abs_ddepth",
    "aw", "ah", "bw", "bh", "area_a", "area_b",
    "x_overlap", "vgap_ab", "vgap_ba", "gap_rel",
    "contact_ab", "contact_ba", "dist3d",
]


def pair_features(a: Obj, b: Obj, contact_ab: float, contact_ba: float) -> np.ndarray:
    """Geometric features for ordered pair (A, B). No labels involved."""
    dx = b.cx - a.cx
    dy = b.cy - a.cy
    dd = b.depth - a.depth
    ax1, ay1, ax2, ay2 = a.box
    bx1, by1, bx2, by2 = b.box
    aw, ah = ax2 - ax1, ay2 - ay1
    bw, bh = bx2 - bx1, by2 - by1
    dist = float(np.linalg.norm(a.pos3d - b.pos3d)) if a.pos3d is not None else 0.0
    return np.array([
        dx, dy, dd, abs(dd),
        aw, ah, bw, bh, aw * ah, bw * bh,
        _x_extent_overlap(a, b),
        _vertical_gap(top=a, bottom=b),
        _vertical_gap(top=b, bottom=a),
        box_gap_rel(a, b),
        contact_ab, contact_ba, dist,
    ], dtype=float)


def make_mlp(seed: int = 42):
    """The small MLP used for every predicate and both label sources."""
    from sklearn.neural_network import MLPClassifier  # noqa: PLC0415

    return MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                         max_iter=60, random_state=seed, tol=1e-4)


def oversample_positives(X, y, cap_ratio: float = 0.10, seed: int = 42):
    """Repeat positive rows until positives are >= cap_ratio of negatives.

    sklearn's MLP has no class weights; seeded oversampling gives rare
    predicates (near is ~1% of pairs) a fighting chance, identically for both
    label sources.
    """
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(pos) >= cap_ratio * len(neg):
        return X, y
    need = math.ceil(cap_ratio * len(neg)) - len(pos)
    extra = rng.choice(pos, size=need, replace=True)
    idx = np.concatenate([np.arange(len(y)), extra])
    rng.shuffle(idx)
    return X[idx], y[idx]
