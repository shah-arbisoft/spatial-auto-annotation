"""Compute the seven spatial predicates from geometry.

This module is the graded core of the project. Each predicate is a small,
readable function with explicit thresholds, implementing docs/predicate_spec.md.
No learned model decides a label here — every relation is a deterministic rule
over measured geometry (mask, box, image position, per-object depth).

Predicates (ordered pair A=subject, B=object):
    on, under, left of, right of, in front of, behind, near.

Image convention: x increases right, y increases down. Depth: smaller = nearer
the camera. Distances/gaps are normalised by image size so one threshold
transfers across resolutions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Canonical predicate names — EXACTLY the dataset's relationship strings, so our
# labels map one-to-one onto the dataset's predicate IDs (see PREDICATE_IDS).
# The dataset's tool ships a 19-predicate Visual Genome list; these seven are the
# spatial subset we compute and target (95.9% of all human triplets).
PREDICATES = (
    "on", "under", "to the left of", "to the right of",
    "in front of", "behind", "near",
)

# Mapping from our predicate names to the dataset's relationship IDs
# (from annotated_data/*/relationships.json, identical across all groups).
PREDICATE_IDS = {
    "on": 10,
    "under": 17,
    "to the left of": 15,
    "to the right of": 16,
    "in front of": 6,
    "behind": 2,
    "near": 9,
}


@dataclass
class Obj:
    """One detected object with everything the rules need.

    All spatial fields are NORMALISED to [0, 1] by image width/height so the
    thresholds in the config are resolution-independent.
    """

    idx: int
    label: str
    # normalised box (x1, y1, x2, y2)
    box: tuple[float, float, float, float]
    # normalised mask centroid
    cx: float
    cy: float
    # per-object relative depth (smaller = nearer camera), arbitrary units
    depth: float
    # lifted 3D position (X, Y, Z); X,Y normalised image coords, Z scaled depth
    pos3d: Optional[np.ndarray] = None


@dataclass
class Thresholds:
    """Explicit, justified thresholds — mirrors configs/default.yaml."""

    near_T: float = 1.372
    on_vertical_gap: float = 0.05
    on_horizontal_overlap: float = 0.20
    on_depth_eps: float = 0.06     # support requires depth co-location (see is_on)
    lateral_center_eps: float = 0.02
    depth_eps: float = 0.03
    flag_near_band: float = 0.15


@dataclass
class PairResult:
    """The predicates emitted for one ordered pair, plus any ambiguity flags."""

    subject: int
    object: int
    predicates: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _x_extent_overlap(a: Obj, b: Obj) -> float:
    """Fraction of the narrower box's x-extent that overlaps the other's.

    Returns a value in [0, 1]; 0 means the boxes share no horizontal range.
    """
    ax1, _, ax2, _ = a.box
    bx1, _, bx2, _ = b.box
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    narrower = max(1e-9, min(ax2 - ax1, bx2 - bx1))
    return inter / narrower


def _vertical_gap(top: Obj, bottom: Obj) -> float:
    """Signed normalised gap between the bottom of `top` and the top of `bottom`.

    Positive when there is clear space between them; small/zero means touching;
    negative means they overlap vertically.
    """
    top_bottom_edge = top.box[3]      # y2 of the upper object
    bottom_top_edge = bottom.box[1]   # y1 of the lower object
    return bottom_top_edge - top_bottom_edge


def dist3d(a: Obj, b: Obj) -> float:
    """Euclidean distance between two lifted 3D centroids."""
    if a.pos3d is None or b.pos3d is None:
        raise ValueError("pos3d required; run geometry.lift first")
    return float(np.linalg.norm(a.pos3d - b.pos3d))


def box_gap_rel(a: Obj, b: Obj) -> float:
    """Size-relative 2D gap between two boxes — the `near` metric.

    Edge-to-edge gap between the (normalised) boxes, divided by the mean of the
    two objects' sizes (sqrt of box area), so "near" scales with the objects:
    a small gap between two books is near, the same absolute gap between a
    person and a cube may not be. 0 when the boxes touch or overlap.

    Chosen empirically over 3D-centroid distance: per-image relative depth made
    centroid distances incomparable across scenes, and human `near` labels are
    reproduced far better by this metric (see docs/DATASET_NOTES.md).
    """
    ax1, ay1, ax2, ay2 = a.box
    bx1, by1, bx2, by2 = b.box
    gap_x = max(0.0, ax1 - bx2, bx1 - ax2)
    gap_y = max(0.0, ay1 - by2, by1 - ay2)
    gap = float(np.hypot(gap_x, gap_y))
    size_a = float(np.sqrt(max(1e-9, (ax2 - ax1) * (ay2 - ay1))))
    size_b = float(np.sqrt(max(1e-9, (bx2 - bx1) * (by2 - by1))))
    return gap / ((size_a + size_b) / 2)


# --------------------------------------------------------------------------- #
# The seven predicate tests (boolean cores; flags handled in evaluate_pair)
# --------------------------------------------------------------------------- #
def is_on(a: Obj, b: Obj, t: Thresholds) -> bool:
    """A rests on B: above, touching, horizontally overlapping, AND co-located
    in depth.

    The depth gate (|depth_A - depth_B| <= on_depth_eps) exists because, on a
    floor plane, "farther away" projects as "higher in the image": an object
    BEHIND another produces the same 2D box signature as one stacked ON it.
    Truly stacked objects share a camera distance; behind-pairs do not.
    Measured on the manual audit sample, the gate removes ~half the false
    support labels at <1 point of recall (calibrated on train groups).
    """
    above = a.cy < b.cy
    gap = _vertical_gap(top=a, bottom=b)
    touching = -t.on_vertical_gap <= gap <= t.on_vertical_gap
    overlap = _x_extent_overlap(a, b) >= t.on_horizontal_overlap
    co_depth = abs(a.depth - b.depth) <= t.on_depth_eps
    return above and touching and overlap and co_depth


def is_under(a: Obj, b: Obj, t: Thresholds) -> bool:
    """A is under B == B is on A (strict inverse)."""
    return is_on(b, a, t)


def is_left_of(a: Obj, b: Obj, t: Thresholds) -> bool:
    """A's horizontal centre is clearly left of B's (camera frame)."""
    return (b.cx - a.cx) > t.lateral_center_eps


def is_right_of(a: Obj, b: Obj, t: Thresholds) -> bool:
    """A's horizontal centre is clearly right of B's (camera frame)."""
    return (a.cx - b.cx) > t.lateral_center_eps


def is_in_front_of(a: Obj, b: Obj, t: Thresholds) -> bool:
    """A is clearly nearer the camera than B (smaller depth)."""
    return (b.depth - a.depth) > t.depth_eps


def is_behind(a: Obj, b: Obj, t: Thresholds) -> bool:
    """A is clearly farther from the camera than B."""
    return (a.depth - b.depth) > t.depth_eps


def is_near(a: Obj, b: Obj, t: Thresholds) -> bool:
    """A and B are within the fitted size-relative gap threshold.

    Contact exclusion (near never co-occurs with on/under in the human labels —
    0 of 469 pairs) is applied in evaluate_pair, which sees the on/under result.
    """
    return box_gap_rel(a, b) <= t.near_T


# --------------------------------------------------------------------------- #
# Correction + confidence: evaluate a single ordered pair
# --------------------------------------------------------------------------- #
def evaluate_pair(a: Obj, b: Obj, t: Thresholds, correct: bool = True) -> PairResult:
    """Compute every predicate for ordered pair (A, B), correct contradictions,
    and attach ambiguity flags. See docs/predicate_spec.md §8–9."""
    res = PairResult(subject=a.idx, object=b.idx)

    # --- vertical: on / under (mutually exclusive by construction) ---
    on = is_on(a, b, t)
    under = is_under(a, b, t)
    if correct and on and under:
        # Geometrically impossible; demote to a flag rather than emit both.
        res.flags.append("on_under_conflict")
        on = under = False
    if on:
        res.predicates.append("on")
    if under:
        res.predicates.append("under")

    # --- lateral: left / right, with ambiguity band ---
    if is_left_of(a, b, t):
        res.predicates.append("to the left of")
    elif is_right_of(a, b, t):
        res.predicates.append("to the right of")
    else:
        res.flags.append("lateral_ambiguous")  # centres nearly coincide

    # --- depth: in front of / behind, with ambiguity band ---
    if is_in_front_of(a, b, t):
        res.predicates.append("in front of")
    elif is_behind(a, b, t):
        res.predicates.append("behind")
    else:
        res.flags.append("depth_ambiguous")  # depths nearly equal

    # --- near: size-relative box gap, suppressed for contact pairs ---
    # Measured on the human labels: near co-occurs with on/under on 0 of 469
    # pairs — annotators used `near` as "close but no contact relation", so a
    # pair already labelled on/under is not additionally near.
    contact = "on" in res.predicates or "under" in res.predicates
    gap = box_gap_rel(a, b)
    if not contact:
        if gap <= t.near_T:
            res.predicates.append("near")
        if abs(gap - t.near_T) <= t.flag_near_band:
            res.flags.append("near_threshold_edge")

    return res


def evaluate_scene(objs: list[Obj], t: Thresholds, correct: bool = True) -> list[PairResult]:
    """Compute predicates for every ordered pair of distinct objects in a scene."""
    results: list[PairResult] = []
    for a in objs:
        for b in objs:
            if a.idx == b.idx:
                continue
            results.append(evaluate_pair(a, b, t, correct=correct))
    return results


# --------------------------------------------------------------------------- #
# near-threshold fitting (called from eval/fidelity.py; lives here so the rule
# and its calibration stay together)
# --------------------------------------------------------------------------- #
def fit_near_threshold(
    distances: np.ndarray,
    human_near: np.ndarray,
    candidates: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """Fit the `near` threshold to human labels.

    Args:
        distances:   3D distances for each labelled pair, shape (N,).
        human_near:  1 if humans labelled the pair `near`, else 0, shape (N,).
        candidates:  threshold values to sweep; default 100 points over the
                     observed distance range.

    Returns:
        (best_T, best_f1) — the threshold maximising F1 against the human labels.
    """
    distances = np.asarray(distances, dtype=float)
    human_near = np.asarray(human_near, dtype=int)
    if candidates is None:
        candidates = np.linspace(distances.min(), distances.max(), 100)

    best_T, best_f1 = float(candidates[0]), -1.0
    for thr in candidates:
        pred = (distances <= thr).astype(int)
        tp = int(((pred == 1) & (human_near == 1)).sum())
        fp = int(((pred == 1) & (human_near == 0)).sum())
        fn = int(((pred == 0) & (human_near == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if f1 > best_f1:
            best_T, best_f1 = float(thr), f1
    return best_T, best_f1
