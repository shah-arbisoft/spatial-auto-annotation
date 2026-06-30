"""RQ1 — fidelity of auto-labels vs. the ~900 human labels.

Produces, per predicate: precision, recall, F1; a confusion analysis; and box
agreement by IoU. Also fits and reports the `near` threshold. Baselines:
random-predicate, majority-class ("always on"), box-only-without-depth.

Scaffold for Week 3; the metric functions are real so the contract is testable
now, the data-loading is filled in once the human label files arrive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.predicates import PREDICATES, fit_near_threshold  # noqa: F401  (re-exported)


@dataclass
class PRF1:
    precision: float
    recall: float
    f1: float
    support: int


def prf1(pred: np.ndarray, gold: np.ndarray) -> PRF1:
    """Binary precision/recall/F1 for one predicate (1 = present)."""
    tp = int(((pred == 1) & (gold == 1)).sum())
    fp = int(((pred == 1) & (gold == 0)).sum())
    fn = int(((pred == 0) & (gold == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return PRF1(p, r, f, int((gold == 1).sum()))


def per_predicate_report(pred_mat: np.ndarray, gold_mat: np.ndarray) -> dict[str, PRF1]:
    """pred_mat/gold_mat: (N_pairs, 7) binary matrices aligned to PREDICATES."""
    return {name: prf1(pred_mat[:, i], gold_mat[:, i]) for i, name in enumerate(PREDICATES)}


def box_iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# --- baselines (Week 3) ---
def baseline_random(n_pairs: int, seed: int = 42) -> np.ndarray:
    """One random predicate per pair, as a (n_pairs, 7) one-hot matrix."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(PREDICATES), size=n_pairs)
    m = np.zeros((n_pairs, len(PREDICATES)), dtype=int)
    m[np.arange(n_pairs), idx] = 1
    return m


def baseline_majority(n_pairs: int, predicate: str = "on") -> np.ndarray:
    m = np.zeros((n_pairs, len(PREDICATES)), dtype=int)
    m[:, PREDICATES.index(predicate)] = 1
    return m


def main():  # pragma: no cover
    raise NotImplementedError("load human labels + auto labels, then report — Week 3")


if __name__ == "__main__":  # pragma: no cover
    main()
