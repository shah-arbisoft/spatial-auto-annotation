"""Ablations (Week 4): isolate each design choice's contribution to fidelity.

Planned ablations (each re-runs the fidelity report under one change):
  - depth on vs. off        (front/behind/near degrade without depth)
  - near_T sweep            (report F1 vs. threshold curve; mark fitted T)
  - correction step on/off  (does rejecting impossible labels help?)
  - detector swap           (YOLOv10m vs. GroundingDINO)

Also builds the per-predicate failure gallery with causes labelled
(depth error, threshold edge case, occlusion, detector miss).

Scaffold only; implemented in Week 4 against the Week 3 fidelity harness.
"""

from __future__ import annotations

import numpy as np

from src.predicates import PREDICATES, Thresholds


def near_threshold_sweep(distances: np.ndarray, human_near: np.ndarray,
                         candidates: np.ndarray) -> list[tuple[float, float]]:
    """Return [(T, f1), ...] for plotting the near-threshold sweep curve."""
    out = []
    for thr in candidates:
        pred = (distances <= thr).astype(int)
        tp = int(((pred == 1) & (human_near == 1)).sum())
        fp = int(((pred == 1) & (human_near == 0)).sum())
        fn = int(((pred == 0) & (human_near == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        out.append((float(thr), 2 * p * r / (p + r) if (p + r) else 0.0))
    return out


def main():  # pragma: no cover
    raise NotImplementedError("ablation runs — Week 4")


if __name__ == "__main__":  # pragma: no cover
    main()
