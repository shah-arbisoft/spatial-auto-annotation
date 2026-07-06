"""Pairwise mask-contact features: does A physically rest on B?

The support signature in image space: the pixels immediately BELOW the bottom
boundary of A's mask belong to B's mask. Unlike the box-adjacency test, this
also captures the containment case (a small object on a large one whose boxes
nest at a shallow viewing angle) and rejects cluster neighbours whose boxes
touch but whose masks do not.

`contact_below(A, B)` = fraction of A's mask columns whose bottom boundary has
a B-mask pixel within `gap_px` rows below. 1.0 = A sits fully on B; 0.0 = no
support contact.

Computed once per image during the perception pass and cached alongside the
geometry, so rule calibration stays an offline, seconds-scale operation.
"""

from __future__ import annotations

import numpy as np


def contact_below(mask_a: np.ndarray, mask_b: np.ndarray, gap_px: int = 5) -> float:
    """Fraction of A's bottom-boundary columns with B directly below."""
    H, _W = mask_a.shape
    has_col = mask_a.any(axis=0)
    cols = np.where(has_col)[0]
    if cols.size == 0:
        return 0.0
    # bottom-most row of A per column (valid only where has_col)
    bottom = (H - 1) - np.argmax(mask_a[::-1, :], axis=0)
    ys = bottom[cols]
    hit = np.zeros(cols.size, dtype=bool)
    for off in range(1, gap_px + 1):
        rows = np.clip(ys + off, 0, H - 1)
        hit |= mask_b[rows, cols]
    return float(hit.mean())


def pair_contacts(masks: list[np.ndarray], gap_px: int = 5,
                  min_val: float = 0.05) -> dict[tuple[int, int], float]:
    """contact_below for every ordered pair of masks; small values dropped.

    Returns {(i, j): fraction} meaning "mask i rests on mask j to this degree".
    """
    out: dict[tuple[int, int], float] = {}
    n = len(masks)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = contact_below(masks[i], masks[j], gap_px)
            if v >= min_val:
                out[(i, j)] = round(v, 4)
    return out
