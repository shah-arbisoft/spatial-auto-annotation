"""Lift segmented objects to simple 3D positions using mask + depth.

Each object's 3D position is (X, Y, Z) where X, Y are the normalised image
centroid of its mask and Z is a representative depth sampled over the mask.
This is intentionally simple: a relative, per-image lift sufficient for the
ordinal/threshold predicate rules. It is NOT metric reconstruction — that
limitation is discussed in the dissertation's critical chapter.
"""

from __future__ import annotations

import numpy as np

from .predicates import Obj


def mask_centroid_norm(mask: np.ndarray, width: int, height: int) -> tuple[float, float]:
    """Normalised (cx, cy) of a boolean mask. Falls back to image centre if empty."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0.5, 0.5
    return float(xs.mean()) / width, float(ys.mean()) / height


def sample_depth(depth_map: np.ndarray, mask: np.ndarray, robust: bool = True) -> float:
    """Representative depth over a mask.

    Uses the median (robust to depth outliers at object edges) by default.
    """
    vals = depth_map[mask.astype(bool)]
    if vals.size == 0:
        return float(np.median(depth_map))
    return float(np.median(vals)) if robust else float(vals.mean())


def lift(
    idx: int,
    label: str,
    box_xyxy: tuple[float, float, float, float],
    mask: np.ndarray,
    depth_map: np.ndarray,
    width: int,
    height: int,
    z_scale: float = 1.0,
) -> Obj:
    """Build an `Obj` with normalised box/centroid and a lifted 3D position.

    Args:
        box_xyxy: pixel box (x1, y1, x2, y2) from the detector.
        mask:     boolean object mask, full image resolution.
        depth_map: relative depth map, full image resolution.
        z_scale:  scales depth into units comparable to normalised X, Y so the
                  3D distance used by `near` is balanced across axes. Tune
                  alongside near_T during fitting.
    """
    cx, cy = mask_centroid_norm(mask, width, height)
    d = sample_depth(depth_map, mask)

    x1, y1, x2, y2 = box_xyxy
    norm_box = (x1 / width, y1 / height, x2 / width, y2 / height)

    pos3d = np.array([cx, cy, d * z_scale], dtype=float)
    return Obj(idx=idx, label=label, box=norm_box, cx=cx, cy=cy, depth=d, pos3d=pos3d)
