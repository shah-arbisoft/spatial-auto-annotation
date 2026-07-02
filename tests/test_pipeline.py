"""Offline test of the pipeline wiring (geometry -> predicates), no GPU needed.

Feeds synthetic depth + rectangular masks so objects_from() and
annotate_objects() are exercised end to end without SAM2 or a depth model.
"""

import numpy as np

from src.pipeline import annotate_objects, objects_from, rect_mask

CFG = {
    "predicates": {
        "near_T": 0.30, "on_vertical_gap": 0.05, "on_horizontal_overlap": 0.20,
        "lateral_center_eps": 0.02, "depth_eps": 0.03,
    },
    "confidence": {"flag_near_band": 0.05},
    "correction": {"enabled": True},
    "geometry": {"z_scale": 1.0},
}


def _scene():
    """Two objects: A left+near, B right+far, in a 100x100 image."""
    W = H = 100
    depth = np.full((H, W), 0.8)     # right/background far
    depth[:, :50] = 0.2              # left half nearer
    boxes = [(10, 40, 30, 60), (70, 40, 90, 60)]   # A left, B right, no x-overlap
    masks = [rect_mask(b, H, W) for b in boxes]
    objs = objects_from(boxes, ["book", "bottle"], masks, depth, W, H, z_scale=1.0)
    return objs


def test_pipeline_lateral_and_depth():
    objs = _scene()
    pairs = {(p.subject, p.object): p for p in annotate_objects(objs, CFG)}

    ab = pairs[(0, 1)].predicates
    ba = pairs[(1, 0)].predicates
    assert "to the left of" in ab and "in front of" in ab
    assert "to the right of" in ba and "behind" in ba
    # far apart in 3D -> not near; no vertical overlap -> no on/under
    assert "near" not in ab
    assert "on" not in ab and "under" not in ab


def test_pipeline_geometry_values():
    objs = _scene()
    a, b = objs
    assert a.cx < b.cx                    # A is left of B
    assert a.depth < b.depth              # A is nearer
    assert a.pos3d.shape == (3,)


def test_lift_empty_mask_falls_back_to_box():
    """A failed segmentation must not teleport the object to the image centre."""
    W = H = 100
    depth = np.full((H, W), 0.9)
    depth[40:60, 10:30] = 0.1             # the object's true region is near
    empty = np.zeros((H, W), dtype=bool)
    objs = objects_from([(10, 40, 30, 60)], ["book"], [empty], depth, W, H)
    o = objs[0]
    assert abs(o.cx - 0.20) < 0.02        # box centroid, not 0.5
    assert abs(o.cy - 0.50) < 0.02
    assert o.depth < 0.2                  # depth sampled from the box region
