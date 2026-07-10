"""Unit tests for the predicate rules — the graded core.

These run with no models or GPU (pure geometry), so the rules are verifiable
today. They encode the worked examples from docs/predicate_spec.md.
"""

import numpy as np
import pytest

from src.predicates import (
    Obj, Thresholds, evaluate_pair, fit_near_threshold,
    is_on, is_under, is_left_of, is_right_of, is_in_front_of, is_behind, is_near,
)

T = Thresholds()


def make(idx, label, box, depth):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return Obj(idx=idx, label=label, box=box, cx=cx, cy=cy, depth=depth,
               pos3d=np.array([cx, cy, depth], dtype=float))


def test_on_and_under_are_inverse():
    # A small box resting directly on top of a wide box, touching, overlapping.
    cup = make(0, "cup", (0.40, 0.30, 0.60, 0.50), depth=0.5)
    box = make(1, "box", (0.30, 0.50, 0.70, 0.80), depth=0.5)
    assert is_on(cup, box, T)
    assert is_under(box, cup, T)
    assert not is_on(box, cup, T)


def test_no_on_without_horizontal_overlap():
    a = make(0, "a", (0.05, 0.30, 0.15, 0.50), depth=0.5)  # far left
    b = make(1, "b", (0.70, 0.50, 0.90, 0.80), depth=0.5)  # far right
    assert not is_on(a, b, T)


def test_left_right_mirror():
    left = make(0, "l", (0.10, 0.40, 0.20, 0.60), depth=0.5)
    right = make(1, "r", (0.70, 0.40, 0.80, 0.60), depth=0.5)
    assert is_left_of(left, right, T)
    assert is_right_of(right, left, T)
    assert not is_right_of(left, right, T)


def test_front_behind_depth():
    near = make(0, "near", (0.40, 0.40, 0.50, 0.50), depth=0.10)  # smaller = nearer
    far = make(1, "far", (0.50, 0.40, 0.60, 0.50), depth=0.90)
    assert is_in_front_of(near, far, T)
    assert is_behind(far, near, T)


def test_lateral_ambiguous_flag():
    a = make(0, "a", (0.49, 0.40, 0.51, 0.60), depth=0.5)
    b = make(1, "b", (0.49, 0.40, 0.51, 0.60), depth=0.5)  # centres coincide
    res = evaluate_pair(a, b, T)
    assert "lateral_ambiguous" in res.flags
    assert "to the left of" not in res.predicates and "to the right of" not in res.predicates


def test_near_uses_relative_gap_and_contact_exclusion():
    # two books side by side, small gap relative to size -> near (both directions)
    a = make(0, "book", (0.30, 0.40, 0.45, 0.60), depth=0.5)
    b = make(1, "book", (0.50, 0.40, 0.65, 0.60), depth=0.5)
    res = evaluate_pair(a, b, T)
    assert "near" in res.predicates

    # cup resting on a box: contact pair -> on, and near is suppressed
    cup = make(0, "cube", (0.40, 0.30, 0.60, 0.50), depth=0.5)
    box = make(1, "box", (0.30, 0.50, 0.70, 0.80), depth=0.5)
    res_on = evaluate_pair(cup, box, T)
    assert "on" in res_on.predicates and "near" not in res_on.predicates
    res_under = evaluate_pair(box, cup, T)
    assert "under" in res_under.predicates and "near" not in res_under.predicates


def test_on_requires_depth_colocation():
    """A behind-pair that mimics a stack in 2D must not fire `on`.

    On a floor plane, farther projects higher: the boxes touch vertically and
    overlap horizontally, but the depths differ - the depth gate rejects it."""
    front = make(0, "remote", (0.40, 0.55, 0.60, 0.65), depth=0.30)
    back = make(1, "cube", (0.42, 0.45, 0.58, 0.56), depth=0.60)
    assert not is_on(back, front, T)          # depth gap 0.30 > on_depth_eps
    stacked_top = make(0, "cube", (0.42, 0.45, 0.58, 0.56), depth=0.31)
    assert is_on(stacked_top, front, T)       # same geometry, co-located depth


def test_plane_fallback_orders_floor_objects_in_depth_band():
    """Depth-ambiguous floor pair: lower box bottom = nearer wins the fallback.

    Depths are inside depth_eps so the depth rule abstains; both objects have
    contact evidence saying they rest on nothing -> the bottom edges decide."""
    front = make(0, "bottle", (0.30, 0.40, 0.40, 0.80), depth=0.50)
    back = make(1, "book", (0.55, 0.35, 0.70, 0.60), depth=0.51)  # |dz| < 0.03
    res = evaluate_pair(front, back, T, contact_ab=0.0, contact_ba=0.0,
                        elevated_a=False, elevated_b=False)
    assert "in front of" in res.predicates       # bottom 0.80 vs 0.60
    assert "depth_ambiguous" not in res.flags
    rev = evaluate_pair(back, front, T, contact_ab=0.0, contact_ba=0.0,
                        elevated_a=False, elevated_b=False)
    assert "behind" in rev.predicates


def test_plane_fallback_blocked_by_elevation_and_masklessness():
    """An elevated object (rests on another) or a maskless scene keeps the flag."""
    a = make(0, "cube", (0.30, 0.40, 0.40, 0.60), depth=0.50)
    b = make(1, "book", (0.55, 0.35, 0.70, 0.80), depth=0.51)
    # a rests on something -> ground-plane reasoning invalid for the pair
    res = evaluate_pair(a, b, T, contact_ab=0.0, contact_ba=0.0,
                        elevated_a=True, elevated_b=False)
    assert not {"in front of", "behind"} & set(res.predicates)
    assert "depth_ambiguous" in res.flags
    # no mask evidence at all (box-only mode) -> fallback off, behaviour as before
    res2 = evaluate_pair(a, b, T)
    assert not {"in front of", "behind"} & set(res2.predicates)
    assert "depth_ambiguous" in res2.flags


def test_plane_fallback_abstains_inside_band():
    """Bottom edges closer than plane_band: still ambiguous, still flagged."""
    a = make(0, "book", (0.30, 0.40, 0.45, 0.600), depth=0.50)
    b = make(1, "book", (0.55, 0.40, 0.70, 0.602), depth=0.51)  # |dbottom| < 0.005
    res = evaluate_pair(a, b, T, contact_ab=0.0, contact_ba=0.0,
                        elevated_a=False, elevated_b=False)
    assert not {"in front of", "behind"} & set(res.predicates)
    assert "depth_ambiguous" in res.flags


def test_evaluate_scene_derives_elevation_from_contact_map():
    """The scene evaluator must gate the fallback with its own contact map:
    a cube stacked on a box is elevated, so its depth-ambiguous pair with a
    bystander stays flagged, while two floor objects get plane-ordered."""
    from src.predicates import evaluate_scene
    cube = make(0, "cube", (0.40, 0.30, 0.60, 0.50), depth=0.50)   # on the box
    box = make(1, "box", (0.30, 0.50, 0.70, 0.80), depth=0.50)
    bottle = make(2, "bottle", (0.05, 0.20, 0.15, 0.60), depth=0.505)
    contact = {(0, 1): 0.9}                     # cube rests on box
    res = {(r.subject, r.object): r for r in evaluate_scene([cube, box, bottle], T,
                                                            contact=contact)}
    # cube (elevated) vs bottle: fallback blocked despite different bottoms
    assert "depth_ambiguous" in res[(0, 2)].flags
    # box vs bottle: both floor-standing, bottoms 0.80 vs 0.60 -> box in front
    assert "in front of" in res[(1, 2)].predicates


def test_near_threshold_fit_recovers_separating_value():
    # distances clearly separable at ~0.5; humans call <0.5 "near".
    distances = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    human_near = np.array([1, 1, 1, 0, 0, 0])
    best_T, best_f1 = fit_near_threshold(distances, human_near)
    assert best_f1 == pytest.approx(1.0)
    assert 0.3 <= best_T < 0.7
