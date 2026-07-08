"""RQ2 building blocks: pair features and the oversampling helper."""

import numpy as np

from eval.classifier import FEATURE_NAMES, oversample_positives, pair_features
from src.predicates import Obj


def obj(idx, box, depth):
    x1, y1, x2, y2 = box
    return Obj(idx, "obj", box, (x1 + x2) / 2, (y1 + y2) / 2, depth,
               np.array([(x1 + x2) / 2, (y1 + y2) / 2, depth]))


def test_pair_features_shape_and_values():
    a = obj(0, (0.10, 0.40, 0.30, 0.60), 0.20)
    b = obj(1, (0.50, 0.40, 0.70, 0.60), 0.50)
    f = pair_features(a, b, contact_ab=0.8, contact_ba=0.0)
    assert f.shape == (len(FEATURE_NAMES),)
    named = dict(zip(FEATURE_NAMES, f))
    assert named["dx"] == 0.4                       # b is to the right
    assert named["ddepth"] == 0.3                   # b is farther
    assert named["contact_ab"] == 0.8 and named["contact_ba"] == 0.0
    assert named["x_overlap"] == 0.0                # disjoint x-extents
    # features must be finite for any well-formed pair
    assert np.isfinite(f).all()


def test_pair_features_direction_antisymmetry():
    a = obj(0, (0.10, 0.40, 0.30, 0.60), 0.20)
    b = obj(1, (0.50, 0.40, 0.70, 0.60), 0.50)
    fab = dict(zip(FEATURE_NAMES, pair_features(a, b, 0.0, 0.0)))
    fba = dict(zip(FEATURE_NAMES, pair_features(b, a, 0.0, 0.0)))
    assert fab["dx"] == -fba["dx"]
    assert fab["ddepth"] == -fba["ddepth"]
    assert fab["gap_rel"] == fba["gap_rel"]         # symmetric metric


def test_oversample_positives_reaches_cap():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 3))
    y = np.zeros(1000, dtype=int)
    y[:5] = 1                                       # 0.5% positives
    Xo, yo = oversample_positives(X, y, cap_ratio=0.10, seed=42)
    assert yo.sum() >= 0.10 * (yo == 0).sum()
    assert (yo == 0).sum() == 995                   # negatives untouched
    # deterministic given the seed
    Xo2, yo2 = oversample_positives(X, y, cap_ratio=0.10, seed=42)
    assert np.array_equal(yo, yo2) and np.array_equal(Xo, Xo2)


def test_oversample_noop_when_balanced():
    X = np.zeros((100, 2))
    y = np.array([1] * 50 + [0] * 50)
    Xo, yo = oversample_positives(X, y, cap_ratio=0.10)
    assert len(yo) == 100
