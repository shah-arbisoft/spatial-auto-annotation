"""Randomised invariant checks over the predicate rules.

Fuzzes random scenes and asserts the structural guarantees the spec promises:
mutually exclusive families never co-occur, inverse predicates mirror between
the two orderings of a pair, near is symmetric, and near never co-occurs with a
contact relation. These hold BY CONSTRUCTION (strict inequalities and the
contact exclusion), and this test pins that down against future rule edits.
"""

import random

import numpy as np

from src.predicates import Obj, Thresholds, evaluate_pair

T = Thresholds()


def rand_obj(idx, rng):
    x1 = rng.uniform(0, 0.9)
    y1 = rng.uniform(0, 0.9)
    x2 = x1 + rng.uniform(0.02, 1.0 - x1)
    y2 = y1 + rng.uniform(0.02, 1.0 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    d = rng.uniform(0, 1)
    return Obj(idx, "obj", (x1, y1, x2, y2), cx, cy, d,
               np.array([cx, cy, d]))


def test_invariants_random_scenes():
    rng = random.Random(42)
    for _ in range(2000):
        a, b = rand_obj(0, rng), rand_obj(1, rng)
        ab = evaluate_pair(a, b, T)
        ba = evaluate_pair(b, a, T)
        pa, pb = set(ab.predicates), set(ba.predicates)

        # mutually exclusive families never co-occur on one ordered pair
        assert not ({"on", "under"} <= pa)
        assert not ({"to the left of", "to the right of"} <= pa)
        assert not ({"in front of", "behind"} <= pa)

        # inverses mirror across the pair orderings
        assert ("on" in pa) == ("under" in pb)
        assert ("to the left of" in pa) == ("to the right of" in pb)
        assert ("in front of" in pa) == ("behind" in pb)

        # near is symmetric and never co-occurs with contact
        assert ("near" in pa) == ("near" in pb)
        if "near" in pa:
            assert not ({"on", "under"} & pa)
