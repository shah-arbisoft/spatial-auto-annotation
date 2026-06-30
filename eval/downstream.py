"""RQ2 driver — train the classifier once on human labels, once on auto labels,
and compare per-predicate recall and mean recall.

Controlled experiment: identical features, model, seed, and splits; the ONLY
thing that changes is the label source. Report the gap. Scaffold (Week 5).
"""

from __future__ import annotations


def run_comparison(seed: int = 42):  # pragma: no cover
    """Train on human labels, train on auto labels, return paired metrics."""
    raise NotImplementedError("downstream comparison — Week 5")


if __name__ == "__main__":  # pragma: no cover
    run_comparison()
