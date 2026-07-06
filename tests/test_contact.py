"""Mask-contact feature and its integration with the support rule."""

import numpy as np

from src.contact import contact_below, pair_contacts
from src.predicates import Obj, Thresholds, evaluate_pair

T = Thresholds()


def block(h, w, r0, r1, c0, c1):
    m = np.zeros((h, w), dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


def test_contact_below_stacked():
    a = block(100, 100, 30, 40, 30, 50)   # A directly above B, touching
    b = block(100, 100, 40, 80, 20, 60)
    assert contact_below(a, b) == 1.0
    assert contact_below(b, a) == 0.0     # B's bottom borders the floor, not A


def test_contact_below_captures_containment():
    """Small object on a big one whose BOX nests inside the support's box.

    The box vertical-gap test fails here (gap strongly negative), but the
    pixels below A's mask bottom belong to B — the contact test fires."""
    b = block(100, 100, 40, 80, 20, 60)
    b[45:55, 30:50] = False               # carve the hole where A sits
    a = block(100, 100, 45, 55, 30, 50)   # A entirely inside B's box extent
    assert contact_below(a, b) == 1.0


def test_contact_below_rejects_cluster_neighbour():
    a = block(100, 100, 50, 70, 10, 30)   # side by side, no vertical support
    b = block(100, 100, 50, 70, 32, 60)
    assert contact_below(a, b) == 0.0


def test_pair_contacts_and_rule_integration():
    b = block(100, 100, 40, 80, 20, 60)
    b[45:55, 30:50] = False
    a = block(100, 100, 45, 55, 30, 50)
    contacts = pair_contacts([a, b])
    assert contacts[(0, 1)] == 1.0 and (1, 0) not in contacts

    # containment pair: box test misses, contact evidence recovers it
    oa = Obj(0, "cube", (0.30, 0.45, 0.50, 0.55), 0.40, 0.50, 0.50,
             np.array([0.40, 0.50, 0.50]))
    ob = Obj(1, "box", (0.20, 0.40, 0.60, 0.80), 0.40, 0.60, 0.52,
             np.array([0.40, 0.60, 0.52]))
    box_path = evaluate_pair(oa, ob, T)
    assert "on" not in box_path.predicates            # nested boxes: gap negative
    with_contact = evaluate_pair(oa, ob, T, contact_ab=1.0, contact_ba=0.0)
    assert "on" in with_contact.predicates            # contact evidence fires
    # computed contact map present but zero -> cluster neighbour stays off
    zeroed = evaluate_pair(oa, ob, T, contact_ab=0.0, contact_ba=0.0)
    assert "on" not in zeroed.predicates
