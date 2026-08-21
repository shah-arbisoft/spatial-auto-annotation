"""A10 - can the class guard be replaced by geometry?

The shipped support rule refuses `on`/`under` for any pair involving a person
(`no_support_classes`). The reason is real: a held object satisfies mask
contact without resting on anything. The objection to it is also real, that a
class list is a semantic patch on a geometric rule and says nothing about a
manipulator, a trolley or a dog.

There is a geometric signature available. An object *resting* on something
meets it at the supporter's top edge; an object *held* meets it partway down
the holder's body. So measure, for every contacting pair, where the subject's
bottom edge falls within the object's vertical extent:

    drop = (a.bottom - b.top) / (b.bottom - b.top)

drop near 0 means A sits on B's top surface. A held object should sit well
inside B. If gold support pairs and the blocked human pairs separate on this
number, the class list is replaceable by a threshold that names no class.

This is an ablation, not a shipped change: the audited and benchmarked labels
were produced with the class guard, and swapping the rule now would invalidate
them. It reports whether the guard *could* be replaced and at what cost.

    python eval/support_guard_ablation.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import SpatialDataset          # noqa: E402
from src.pipeline import load_config            # noqa: E402

OUT = ROOT / "outputs" / "support_guard_ablation.json"


def drop_fraction(a: dict, b: dict) -> float | None:
    """Where A's bottom edge falls inside B's vertical extent (0 = B's top)."""
    _, _, _, a_bot = a["box"]
    _, b_top, _, b_bot = b["box"]
    h = b_bot - b_top
    if h <= 0:
        return None
    return (a_bot - b_top) / h


def main() -> int:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    thr = cfg["predicates"]
    contact_min = float(thr["on_contact_min"])
    depth_eps = float(thr.get("on_depth_eps", thr.get("depth_eps", 0.03)))

    ds = SpatialDataset(cfg["dataset"]["root"])
    gold = {}
    for gt in ds:
        for r in gt.relations:
            gold.setdefault((gt.image_id, r.subject, r.object), set()).add(r.predicate)

    resting, held = [], []          # drop fractions
    n_img = 0
    for gdir in sorted((ROOT / "outputs" / "geometry").iterdir()):
        if not gdir.is_dir():
            continue
        for gf in sorted(gdir.glob("*.json")):
            if gf.name.endswith(".contact.json"):
                continue
            cf = gf.with_suffix(".contact.json")
            if not cf.exists():
                continue
            objs = json.loads(gf.read_text())
            contact = json.loads(cf.read_text())
            image_id = f"{gdir.name}/{gf.stem}"
            n_img += 1
            by_idx = {o["idx"]: o for o in objs}
            for key, frac in contact.items():
                i, j = (int(x) for x in key.split("-"))
                a, b = by_idx.get(i), by_idx.get(j)
                if a is None or b is None or frac < contact_min:
                    continue
                if not (a["cy"] < b["cy"]):
                    continue
                if abs(a.get("depth", 0) - b.get("depth", 0)) > depth_eps:
                    continue
                d = drop_fraction(a, b)
                if d is None:
                    continue
                if b["label"] == "human" or a["label"] == "human":
                    held.append(d)
                elif "on" in gold.get((image_id, i, j), set()):
                    resting.append(d)

    out = {"n_images": n_img, "contact_min": contact_min,
           "n_resting_gold": len(resting), "n_human_pairs": len(held)}
    print(f"images with cached geometry: {n_img}")
    print(f"gold-confirmed resting pairs above the contact threshold: {len(resting)}")
    print(f"pairs the class guard blocks (a person on either side):    {len(held)}")

    def describe(name, xs):
        if not xs:
            print(f"  {name}: none")
            return None
        xs = sorted(xs)
        q = {"min": xs[0], "p10": xs[int(.10 * (len(xs) - 1))],
             "median": st.median(xs), "p90": xs[int(.90 * (len(xs) - 1))],
             "max": xs[-1]}
        print(f"  {name:26} min {q['min']:+.2f}  p10 {q['p10']:+.2f}  "
              f"median {q['median']:+.2f}  p90 {q['p90']:+.2f}  max {q['max']:+.2f}")
        return q

    print("\ndrop fraction (0 = subject's bottom at object's top edge):")
    out["resting"] = describe("resting (gold `on`)", resting)
    out["held"] = describe("blocked (person involved)", held)

    if resting and held:
        # A class-free guard would keep pairs below some drop threshold.
        print("\nwhat a class-free threshold would cost and catch:")
        rows = []
        for tau in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
            keep = sum(1 for d in resting if d <= tau) / len(resting)
            block = sum(1 for d in held if d > tau) / len(held)
            rows.append({"tau": tau, "resting_kept": keep, "human_pairs_blocked": block})
            print(f"  drop <= {tau:.2f}: keeps {keep:6.1%} of gold resting pairs, "
                  f"blocks {block:6.1%} of the person pairs")
        out["sweep"] = rows

    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nreport -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
