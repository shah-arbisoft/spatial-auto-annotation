"""Check what the benchmark's zero-shot recall is actually measuring.

zR@K is defined in the SGG literature as recall on subject-predicate-object
types absent from the model's *own* training data (Tang et al., 2020). The
benchmark here scores every arm against one fixed reference, the human
training annotation, because that is the only way the arms' numbers can be
put in the same column. Those two things are not the same question, and this
script measures the gap.

The distinction matters because a label source that is denser than the human
annotation contains relation types the human annotation lacks. An arm trained
on it has therefore *seen* much of what the shared reference counts as
zero-shot, so its zR score reflects the coverage of its training labels
rather than compositional generalisation.

    python eval/zeroshot_reference_check.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def triplet_types(path: Path) -> set:
    """Distinct (subject class, predicate, object class) types in a split."""
    d = json.loads(path.read_text(encoding="utf-8"))
    cat = {a["id"]: a["category_id"] for a in d["annotations"]}
    return {(cat[r["subject_id"]], r["predicate_id"], cat[r["object_id"]])
            for r in d["rel_annotations"]
            if r["subject_id"] in cat and r["object_id"] in cat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/spatial_sgg")
    ap.add_argument("--arms", default="human,auto,vlm")
    ap.add_argument("--reference", default="human",
                    help="the arm whose training annotation defines the "
                         "shared zero-shot set")
    args = ap.parse_args()

    root = Path(args.root)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    train = {a: triplet_types(root / f"train/_annotations.{a}.coco.json")
             for a in arms}
    test = triplet_types(root / "test/_annotations.human.coco.json")
    shared_zs = test - train[args.reference]

    print(f"test triplet types: {len(test)}")
    print(f"shared zero-shot set (test minus {args.reference} train): "
          f"{len(shared_zs)}\n")

    print(f"{'arm':8}{'train types':>12}{'saw of shared zs':>18}"
          f"{'own zs set':>12}")
    rows = {}
    for a in arms:
        seen = shared_zs & train[a]
        own = test - train[a]
        rows[a] = {"train_types": len(train[a]), "saw_of_shared": len(seen),
                   "shared_zs": len(shared_zs), "own_zs": len(own)}
        print(f"{a:8}{len(train[a]):>12}{len(seen):>10} / {len(shared_zs):<5}"
              f"{len(own):>12}")

    print()
    for a in arms:
        r = rows[a]
        if r["saw_of_shared"] == 0:
            print(f"  {a}: genuinely zero-shot under the shared reference.")
        else:
            pct = r["saw_of_shared"] / max(1, r["shared_zs"])
            print(f"  {a}: saw {pct:.0%} of the shared zero-shot set in its own "
                  f"training, so its zR is not zero-shot for it.")
    print()
    print("Reading: zR under a shared reference measures recall of relation "
          "types the reference annotation lacks. That is a statement about "
          "label coverage. It is a statement about compositional "
          "generalisation only for the arm whose labels defined the "
          "reference.")

    out = Path("outputs/zeroshot_reference_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"reference": args.reference, "test_types": len(test),
         "shared_zero_shot": len(shared_zs), "arms": rows}, indent=2),
        encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
