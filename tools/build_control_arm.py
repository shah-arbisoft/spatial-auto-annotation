"""Build the human-annotated control arm for the validation study.

The study as deployed samples only the tool's *extra* predictions, so a crowd
precision of 0.585 has nothing to be read against: the same people, the same
640x480 images and the same "answer WRONG when unsure" instruction might score
the human annotations no higher. Without that control the number is
uninterpretable, which is the same argument 4.2 makes for its baselines and
4.6 for its tenth annotator.

This emits claims drawn from the *human* annotations, in the identical format
and the identical strata, so the two arms are scored by the same raters in one
indistinguishable pool. Nothing marks a claim as control in the exported
file the site consumes; the arm is recovered at scoring time from the key.

    python tools/build_control_arm.py --per-predicate 100

Writes two files:
    outputs/validation/control_claims.csv   -> load into the site
    outputs/validation/control_key.csv      -> keep private, join at scoring

Seeded, so the same call reproduces the same sample.
"""
from __future__ import annotations

import argparse
import collections
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "outputs" / "pairs.csv"
OUT = ROOT / "outputs" / "validation"

PREDICATES = ["on", "under", "to the left of", "to the right of",
              "in front of", "behind", "near"]


def split(cell: str) -> list[str]:
    return [p.strip() for p in (cell or "").split(";") if p.strip()]


def load_pairs():
    with open(PAIRS, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-predicate", type=int, default=100,
                    help="control claims per predicate (default 100)")
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--start-id", type=int, default=3001,
                    help="ids start here so they cannot collide with the "
                         "deployed 1-2002 treatment claims")
    a = ap.parse_args()

    if not PAIRS.exists():
        print(f"missing {PAIRS}; run scripts/reannotate_from_cache.py first")
        return 1

    rows = load_pairs()
    print(f"  {len(rows):,} ordered pairs in the cache")

    # a control claim is a relation a HUMAN wrote down for this pair
    by_pred: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        for g in split(r.get("gold", "")):
            if g in PREDICATES:
                by_pred[g].append({"image_id": r["image_id"], "subj": r["subj"],
                                   "obj": r["obj"], "predicate": g})

    print("\n  human-annotated triplets available per predicate:")
    short = []
    for p in PREDICATES:
        n = len(by_pred[p])
        flag = ""
        if n < a.per_predicate:
            flag = f"   <-- fewer than {a.per_predicate}, will take all"
            short.append(p)
        print(f"    {p:20} {n:6}{flag}")

    rng = random.Random(a.seed)
    claims, key = [], []
    cid = a.start_id
    for p in PREDICATES:
        pool = by_pred[p]
        take = pool if len(pool) <= a.per_predicate else rng.sample(pool, a.per_predicate)
        for c in sorted(take, key=lambda x: (x["image_id"], int(x["subj"]), int(x["obj"]))):
            claims.append({
                "claim_id": cid,
                "image_id": c["image_id"],
                "subj": c["subj"],
                "obj": c["obj"],
                # the sentence the rater sees, identical in form to the
                # treatment arm: no wording distinguishes the two
                "sentence": f"the {c['subj']} is {p} the {c['obj']}",
            })
            key.append({"claim_id": cid, "predicate": p, "arm": "control",
                        "source": "human annotation"})
            cid += 1

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "control_claims.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(claims[0]))
        w.writeheader(); w.writerows(claims)
    with open(OUT / "control_key.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key[0]))
        w.writeheader(); w.writerows(key)

    print(f"\n  {len(claims)} control claims, ids {a.start_id}-{cid-1}")
    print(f"    -> {OUT / 'control_claims.csv'}  (load into the site)")
    print(f"    -> {OUT / 'control_key.csv'}     (private; join at scoring)")
    if short:
        print(f"\n  note: {', '.join(short)} had fewer than {a.per_predicate} "
              f"human triplets, so those strata are smaller")
    print("\n  interleave these with the treatment claims so a rater cannot "
          "tell the arms apart;\n  score with:  python analysis/score_votes.py "
          "votes.csv --claims <treatment+control key>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
