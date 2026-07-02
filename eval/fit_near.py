"""Fit the `near` threshold (size-relative box gap) to the human labels.

Reads the per-pair records written by scripts/run_annotator.py (pairs.csv),
deduplicates to unordered pairs (near is symmetric), restricts to human-annotated
non-contact pairs (see --all-pairs for why), and sweeps the threshold to maximise
F1 against the human `near` annotations — fitted on the training annotator groups,
reported on the held-out groups. Set the fitted value as predicates.near_T.

Context (measured, see docs/DATASET_NOTES.md): only 3 of 9 annotator groups ever
used `near`; at a common threshold their recall is ~1.0 but each labelled a
different fraction of qualifying close pairs, so held-out precision reflects
annotator exhaustiveness, not tool error.

    python eval/fit_near.py --pairs outputs/pairs.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.predicates import fit_near_threshold


def load_unordered(pairs_csv: str):
    """Return (metric, gold_near, groups, annotated, contact) over unique
    unordered pairs.

    Each unordered pair {i,j} appears twice (i,j) and (j,i); the metric is
    identical, and we treat the pair as human-`near` if either direction was."""
    seen: dict[tuple, list] = {}
    with open(pairs_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["image_id"], frozenset((r["subj"], r["obj"])))
            near = int(r["gold_near"])
            if key in seen:
                seen[key][1] = max(seen[key][1], near)
            else:
                seen[key] = [float(r["near_metric"]), near,
                             r["image_id"].split("/")[0],
                             int(r["gold_any"]), int(r["gold_contact"])]
    vals = list(seen.values())
    metric = np.array([v[0] for v in vals], dtype=float)
    gold = np.array([v[1] for v in vals], dtype=int)
    groups = np.array([v[2] for v in vals])
    annotated = np.array([v[3] for v in vals], dtype=bool)
    contact = np.array([v[4] for v in vals], dtype=bool)
    return metric, gold, groups, annotated, contact


def prf(dists, gold, T):
    pred = (dists <= T).astype(int)
    tp = int(((pred == 1) & (gold == 1)).sum())
    fp = int(((pred == 1) & (gold == 0)).sum())
    fn = int(((pred == 0) & (gold == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1, tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="outputs/pairs.csv")
    ap.add_argument("--report", default="outputs/near_fit.json")
    ap.add_argument("--train-groups", default="group_0,group_1,group_2,group_3,group_4,group_5",
                    help="comma-separated groups to fit T on; the rest are the "
                         "held-out check. Pass 'all' to fit on everything.")
    ap.add_argument("--all-pairs", action="store_true",
                    help="fit on every object pair instead of the default "
                         "human-annotated, non-contact pairs. The default avoids "
                         "two contaminations: unannotated pairs are not reliable "
                         "negatives (annotation is sparse), and contact (on/under) "
                         "pairs are never labelled near by the annotators.")
    ap.add_argument("--min-near-per-group", type=int, default=10,
                    help="only groups whose annotators actually used `near` at "
                         "least this often participate in fitting/evaluation. "
                         "Measured: 6 of 9 groups never used the label — their "
                         "pairs are not meaningful near-negatives. Set 0 to keep "
                         "all groups.")
    args = ap.parse_args()

    dists, gold, groups, annotated, contact = load_unordered(args.pairs)
    if len(dists) == 0:
        raise SystemExit("no pairs found — run scripts/run_annotator.py first")

    keep = np.ones(len(dists), dtype=bool) if args.all_pairs else (annotated & ~contact)

    # restrict to annotator groups that used `near` at all
    near_per_group: dict[str, int] = {}
    for g in np.unique(groups):
        near_per_group[str(g)] = int(gold[(groups == g)].sum())
    near_groups = {g for g, c in near_per_group.items() if c >= args.min_near_per_group}
    if args.min_near_per_group > 0:
        skipped = sorted(set(near_per_group) - near_groups)
        print(f"groups using `near` (>= {args.min_near_per_group} labels): "
              f"{sorted(near_groups)}  |  excluded (never/rarely used it): {skipped}")
        keep &= np.isin(groups, list(near_groups))

    dists, gold, groups = dists[keep], gold[keep], groups[keep]
    n, n_near = len(dists), int(gold.sum())

    # Fit on the training groups only, so the reported held-out agreement is not
    # circular (the threshold never saw those annotators' images).
    if args.train_groups == "all":
        tr = np.ones(n, dtype=bool)
    else:
        train_set = set(args.train_groups.split(","))
        tr = np.isin(groups, list(train_set))
    ho = ~tr

    T, f1_train = fit_near_threshold(dists[tr], gold[tr])
    p_tr, r_tr, f1_tr, *_ = prf(dists[tr], gold[tr], T)

    report = {
        "fitted_near_T": round(T, 4),
        "train": {"groups": sorted(set(groups[tr])), "n_pairs": int(tr.sum()),
                  "n_human_near": int(gold[tr].sum()),
                  "precision": round(p_tr, 4), "recall": round(r_tr, 4), "f1": round(f1_tr, 4)},
    }
    print(f"pairs: {n}  human-near: {n_near} ({100*n_near/n:.1f}%)")
    print(f"fitted near_T = {T:.4f} on {int(tr.sum())} train pairs "
          f"-> train F1={f1_tr:.4f} (P={p_tr:.4f} R={r_tr:.4f})")

    if ho.any():
        p_ho, r_ho, f1_ho, tp, fp, fn = prf(dists[ho], gold[ho], T)
        report["holdout"] = {
            "groups": sorted(set(groups[ho])), "n_pairs": int(ho.sum()),
            "n_human_near": int(gold[ho].sum()),
            "precision": round(p_ho, 4), "recall": round(r_ho, 4), "f1": round(f1_ho, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }
        print(f"HELD-OUT ({','.join(sorted(set(groups[ho])))}): "
              f"F1={f1_ho:.4f} (P={p_ho:.4f} R={r_ho:.4f})  [the number to report]")

    # per-annotator-group breakdown at the fitted T — the inter-annotator
    # consistency table (recall ~1.0 with varying precision means annotators
    # agree on what near looks like but applied the label non-exhaustively)
    report["per_group_at_T"] = {}
    print("per-group agreement at the fitted T:")
    for g in sorted(set(groups)):
        m = groups == g
        p_g, r_g, f1_g, *_ = prf(dists[m], gold[m], T)
        report["per_group_at_T"][g] = {"n_near": int(gold[m].sum()),
                                       "precision": round(p_g, 4),
                                       "recall": round(r_g, 4), "f1": round(f1_g, 4)}
        print(f"  {g}: near={int(gold[m].sum()):>4}  P={p_g:.3f} R={r_g:.3f} F1={f1_g:.3f}")

    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report -> {args.report}")
    print(f"\nSet this in configs/default.yaml:  predicates.near_T: {round(T, 4)}")


if __name__ == "__main__":
    main()
