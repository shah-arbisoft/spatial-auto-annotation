"""How much do the human annotators differ from each other?

The dataset gives no direct answer: annotator groups never labelled the same
images, so annotator-to-annotator agreement cannot be computed. This script
recovers two things that can be computed, using the automatic annotator as a
fixed common reference.

1. ANNOTATOR HETEROGENEITY (assumption-free). The tool is deterministic, so
   it is literally the same labeller for every group. Any variation in its
   agreement across annotators is therefore variation in the annotators, not
   in the tool. The spread of per-group agreement is a direct, conservative
   read on how much the annotators differ.

2. BOUNDS ON HUMAN-HUMAN AGREEMENT. For any common reference R, if annotator
   A agrees with R on a fraction p_A of items and B on p_B, then their mutual
   agreement p_AB is constrained by the Frechet inequalities

       max(0, p_A + p_B - 1)  <=  p_AB  <=  1 - |p_A - p_B|

   These hold for agreement measured on shared items. Because the groups
   labelled disjoint image batches, applying them here assumes each
   annotator's agreement rate would carry over to another batch
   (exchangeability of the ~100-image batches). The assumption is stated
   rather than hidden, and the resulting interval is reported as an estimate,
   never as a measurement.

Convention-inverted groups (6 and 8, see chapter 4) are excluded from the
consistency analysis and reported separately: their disagreement is a known
direction convention, not a measure of annotation noise.

    python eval/annotator_agreement.py
"""

from __future__ import annotations

import itertools
import json
import statistics
import sys
from pathlib import Path

INVERTED = {"group_6", "group_8"}
REPORT = Path("outputs/fidelity_report.json")
OUT_JSON = Path("outputs/annotator_agreement.json")
OUT_MD = Path("outputs/tables/annotator_agreement.md")


def main():
    if not REPORT.exists():
        sys.exit(f"missing {REPORT}; run eval/fidelity.py first")
    per_group = json.loads(REPORT.read_text(encoding="utf-8"))["per_group_recall"]

    consistent = {g: v["recall"] for g, v in per_group.items() if g not in INVERTED}
    vals = list(consistent.values())
    mean_tool = statistics.mean(vals)

    pairs = []
    for (ga, pa), (gb, pb) in itertools.combinations(sorted(consistent.items()), 2):
        pairs.append({"a": ga, "b": gb,
                      "lower": max(0.0, pa + pb - 1.0),
                      "upper": 1.0 - abs(pa - pb)})
    lo = statistics.mean(p["lower"] for p in pairs)
    hi = statistics.mean(p["upper"] for p in pairs)

    report = {
        "per_group_agreement": {g: round(v["recall"], 4) for g, v in sorted(per_group.items())},
        "inverted_groups": sorted(INVERTED),
        "consistent_annotators": {
            "n": len(vals), "mean": round(mean_tool, 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4),
            "spread": round(max(vals) - min(vals), 4),
            "sd": round(statistics.pstdev(vals), 4),
        },
        "human_human_estimate": {
            "n_pairs": len(pairs),
            "mean_lower_bound": round(lo, 4),
            "mean_upper_bound": round(hi, 4),
            "tool_inside_interval": bool(lo <= mean_tool <= hi),
            "assumption": "batch exchangeability; see module docstring",
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# Annotator heterogeneity and estimated human-human agreement\n",
        "The tool is deterministic, so it is the same labeller for every "
        "annotator group; variation in its agreement is variation in the "
        "annotators.\n",
        "| annotator group | tool agreement | gold triplets | note |",
        "|---|---|---|---|",
    ]
    for g, v in sorted(per_group.items()):
        note = "convention-inverted" if g in INVERTED else ""
        md.append(f"| {g} | {v['recall']:.3f} | {v['gold_triplets']} | {note} |")
    md += [
        "",
        f"Consistent annotators (n={len(vals)}): agreement spans "
        f"{min(vals):.3f}-{max(vals):.3f} (spread {max(vals) - min(vals):.3f}, "
        f"sd {statistics.pstdev(vals):.3f}, mean {mean_tool:.3f}). A single "
        "fixed labeller varying this much across annotators is a direct "
        "measure of how much the annotators differ from one another.",
        "",
        f"Frechet bounds over all {len(pairs)} annotator pairs place "
        f"annotator-to-annotator agreement in **[{lo:.2f}, {hi:.2f}]** on "
        f"average, under the batch-exchangeability assumption stated in the "
        f"script. The tool's own mean agreement, {mean_tool:.3f}, lies "
        f"{'inside' if lo <= mean_tool <= hi else 'outside'} that interval.",
    ]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    print("\n".join(md))
    print(f"\nreport -> {OUT_JSON} ; table -> {OUT_MD}")


if __name__ == "__main__":
    main()
