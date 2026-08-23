"""Score the volunteer validation arm against the two audit judges.

Appendix E.3 specifies a study that re-estimates precision with judges who
did not build the tool. This reads what it returned and places it beside the
author and model verdicts from the same generation of labels, which is the
comparison the study exists to make.

The crowd item set was drawn on 17 July, a month before `on_contact_min` was
refitted to 0.85, so the arm judged the pre-refit labels and belongs against
the v3 audit pack rather than v4. For the five predicates the support
threshold does not touch, that distinction does not arise.

    python eval/crowd_validation.py

Reads outputs/validation/report.json, which `analysis/score_votes.py` in the
study repository writes from the raw votes and the item key. Neither of those
is in this repository: the key would let a future participant look up the
answers, and the votes are participant data.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs" / "validation" / "report.json"
PACK = ROOT / "outputs" / "audit_v3"
OUT = ROOT / "outputs" / "crowd_validation.json"
TABLE = ROOT / "outputs" / "tables" / "crowd_validation.md"

PREDICATES = ["on", "under", "to the left of", "to the right of",
              "in front of", "behind", "near"]
SUPPORT = ["on", "under"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / d, (centre + half) / d)


def arm(pair) -> dict:
    """A [correct, total] pair from the audit scorer, with its rate."""
    if not pair:
        return {"k": None, "n": None, "p": None}
    k, n = pair
    return {"k": k, "n": n, "p": k / n if n else None}


def audit_arm() -> dict:
    """Author and model precision from the v3 pack, via its own scorer."""
    tmp = ROOT / "outputs" / ".audit_v3_scratch.json"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "score_audit_v3.py"),
                    "--pack", str(PACK), "--out", str(tmp)],
                   check=True, capture_output=True)
    data = json.loads(tmp.read_text(encoding="utf-8"))
    tmp.unlink()
    return data


def main() -> int:
    if not REPORT.exists():
        print(f"  no validation report at {REPORT.relative_to(ROOT)};"
              " run analysis/score_votes.py in the study repository first")
        return 1
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    audit = audit_arm()
    a_pred = audit["precision"]

    rows = []
    ck = cn = 0
    for p in PREDICATES:
        c = rep["crowd_precision_per_predicate"].get(p)
        if not c:
            continue
        k = round(c["precision"] * c["n"])
        if p in SUPPORT:
            ck += k
            cn += c["n"]
        a = a_pred.get(p, {})
        rows.append({
            "predicate": p,
            "crowd": {"k": k, "n": c["n"], "p": c["precision"],
                      "ci": [c["lo"], c["hi"]]},
            "author": arm(a.get("author")),
            "model": arm(a.get("model")),
        })

    sup = audit["support_pooled"]
    pooled = {
        "crowd": {"k": ck, "n": cn, "p": ck / cn if cn else None,
                  "ci": list(wilson(ck, cn))},
        "author": arm(sup["author"]),
        "model": arm(sup["model"]),
    }

    out = {
        "label_generation": "pre-refit (on_contact_min 0.60), matching audit_v3",
        "turnout": {
            "raters": rep["n_raters"],
            "judgements_clean": rep["n_clean"],
            "claims_judged": rep["n_items_voted"],
            "claims_sampled": rep["n_items_total"],
            "coverage_fraction": rep["n_items_voted"] / rep["n_items_total"],
            "largest_rater_share": max(r["votes"] for r in rep["raters"])
                                   / rep["n_clean"],
            "claims_with_2plus": rep["coverage_by_votes"]["2"],
            # Duplicate rows were a real defect: a flush whose response never
            # reached the client left the queue intact and re-sent votes the
            # sheet already held. Fixed server-side on 11 August by a vote id
            # minted at answer time, so this should read 0 on every export
            # after that date.
            "duplicate_rows_dropped": rep["n_raw"] - rep["n_clean"],
        },
        "crowd_precision_overall": rep["crowd_precision_overall"],
        "per_predicate": rows,
        "support_pooled": pooled,
        "author_bias_check": rep["author_bias_check"],
        "krippendorff_alpha_crowd": rep["krippendorff_alpha_crowd"],
        "control_arm": "not delivered; the crowd arm carries no decoys, so"
                       " rater calibration is unmeasured",
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    def cell(d):
        if d.get("p") is None:
            return "--"
        return f"{d['k']}/{d['n']} {d['p']:.3f}"

    lines = ["| Predicate | Volunteers | Author | Model |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['predicate']} | {cell(r['crowd'])} | "
                     f"{cell(r['author'])} | {cell(r['model'])} |")
    lines.append(f"| **support pooled** | **{cell(pooled['crowd'])}** | "
                 f"**{cell(pooled['author'])}** | **{cell(pooled['model'])}** |")
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    t = out["turnout"]
    print(f"  {t['raters']} raters, {t['judgements_clean']} clean judgements, "
          f"{t['claims_judged']}/{t['claims_sampled']} claims "
          f"({t['coverage_fraction']:.1%})")
    print(f"  largest single rater supplied {t['largest_rater_share']:.0%}"
          " of the judgements")
    dups = t["duplicate_rows_dropped"]
    if dups:
        print(f"  {dups} duplicate row(s) dropped before scoring"
              + ("  <-- expected 0 after the 11 Aug server fix"
                 if dups and rep.get("n_raw", 0) > 238 else ""))
    print("\n".join("  " + ln for ln in lines))
    ab = out["author_bias_check"]
    print(f"  author agreement {ab['agreement']:.3f}, kappa {ab['kappa']:.3f}"
          f" on n={ab['n']}")
    print(f"  crowd Krippendorff alpha {out['krippendorff_alpha_crowd']:.3f}"
          f" on {t['claims_with_2plus']} claims with 2+ judgements")
    print(f"  -> {OUT.relative_to(ROOT)}, {TABLE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
