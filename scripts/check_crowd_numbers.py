"""Check every crowd-study figure quoted in the dissertation against the data.

Section 4.15 and Appendix E.3 quote the validation arm in prose, and the
comparison table in 4.15 is a copy of the generated one. None of that
updates itself. When the study returns more votes, `eval/crowd_validation.py`
recomputes the JSON and this says which sentences still carry the old
numbers, so the refresh is a list rather than a hunt.

    python eval/crowd_validation.py && python scripts/check_crowd_numbers.py

Exits non-zero if anything is stale.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISS = ROOT / "dissertation"
DATA = ROOT / "outputs" / "crowd_validation.json"
TABLE = ROOT / "outputs" / "tables" / "crowd_validation.md"

WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
         7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
         12: "twelve"}


def flat(text: str) -> str:
    """Markdown is hard-wrapped, so a quoted figure can straddle a line."""
    return " ".join(text.split())


def main() -> int:
    if not DATA.exists():
        print(f"  no {DATA.relative_to(ROOT)}; run eval/crowd_validation.py")
        return 1
    # The volunteer arm is reported only in the branch that carries it. On the
    # branch without it there is nothing to check and that is not a failure;
    # the data and this script stay so the other version still builds.
    ch4 = (DISS / "chapter4_results_rq1.md").read_text(encoding="utf-8")
    if "4.15 A disinterested check" not in ch4:
        print("  4.15 and E.3 are not in this version of the dissertation; "
              "nothing to check")
        return 0
    d = json.loads(DATA.read_text(encoding="utf-8"))
    t = d["turnout"]
    ab = d["author_bias_check"]
    sup = d["support_pooled"]
    per = {r["predicate"]: r for r in d["per_predicate"]}

    files = {p.name: flat(p.read_text(encoding="utf-8"))
             for p in list(DISS.glob("chapter*.md"))
             + [DISS / "appendices.md", DISS / "frontmatter.md"]}

    # (expected string, where it must appear, what it is)
    checks: list[tuple[str, tuple[str, ...], str]] = [
        (f"{sup['crowd']['p']:.3f}", ("chapter4_results_rq1.md",),
         "crowd support precision"),
        (f"{sup['author']['p']:.3f}", ("chapter4_results_rq1.md",),
         "author support precision"),
        (f"{sup['model']['p']:.3f}", ("chapter4_results_rq1.md",),
         "model support precision"),
        (f"{ab['agreement']:.3f}", ("chapter4_results_rq1.md",),
         "crowd/author agreement"),
        (f"{ab['kappa']:.3f}", ("chapter4_results_rq1.md",),
         "crowd/author kappa"),
        (f"{d['krippendorff_alpha_crowd']:.3f}",
         ("chapter4_results_rq1.md",), "crowd Krippendorff alpha"),
        (str(ab["n"]), ("chapter4_results_rq1.md", "appendices.md"),
         "claims carrying both verdicts"),
        (str(t["judgements_clean"]),
         ("chapter4_results_rq1.md", "appendices.md"), "clean judgements"),
        (str(t["claims_judged"]),
         ("chapter4_results_rq1.md", "appendices.md"), "claims judged"),
        (f"{t['claims_with_2plus']}", ("chapter4_results_rq1.md",
                                       "appendices.md"),
         "claims with 2+ judgements"),
        (f"{t['coverage_fraction']:.1%}", ("chapter4_results_rq1.md",),
         "share of the designed sample"),
        (f"{t['largest_rater_share']:.0%}", ("chapter4_results_rq1.md",
                                             "appendices.md"),
         "largest rater's share"),
        (WORDS.get(t["raters"], str(t["raters"])) + " raters",
         ("chapter4_results_rq1.md", "appendices.md",
          "chapter1_introduction.md", "chapter7_critical_evaluation.md",
          "chapter9_conclusions.md"),
         "rater count in prose"),
    ]
    for pred in ("to the left of", "to the right of", "in front of",
                 "behind", "near"):
        r = per.get(pred)
        if r:
            checks.append((f"{r['crowd']['p']:.3f}",
                           ("chapter4_results_rq1.md",),
                           f"crowd precision, {pred}"))

    stale = 0
    for expected, where, what in checks:
        # The prose writes thousands with a separator, so 1415 appears as
        # "1,415". Accept either form rather than reporting a formatting
        # difference as a stale figure.
        forms = {expected}
        if expected.isdigit() and len(expected) > 3:
            forms.add(f"{int(expected):,}")
        hits = [f for f in where
                if any(v in files.get(f, "") for v in forms)]
        if not hits:
            stale += 1
            print(f"  STALE  {what}: no file quotes {expected!r}")
            print(f"         looked in: {', '.join(where)}")

    # the comparison table in 4.15 is a verbatim copy of the generated one
    if TABLE.exists():
        want = [flat(ln) for ln in TABLE.read_text(encoding="utf-8").strip()
                .splitlines()]
        ch4 = files["chapter4_results_rq1.md"]
        missing = [ln for ln in want if ln not in ch4]
        if missing:
            stale += len(missing)
            print(f"  STALE  4.15's table: {len(missing)} of {len(want)} rows "
                  "differ from the generated one")
            for ln in missing[:4]:
                print(f"         want: {ln[:78]}")

    if stale:
        print(f"\n  {stale} figure(s) in the text no longer match the data.")
        print("  Update 4.15 and E.3, then re-run this.")
        return 1
    print(f"  all {len(checks)} quoted crowd figures and the 4.15 table "
          "match the data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
