"""Score the independent validation study's votes. Rules fixed before the data.

Appendix E.3 pre-declares how this study is scored, and a pre-declaration is
only worth something if the rules are written down before the responses are
seen. This file is committed for that reason: the collection was still running
when it was written, and every threshold below comes from E.3 rather than from
anything observed in the returns.

What E.3 promises, and this implements:

  ties resolve to WRONG          matching the conservative audit protocol of
                                 4.4, where anything not clearly true is wrong
  reflex-speed filter            responses faster than MIN_RESPONSE_MS are
                                 dropped as not-looked-at
  outlier-rater filter           a rater disagreeing with the majority on more
                                 than MAX_RATER_DISAGREEMENT of their shared
                                 items is dropped
  crowd precision per predicate  with Wilson binomial intervals
  author agreement               percentage and Cohen's kappa (Cohen, 1960) on
                                 the claims carrying both verdicts
  crowd reliability              Krippendorff's alpha, nominal

Usage:

    python analysis/score_votes.py votes.csv --out outputs/validation_study.json
    python analysis/score_votes.py --schema      # what the CSV must contain
    python analysis/score_votes.py --selftest    # synthetic end-to-end check

The CSV needs one row per judgement:

    claim_id,predicate,verdict,rater_id,response_ms[,author_verdict]

verdict is TRUE or WRONG (WRONG covers "can't tell", as the instructions told
participants). author_verdict is present only on the overlap claims.
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import math
import random
import sys
from pathlib import Path

# --- pre-declared constants (Appendix E.3) ----------------------------------
MIN_RESPONSE_MS = 800          # below this the item cannot have been looked at
MAX_RATER_DISAGREEMENT = 0.60  # a rater disagreeing with the majority more
                               # often than this is not judging the same task
TIE_RESOLVES_TO = "WRONG"
PREDICATES = ["on", "under", "to the left of", "to the right of",
              "in front of", "behind", "near"]

TRUE, WRONG = "TRUE", "WRONG"


def normalise(v: str) -> str:
    """Map whatever the site exported onto TRUE/WRONG.

    "can't tell" is WRONG by instruction: participants were told to answer
    WRONG when unsure, which reproduces the audit's conservative rule.
    """
    s = (v or "").strip().lower()
    if s in {"true", "t", "yes", "y", "1", "correct"}:
        return TRUE
    if s in {"wrong", "w", "no", "n", "0", "false", "incorrect",
             "can't tell", "cant tell", "unsure", "skip"}:
        return WRONG
    raise ValueError(f"unrecognised verdict {v!r}")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval; the same one 4.4 reports for the author audit."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    """Cohen's kappa between two raters over the same items (Cohen, 1960)."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    agree = sum(1 for a, b in pairs if a == b) / n
    a_counts = collections.Counter(a for a, _ in pairs)
    b_counts = collections.Counter(b for _, b in pairs)
    chance = sum(a_counts[c] / n * b_counts[c] / n for c in {TRUE, WRONG})
    return float("nan") if chance == 1 else (agree - chance) / (1 - chance)


def krippendorff_alpha(units: dict[str, list[str]]) -> float:
    """Nominal-scale Krippendorff's alpha over units with 2+ judgements.

    Computed from the coincidence matrix directly rather than through a
    library, so the dissertation's one dependency-free claim holds here too.
    """
    used = {u: v for u, v in units.items() if len(v) >= 2}
    if not used:
        return float("nan")
    coin: collections.Counter = collections.Counter()
    n_total = 0.0
    for vals in used.values():
        m = len(vals)
        for i, a in enumerate(vals):
            for j, b in enumerate(vals):
                if i != j:
                    coin[(a, b)] += 1.0 / (m - 1)
        n_total += m
    marg = collections.Counter()
    for (a, b), c in coin.items():
        marg[a] += c
    obs = sum(coin[(c, c)] for c in marg)
    exp = sum(marg[c] * (marg[c] - 1) for c in marg) / (n_total - 1) if n_total > 1 else 0
    do = n_total - obs
    de = n_total - exp
    return float("nan") if de == 0 else 1 - do / de


# --- pipeline ---------------------------------------------------------------
def read_votes(handle) -> list[dict]:
    rows = []
    for i, r in enumerate(csv.DictReader(handle), 2):
        r = { (k or "").strip().lower(): (v or "").strip() for k, v in r.items() }
        if not r.get("claim_id"):
            continue
        try:
            rows.append({
                "claim_id": r["claim_id"],
                "predicate": r.get("predicate", ""),
                "verdict": normalise(r["verdict"]),
                "rater_id": r.get("rater_id", f"anon{i}"),
                "response_ms": int(float(r["response_ms"])) if r.get("response_ms") else None,
                "author_verdict": normalise(r["author_verdict"]) if r.get("author_verdict") else None,
            })
        except (KeyError, ValueError) as e:
            raise SystemExit(f"row {i}: {e}") from None
    return rows


def apply_filters(rows: list[dict]) -> tuple[list[dict], dict]:
    """The two pre-declared exclusions, reported rather than applied silently."""
    kept = [r for r in rows if r["response_ms"] is None
            or r["response_ms"] >= MIN_RESPONSE_MS]
    dropped_fast = len(rows) - len(kept)

    # majority per claim on what survives the speed filter
    by_claim: dict[str, list[str]] = collections.defaultdict(list)
    for r in kept:
        by_claim[r["claim_id"]].append(r["verdict"])
    majority = {c: majority_of(v) for c, v in by_claim.items()}

    disagreement: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for r in kept:
        if len(by_claim[r["claim_id"]]) < 2:
            continue                                   # no majority to differ from
        d = disagreement[r["rater_id"]]
        d[1] += 1
        if r["verdict"] != majority[r["claim_id"]]:
            d[0] += 1
    outliers = {rid for rid, (bad, tot) in disagreement.items()
                if tot >= 5 and bad / tot > MAX_RATER_DISAGREEMENT}
    final = [r for r in kept if r["rater_id"] not in outliers]
    return final, {
        "judgements_submitted": len(rows),
        "dropped_faster_than_%dms" % MIN_RESPONSE_MS: dropped_fast,
        "raters_excluded_as_outliers": sorted(outliers),
        "judgements_scored": len(final),
    }


def majority_of(verdicts: list[str]) -> str:
    """Ties resolve to WRONG, as E.3 pre-declares."""
    t = verdicts.count(TRUE)
    w = verdicts.count(WRONG)
    if t > w:
        return TRUE
    if w > t:
        return WRONG
    return TIE_RESOLVES_TO


def score(rows: list[dict]) -> dict:
    rows, filt = apply_filters(rows)

    by_claim: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_claim[r["claim_id"]].append(r)

    verdict = {c: majority_of([r["verdict"] for r in rs])
               for c, rs in by_claim.items()}
    predicate = {c: rs[0]["predicate"] for c, rs in by_claim.items()}

    per_pred = {}
    for p in PREDICATES:
        claims = [c for c in verdict if predicate.get(c) == p]
        k = sum(1 for c in claims if verdict[c] == TRUE)
        lo, hi = wilson(k, len(claims))
        per_pred[p] = {"claims": len(claims), "judged_true": k,
                       "crowd_precision": (k / len(claims)) if claims else None,
                       "wilson_95": [round(lo, 4), round(hi, 4)]}

    n = len(verdict)
    k = sum(1 for c in verdict if verdict[c] == TRUE)
    lo, hi = wilson(k, n)

    # author comparison, on the claims carrying both
    pairs = []
    for c, rs in by_claim.items():
        av = next((r["author_verdict"] for r in rs if r["author_verdict"]), None)
        if av:
            pairs.append((verdict[c], av))
    agree = sum(1 for a, b in pairs if a == b)

    alpha = krippendorff_alpha(
        {c: [r["verdict"] for r in rs] for c, rs in by_claim.items()})

    return {
        "filters": filt,
        "claims_with_a_verdict": n,
        "crowd_precision_pooled": (k / n) if n else None,
        "crowd_precision_wilson_95": [round(lo, 4), round(hi, 4)],
        "per_predicate": per_pred,
        "author_comparison": {
            "claims_with_both_verdicts": len(pairs),
            "agreement": (agree / len(pairs)) if pairs else None,
            "cohens_kappa": round(cohen_kappa(pairs), 4) if pairs else None,
        },
        "crowd_reliability": {
            "krippendorff_alpha": round(alpha, 4) if alpha == alpha else None,
            "claims_with_2plus_judgements":
                sum(1 for rs in by_claim.values() if len(rs) >= 2),
        },
        "rules": {
            "ties_resolve_to": TIE_RESOLVES_TO,
            "min_response_ms": MIN_RESPONSE_MS,
            "max_rater_disagreement": MAX_RATER_DISAGREEMENT,
            "note": "fixed in Appendix E.3 before collection closed",
        },
    }


SCHEMA = """one row per judgement, header required:

  claim_id        the claim judged (repeats across raters)
  predicate       one of: """ + ", ".join(PREDICATES) + """
  verdict         TRUE or WRONG ("can't tell" counts as WRONG, by instruction)
  rater_id        the browser's random identifier
  response_ms     milliseconds on the item; blank is allowed, never dropped
  author_verdict  optional, only on the claims that also carry one

example:

  claim_id,predicate,verdict,rater_id,response_ms,author_verdict
  c0001,on,TRUE,r7f3a,4210,TRUE
  c0001,on,WRONG,r91bc,3115,
"""


def selftest() -> int:
    """End-to-end on synthetic votes, so the script is known to run today."""
    rng = random.Random(42)
    rows = ["claim_id,predicate,verdict,rater_id,response_ms,author_verdict"]
    for i in range(300):
        p = PREDICATES[i % 7]
        truth = TRUE if rng.random() < 0.85 else WRONG
        n_raters = 3 if i < 40 else 1
        for j in range(n_raters):
            v = truth if rng.random() < 0.9 else (WRONG if truth == TRUE else TRUE)
            rows.append(f"c{i:04d},{p},{v},r{j}{i%11},{rng.randint(900,9000)},"
                        f"{truth if i < 40 and j == 0 else ''}")
    rows.append(f"c0001,on,WRONG,rSPEED,120,")          # reflex, must be dropped
    for i in range(8):                                   # contrarian, must be dropped
        rows.append(f"c{i:04d},{PREDICATES[i%7]},WRONG,rBAD,4000,")

    res = score(read_votes(io.StringIO("\n".join(rows))))
    f = res["filters"]
    assert f["dropped_faster_than_800ms"] == 1, f
    assert "rBAD" in f["raters_excluded_as_outliers"], f
    assert 0.7 < res["crowd_precision_pooled"] < 1.0, res["crowd_precision_pooled"]
    assert res["author_comparison"]["claims_with_both_verdicts"] == 40
    assert res["crowd_reliability"]["krippendorff_alpha"] is not None
    for p in PREDICATES:
        assert res["per_predicate"][p]["claims"] > 0
    print("selftest ok: filters fire, precision computed, kappa and alpha "
          "defined, all seven predicates covered")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("votes", nargs="?", help="exported votes CSV")
    ap.add_argument("--out", default="outputs/validation_study.json")
    ap.add_argument("--schema", action="store_true", help="print the expected CSV")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.schema:
        print(SCHEMA)
        return 0
    if a.selftest:
        return selftest()
    if not a.votes:
        ap.error("give a votes CSV, or --schema, or --selftest")

    with open(a.votes, encoding="utf-8-sig", newline="") as fh:
        res = score(read_votes(fh))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, sort_keys=True), encoding="utf-8")

    print(f"{res['claims_with_a_verdict']} claims scored from "
          f"{res['filters']['judgements_scored']} judgements")
    print(f"crowd precision {res['crowd_precision_pooled']:.3f} "
          f"{res['crowd_precision_wilson_95']}")
    ac = res["author_comparison"]
    if ac["claims_with_both_verdicts"]:
        print(f"author agreement {ac['agreement']:.3f} "
              f"(kappa {ac['cohens_kappa']}) on {ac['claims_with_both_verdicts']} claims")
    print(f"krippendorff alpha {res['crowd_reliability']['krippendorff_alpha']}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
