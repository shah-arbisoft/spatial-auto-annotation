"""Score the blinded, decoy-controlled audit: author, model, and the two together.

The re-audit behind the reported support precision had thirty samples, no
decoys and one judge, who was the author. This scores the pack built to fix
all three at once, and it reports four things the old sheet could not.

**Precision per predicate, with intervals.** Claims only, Wilson score
intervals, so the estimate can be quoted as a number rather than a rounding.

**How generous each judge is.** Decoys are relations the tool did not emit and
no human labelled. A judge who agreed with everything scores zero on them. The
gap between a judge's claim rate and their decoy rejection rate is the closest
thing this design has to a direct measure of bias, and comparing the author's
to the model's is what answers the circularity 2.9 raises and 7.4 concedes.

**Whether the two judges agree**, as raw agreement and as Cohen's kappa, since
raw agreement flatters any pair of judges who both mostly say yes.

**The support figure**, pooled over `on` and `under`, which is the number the
abstract, 4.4, 4.8, 7.1 and Appendix D all state and which this pack exists to
put on a defensible footing.

One caveat is applied throughout rather than mentioned once: a decoy is a
relation the tool did not emit *and* the annotators did not label, but gold
covers about a tenth of pairs, so an unemitted relation can still be true. A
decoy accepted may be a judge being generous or the tool having missed
something real. Decoy rejection is therefore a lower bound on calibration.

    python scripts/score_audit_v3.py --pack outputs/audit_v3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

PREDS = ("on", "under", "to the left of", "to the right of",
         "in front of", "behind", "near")
SUPPORT = ("on", "under")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(hits: int, n: int) -> str:
    if not n:
        return "     -        "
    lo, hi = wilson(hits, n)
    return f"{hits:3}/{n:<3} {hits/n:.3f} [{lo:.2f},{hi:.2f}]"


def kappa(a: dict, b: dict, ids) -> tuple[float, float]:
    """Cohen's kappa between two judges, plus raw agreement."""
    ids = [i for i in ids if i in a and i in b]
    if not ids:
        return (0.0, 0.0)
    po = sum(a[i] == b[i] for i in ids) / len(ids)
    pe = sum((sum(a[i] == c for i in ids) / len(ids)) *
             (sum(b[i] == c for i in ids) / len(ids)) for c in ("y", "n"))
    return ((po - pe) / (1 - pe) if pe < 1 else 1.0), po


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="outputs/audit_v3")
    ap.add_argument("--out", default="outputs/audit_v3_scores.json")
    a = ap.parse_args()
    pack = Path(a.pack)

    key = {r["id"]: r for r in csv.DictReader(
        open(pack / "_key_do_not_share.csv", encoding="utf-8"))}
    author = {r["id"]: r["verdict (y/n)"].strip().lower()
              for r in csv.DictReader(
                  open(pack / "audit_sheet_blind.csv", encoding="utf-8"))
              if r["verdict (y/n)"].strip().lower() in ("y", "n")}
    model = {r["id"]: r["verdict"] for r in
             (json.loads(l) for l in
              open(pack / "vlm_verdicts.jsonl", encoding="utf-8") if l.strip())
             if r.get("verdict") in ("y", "n")}

    claims = [i for i in key if key[i]["kind"] == "claim"]
    decoys = [i for i in key if key[i]["kind"] == "decoy"]
    print(f"  {len(key)} items: {len(claims)} claims, {len(decoys)} decoys")
    print(f"  author verdicts {len(author)}, model verdicts {len(model)}")
    missing = sorted(set(key) - set(author), key=int)
    if missing:
        print(f"  author sheet incomplete, ids {missing} excluded throughout")

    out: dict = {"n_claims": len(claims), "n_decoys": len(decoys)}

    # ---- precision on claims, per predicate --------------------------------
    print(f"\n  PRECISION ON EMITTED CLAIMS\n"
          f"    {'predicate':18} {'author':22} {'model':22}")
    out["precision"] = {}
    for p in PREDS:
        ids = [i for i in claims if key[i]["predicate"] == p]
        ah = sum(1 for i in ids if author.get(i) == "y")
        an = sum(1 for i in ids if i in author)
        mh = sum(1 for i in ids if model.get(i) == "y")
        mn = sum(1 for i in ids if i in model)
        print(f"    {p:18} {rate(ah, an):22} {rate(mh, mn):22}")
        out["precision"][p] = {"author": [ah, an], "model": [mh, mn]}

    sup = [i for i in claims if key[i]["predicate"] in SUPPORT]
    ah = sum(1 for i in sup if author.get(i) == "y")
    an = sum(1 for i in sup if i in author)
    mh = sum(1 for i in sup if model.get(i) == "y")
    mn = sum(1 for i in sup if i in model)
    print(f"    {'SUPPORT pooled':18} {rate(ah, an):22} {rate(mh, mn):22}")
    out["support_pooled"] = {"author": [ah, an], "model": [mh, mn]}

    # ---- the decoy control: how generous is each judge? ---------------------
    print(f"\n  DECOY REJECTION (a judge who agreed with everything scores 0.000)")
    out["decoys"] = {}
    for name, v in (("author", author), ("model", model)):
        ids = [i for i in decoys if i in v]
        rej = sum(1 for i in ids if v[i] == "n")
        print(f"    {name:8} {rate(rej, len(ids))}")
        out["decoys"][name] = [rej, len(ids)]

    # ---- do the two judges agree? ------------------------------------------
    k_all, po_all = kappa(author, model, key)
    k_cl, po_cl = kappa(author, model, claims)
    print(f"\n  AUTHOR vs MODEL")
    print(f"    all items    agreement {po_all:.3f}   kappa {k_all:.3f}")
    print(f"    claims only  agreement {po_cl:.3f}   kappa {k_cl:.3f}")
    out["agreement"] = {"all": [po_all, k_all], "claims": [po_cl, k_cl]}

    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
