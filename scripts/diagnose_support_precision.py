"""Why the support rule's extra predictions are mostly wrong, and what would fix it.

The blind audit puts support precision at 0.404 by the author and 0.638 by the
model, against a shipped claim of ~0.9. This locates the cause.

It is the threshold. `on_contact_min` ships at 0.60, and the audited claims
below 0.85 contact are correct about one time in eleven while those above are
correct about two times in three. The reason it was set there is visible in
D.2: it was fitted on train F1 against the human annotation, and the human
annotation covers about a tenth of ordered pairs. A false positive on one of
the other nine tenths is not in the gold, so it cost the fit nothing. The
plateau D.2 calls "uncritical" from 0.60 to 0.80 is flat because the metric
could not see the error the parameter controls.

A second, independent signal is the supporting object's size. `on(A, B)`
requires B to be something that can hold A up, and a 20 px cube is not a
surface. Splitting on contact and on base area separates the audited claims
far better than either alone.

**Nothing here is shipped, and the precision figures below must not be quoted
as results.** Both cut-offs were chosen by looking at the audit, so the
precision they produce is measured on the sample that selected them and is
optimistic by an unknown amount. Establishing the gain honestly needs a fresh
sample drawn under the new rule; this script exists to size the opportunity
and to record how it was found.

    python scripts/diagnose_support_precision.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_con: dict = {}
_geo: dict = {}


def contact(iid: str):
    if iid not in _con:
        a, b = iid.split("/")
        p = ROOT / f"outputs/geometry/{a}/{b}.contact.json"
        _con[iid] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return _con[iid]


def objects(iid: str):
    if iid not in _geo:
        a, b = iid.split("/")
        _geo[iid] = {o["idx"]: o for o in json.loads(
            (ROOT / f"outputs/geometry/{a}/{b}.json").read_text(encoding="utf-8"))}
    return _geo[iid]


def frac(iid, s, o):
    c = contact(iid)
    return c.get(f"{s}-{o}", c.get(f"{o}-{s}"))


def area(o) -> float:
    x1, y1, x2, y2 = o["box"]
    return (x2 - x1) * (y2 - y1)


def base_of(iid, subj, obj, predicate):
    """on(A,B): B holds A up. under(A,B): A is below B, so A is the base."""
    o = objects(iid)
    return o[int(obj)] if predicate == "on" else o[int(subj)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="outputs/audit_v3")
    ap.add_argument("--min-base", type=float, default=0.002)
    a = ap.parse_args()
    pack = ROOT / a.pack

    key = {r["id"]: r for r in csv.DictReader(
        open(pack / "_key_do_not_share.csv", encoding="utf-8"))}
    verdict = {r["id"]: r["verdict (y/n)"].strip().lower() for r in csv.DictReader(
        open(pack / "audit_sheet_blind.csv", encoding="utf-8"))}

    audited = []
    for i, r in key.items():
        if r["kind"] != "claim" or r["predicate"] not in ("on", "under"):
            continue
        f = frac(r["image_id"], r["subj"], r["obj"])
        if f is None:
            continue
        b = area(base_of(r["image_id"], r["subj"], r["obj"], r["predicate"]))
        audited.append((f, b, verdict[i] == "y"))

    gold_n, recovered = 0, []
    with open(ROOT / "outputs/pairs.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            gold = set(r["gold"].split(";")) if r["gold"] else set()
            pred = set(r["pred"].split(";")) if r["pred"] else set()
            for k in ("on", "under"):
                if k not in gold:
                    continue
                gold_n += 1
                if k not in pred:
                    continue
                f = frac(r["image_id"], r["subj"], r["obj"])
                if f is not None:
                    recovered.append(
                        (f, area(base_of(r["image_id"], r["subj"], r["obj"], k))))

    print(f"  {len(audited)} audited support claims, {gold_n} gold support triplets\n")
    print(f"  {'contact >=':>11}  {'gold recall':>13}  {'audited precision (IN-SAMPLE)':>30}")
    for t in (0.60, 0.70, 0.80, 0.85, 0.90, 0.95):
        rec = sum(1 for f, _ in recovered if f >= t)
        keep = [v for f, _, v in audited if f >= t]
        pr = f"{sum(keep)}/{len(keep)} = {sum(keep)/len(keep):.3f}" if keep else "-"
        print(f"  {t:11.2f}  {rec/gold_n:13.3f}  {pr:>30}")

    print(f"\n  adding a base-size gate (base > {a.min_base:.1%} of the frame)")
    for t in (0.85, 0.90):
        rec = sum(1 for f, b in recovered if f >= t and b > a.min_base)
        keep = [v for f, b, v in audited if f >= t and b > a.min_base]
        drop = [v for f, b, v in audited if not (f >= t and b > a.min_base)]
        print(f"    contact>={t:.2f} + base   recall {rec/gold_n:.3f}   "
              f"precision {sum(keep)}/{len(keep)} = {sum(keep)/len(keep):.3f}   "
              f"discards {sum(drop)} true of {len(drop)}")

    print("\n  shipped rule for comparison: recall "
          f"{sum(1 for _ in recovered)/gold_n:.3f}, audited precision "
          f"{sum(v for _,_,v in audited)}/{len(audited)} = "
          f"{sum(v for _,_,v in audited)/len(audited):.3f}")
    print("\n  NOT SHIPPED. Both cut-offs were chosen by inspecting this audit, so"
          "\n  the precision above is in-sample. A fresh draw is needed to claim it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
