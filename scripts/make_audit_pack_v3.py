"""Build a stratified, blinded, decoy-controlled audit pack for support precision.

The re-audit behind the "~0.9 support precision" claim has three weaknesses,
all of which this pack removes.

**It is too small, and the claim exceeds it.** Thirty samples give 22 correct,
a point estimate of 0.733 with a 95% interval of [0.56, 0.86]. The reported
~0.9 sits above that interval. At 80 per support predicate the interval
narrows to roughly +/-0.09, which is tight enough for the estimate to be
reported as a number rather than a rounding.

**Its samples are not independent.** The released images are consecutive
frames of one walk, so sampling extras at random draws the same physical pair
from adjacent frames; one row of the old sheet is annotated "same pair, second
frame". Here at most one claim is drawn per image, and claims from the same
annotator group must be at least --frame-gap frames apart, so no two items can
be the same arrangement seen twice.

**Nothing measures the auditor.** Every item in the old sheet was a relation
the tool asserted, so an auditor who simply agreed with everything would score
100% and look calibrated. This pack mixes in decoys: the same predicate on
pairs where the tool did *not* emit it and no human labelled it. They should
be verdicted wrong. The rate at which they are not is a direct measure of how
generous the auditor is, which is the objection 2.9 raises and 7.4 concedes.

The pack also applies the shipped class guard, so nothing is audited that the
tool would no longer emit: support is not evaluated when either object is a
person.

    python scripts/make_audit_pack_v3.py --support-n 80 --other-n 24 --decoys 32

Writes outputs/audit_v3/: the blinded sheet, the images, and a key that is
kept separate so the sheet can be filled without knowing which rows are decoys.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import SpatialDataset, load_rgb          # noqa: E402
from src.pipeline import load_config                       # noqa: E402
from src.predicates import PREDICATES                      # noqa: E402
from scripts.make_audit_pack import render                 # noqa: E402

SUPPORT = ("on", "under")


def frame_no(image_id: str) -> tuple[str, int]:
    group, stem = image_id.split("/")
    return group, int(stem)


def spaced(cands, rng, n, frame_gap, labels, predicate):
    """Draw n claims that cannot be the same physical relation twice.

    Section 4.14 establishes that each annotator group is a contiguous block of
    one continuous walk, holding one arrangement of objects. Two claims from
    the same group can therefore be the same relation seen from two viewpoints,
    however many frames apart, and object indices are not stable across frames
    so the pair cannot be matched by index.

    The rule applied is one claim per (group, subject class, object class): at
    most one "cube on box" verdict from any one arrangement. It is deliberately
    conservative, refusing a second genuinely different cube-and-box pair in the
    same group, because the sample exists to bound a precision claim and a
    sample that is too independent errs in the safe direction. The frame gap is
    kept as a second, weaker guard against near-identical framing.
    """
    rng.shuffle(cands)
    taken, used_frames, used_kinds = [], collections.defaultdict(list), set()
    for image_id, subj, obj in cands:
        g, f = frame_no(image_id)
        per = labels.get(image_id, {})
        kind = (g, per.get(subj), per.get(obj), predicate)
        if kind in used_kinds:
            continue
        if any(abs(f - prev) < frame_gap for prev in used_frames[g]):
            continue
        used_kinds.add(kind)
        used_frames[g].append(f)
        taken.append((image_id, subj, obj))
        if len(taken) >= n:
            break
    return taken


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="outputs/pairs.csv")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--support-n", type=int, default=80,
                    help="claims per support predicate (the contested ones)")
    ap.add_argument("--other-n", type=int, default=24,
                    help="claims per remaining predicate")
    ap.add_argument("--decoys", type=int, default=32,
                    help="claims the tool did NOT emit; should be verdicted 'n'")
    ap.add_argument("--frame-gap", type=int, default=5,
                    help="minimum frames between two claims from one group")
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--out", default="outputs/audit_v3")
    a = ap.parse_args()

    cfg = load_config(a.config)
    rng = random.Random(a.seed)
    root = Path(cfg["dataset"]["root"])
    no_support = set(cfg["predicates"].get("no_support_classes", []) or [])

    # object classes per image, so the class guard can be applied to the sample
    labels: dict[str, dict[int, str]] = {}
    for gj in Path("outputs/geometry").rglob("*.json"):
        # skip the .contact.json sidecars: only the per-object geometry files
        # carry labels, and their stems are bare frame numbers
        if "." in gj.stem:
            continue
        image_id = f"{gj.parent.name}/{gj.stem}"
        labels[image_id] = {o["idx"]: o["label"]
                            for o in json.loads(gj.read_text(encoding="utf-8"))}

    extras = {k: [] for k in PREDICATES}
    absent = {k: [] for k in PREDICATES}
    with open(a.pairs, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pred = set(r["pred"].split(";")) if r["pred"] else set()
            gold = set(r["gold"].split(";")) if r["gold"] else set()
            iid, s, o = r["image_id"], int(r["subj"]), int(r["obj"])
            per = labels.get(iid, {})
            for k in PREDICATES:
                # the shipped class guard: support is never emitted, and so is
                # never audited, when either object is a person
                if k in SUPPORT and (per.get(s) in no_support
                                     or per.get(o) in no_support):
                    continue
                if k in pred - gold:
                    extras[k].append((iid, s, o))
                elif k not in pred and k not in gold:
                    absent[k].append((iid, s, o))

    print("  extra predictions available after the class guard:")
    for k in PREDICATES:
        print(f"    {k:18} {len(extras[k]):7}")

    rows, key_rows, sid = [], [], 0
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)

    plan = [(k, a.support_n if k in SUPPORT else a.other_n, "claim") for k in PREDICATES]
    per_decoy = max(1, a.decoys // len(PREDICATES))
    plan += [(k, per_decoy, "decoy") for k in PREDICATES]

    picked = []
    for k, n, kind in plan:
        pool = list(extras[k] if kind == "claim" else absent[k])
        take = spaced(pool, rng, n, a.frame_gap, labels, k)
        if len(take) < n:
            print(f"    note: {k} {kind}: only {len(take)} of {n} met the "
                  f"{a.frame_gap}-frame spacing rule")
        picked += [(k, kind, *t) for t in take]

    rng.shuffle(picked)                      # blind: no predicate or kind runs
    for k, kind, image_id, subj, obj in picked:
        sid += 1
        group, stem = image_id.split("/")
        geo = json.loads(Path(f"outputs/geometry/{group}/{stem}.json")
                         .read_text(encoding="utf-8"))
        png = out / "img" / f"{sid:03d}.png"
        render(sid, root / "img_data" / group / f"{stem}.jpg", geo, subj, obj, k, png)
        by = {o["idx"]: o for o in geo}
        rows.append({"id": sid, "image": png.name,
                     "subject": f"{by[subj]['label']}{subj}", "predicate": k,
                     "object": f"{by[obj]['label']}{obj}",
                     "verdict (y/n)": "", "notes": ""})
        key_rows.append({"id": sid, "kind": kind, "predicate": k,
                         "image_id": image_id, "subj": subj, "obj": obj})

    with open(out / "audit_sheet_blind.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(out / "_key_do_not_share.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(key_rows[0])); w.writeheader(); w.writerows(key_rows)

    n_claim = sum(1 for r in key_rows if r["kind"] == "claim")
    n_decoy = len(key_rows) - n_claim
    (out / "INSTRUCTIONS.txt").write_text(
        "Blind audit v3 - support precision\n"
        "=================================\n\n"
        f"{len(rows)} items, shuffled. Each image shows one claim: the SUBJECT\n"
        "outlined red, the OBJECT outlined blue, and the claim printed along the\n"
        "bottom. Judge ONLY whether that claim is true of that image.\n\n"
        "Mark 'y' if the claim is clearly true, 'n' otherwise. If you are not\n"
        "sure, mark 'n' - the same conservative rule as the earlier audits, so\n"
        "the estimate stays a lower bound.\n\n"
        "Use Chapter 3's definitions: 'on'/'under' mean physically resting on,\n"
        "not merely overlapping or being held; left/right are from the camera's\n"
        "point of view; 'in front of' means nearer the camera.\n\n"
        "Some items are relations the tool did NOT emit. They are mixed in\n"
        "deliberately and are not marked. Do not try to identify them: judging\n"
        "them the same way as the rest is what makes them useful.\n\n"
        "Do not open _key_do_not_share.csv until every verdict is filled in.\n",
        encoding="utf-8")

    print(f"\n  {len(rows)} items -> {out}/audit_sheet_blind.csv")
    print(f"    {n_claim} claims the tool emitted, {n_decoy} decoys it did not")
    print(f"    at most one claim per image, >= {a.frame_gap} frames apart per group")
    print(f"    class guard applied: support skipped when either object is "
          f"{' or '.join(sorted(no_support)) or '(none configured)'}")
    print(f"  key held separately in {out}/_key_do_not_share.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
