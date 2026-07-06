"""RQ1 fidelity study: automatic labels vs. the human annotations.

Metrics follow the sparse-annotation protocol (humans labelled ~10% of pairs,
so raw precision against them undercounts):

  1. Per-predicate RECALL of human triplets (primary; matches the recall-based
     convention the source paper itself uses), reported for all data and for
     the held-out annotator groups (6-8) that no threshold was fitted on.
  2. Precision/recall/F1 RESTRICTED to human-annotated ordered pairs.
  3. Confusion analysis: for each human triplet we missed, what did the tool
     say instead?
  4. Flag rates per type (abstentions vs. the borderline-near review queue).
  5. Per-annotator-group agreement (the "tenth annotator" table).
  6. Baselines: random predicate, majority predicate, and box-only geometry
     (no masks, no depth) rebuilt offline from the geometry cache.

    python eval/fidelity.py            # writes outputs/fidelity_report.json
                                       # and outputs/tables/rq1_tables.md
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.predicates import PREDICATES, Obj
from src.pipeline import annotate_objects, load_config

TRAIN_GROUPS = {f"group_{i}" for i in range(6)}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_rows(pairs_csv: str) -> list[dict]:
    rows = []
    with open(pairs_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "image_id": r["image_id"],
                "group": r["image_id"].split("/")[0],
                "subj": int(r["subj"]), "obj": int(r["obj"]),
                "pred": set(r["pred"].split(";")) if r["pred"] else set(),
                "gold": set(r["gold"].split(";")) if r["gold"] else set(),
            })
    return rows


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def recall_table(rows, pred_of) -> dict:
    """Per-predicate recall of human triplets; overall and held-out groups."""
    out = {}
    for k in PREDICATES:
        g_all = rec_all = g_ho = rec_ho = 0
        for r in rows:
            if k not in r["gold"]:
                continue
            hit = k in pred_of(r)
            g_all += 1; rec_all += hit
            if r["group"] not in TRAIN_GROUPS:
                g_ho += 1; rec_ho += hit
        out[k] = {
            "gold": g_all, "recall": rec_all / g_all if g_all else None,
            "gold_heldout": g_ho,
            "recall_heldout": rec_ho / g_ho if g_ho else None,
        }
    vals = [v["recall"] for v in out.values() if v["recall"] is not None]
    ho = [v["recall_heldout"] for v in out.values() if v["recall_heldout"] is not None]
    out["MEAN"] = {"recall": float(np.mean(vals)), "recall_heldout": float(np.mean(ho))}
    return out


def restricted_prf(rows) -> dict:
    """P/R/F1 per predicate, restricted to ordered pairs the humans annotated."""
    ann = [r for r in rows if r["gold"]]
    out = {}
    for k in PREDICATES:
        tp = sum(1 for r in ann if k in r["gold"] and k in r["pred"])
        fp = sum(1 for r in ann if k not in r["gold"] and k in r["pred"])
        fn = sum(1 for r in ann if k in r["gold"] and k not in r["pred"])
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        out[k] = {"precision": round(p, 4), "recall": round(rc, 4),
                  "f1": round(f1, 4), "support": tp + fn}
    out["n_annotated_pairs"] = len(ann)
    return out


def confusion(rows) -> dict:
    """For each gold predicate: what the tool emitted on those pairs."""
    out = {}
    for k in PREDICATES:
        c = collections.Counter()
        n = 0
        for r in rows:
            if k not in r["gold"]:
                continue
            n += 1
            if k in r["pred"]:
                c["correct"] += 1
            elif not r["pred"]:
                c["(nothing)"] += 1
            else:
                for j in r["pred"]:
                    c[j] += 1
        out[k] = {"n": n, "responses": dict(c.most_common())}
    return out


def per_group_recall(rows) -> dict:
    """Recall of human triplets per annotator group — the tenth-annotator view."""
    out = {}
    for g in sorted({r["group"] for r in rows}):
        gold = rec = 0
        for r in rows:
            if r["group"] != g or not r["gold"]:
                continue
            gold += len(r["gold"])
            rec += sum(1 for k in r["gold"] if k in r["pred"])
        out[g] = {"gold_triplets": gold, "recall": round(rec / gold, 4) if gold else None}
    return out


FLIP = {"in front of": "behind", "behind": "in front of"}


def front_behind_decomposition(rows) -> dict:
    """Decompose front/behind misses per annotator group.

    Three quantities per group: how often the tool commits to a direction at
    all (emit rate; the rest are depth_eps abstentions), how often it agrees
    with the human WHEN it commits (direction agreement), and recall under the
    group's inferred convention (majority direction when committed; one bit per
    group, disclosed). Motivated by the measured result that two groups (6, 8)
    labelled front/behind with the inverted direction convention.
    """
    per = {}
    for r in rows:
        grp = r["group"]
        for k in ("in front of", "behind"):
            if k not in r["gold"]:
                continue
            d = per.setdefault(grp, {"gold": 0, "raw": 0, "flip": 0,
                                     "emitted": 0, "agree": 0})
            d["gold"] += 1
            d["raw"] += k in r["pred"]
            d["flip"] += FLIP[k] in r["pred"]
            if r["pred"] & {"in front of", "behind"}:
                d["emitted"] += 1
                d["agree"] += k in r["pred"]

    out = {}
    aligned_hits = 0
    total_gold = 0
    for grp in sorted(per):
        d = per[grp]
        agree_rate = d["agree"] / d["emitted"] if d["emitted"] else None
        convention = ("inverted" if agree_rate is not None and agree_rate < 0.5
                      else "same")
        aligned = d["flip"] if convention == "inverted" else d["raw"]
        aligned_hits += aligned
        total_gold += d["gold"]
        out[grp] = {
            "gold": d["gold"],
            "emit_rate": round(d["emitted"] / d["gold"], 4),
            "direction_agreement_when_committed": (
                round(agree_rate, 4) if agree_rate is not None else None),
            "raw_recall": round(d["raw"] / d["gold"], 4),
            "convention": convention,
            "convention_aligned_recall": round(aligned / d["gold"], 4),
        }
    out["OVERALL"] = {
        "gold": total_gold,
        "raw_recall": round(sum(per[g]["raw"] for g in per) / total_gold, 4),
        "convention_aligned_recall": round(aligned_hits / total_gold, 4),
    }
    return out


def flag_rates(ann_dir: str, n_pairs: int) -> dict:
    by = collections.Counter(); flagged = 0
    for p in glob.glob(f"{ann_dir}/**/*.json", recursive=True):
        d = json.load(open(p, encoding="utf-8"))
        for f in d.get("review_flags", []):
            flagged += 1
            for x in f["flags"]:
                by[x] += 1
    return {"pairs_flagged": flagged, "rate": round(flagged / n_pairs, 4),
            "by_type": {k: {"count": v, "rate": round(v / n_pairs, 4)}
                        for k, v in by.most_common()}}


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
def baseline_random(rows, seed=42):
    rng = random.Random(seed)
    assign = {id(r): {rng.choice(PREDICATES)} for r in rows}
    return lambda r: assign[id(r)]


def baseline_majority(rows):
    counts = collections.Counter()
    for r in rows:
        if r["group"] in TRAIN_GROUPS:
            counts.update(r["gold"])
    major = counts.most_common(1)[0][0]
    return (lambda r: {major}), major


def baseline_box_only(rows, cfg, geo_root="outputs/geometry"):
    """Rebuild predictions from BOXES ALONE: centroids = box centres, no masks,
    constant depth (front/behind abstain). Isolates the mask+depth contribution."""
    need = {}
    for r in rows:
        need.setdefault(r["image_id"], set())
    pred_map = {}
    for image_id in need:
        group, stem = image_id.split("/")
        gp = Path(geo_root) / group / f"{stem}.json"
        geo = json.loads(gp.read_text(encoding="utf-8"))
        objs = []
        for o in geo:
            x1, y1, x2, y2 = o["box"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            objs.append(Obj(o["idx"], o["label"], tuple(o["box"]), cx, cy, 0.5,
                            np.array([cx, cy, 0.5])))
        for p in annotate_objects(objs, cfg):
            pred_map[(image_id, p.subject, p.object)] = set(p.predicates)
    return lambda r: pred_map.get((r["image_id"], r["subj"], r["obj"]), set())


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def md_recall_table(ours, rnd, majority, boxonly, major_name):
    lines = ["| Predicate | Gold | Ours | Ours (held-out) | Random | Majority"
             f" ({major_name}) | Box-only |",
             "|---|---|---|---|---|---|---|"]
    def fmt(v): return f"{v:.2f}" if v is not None else "—"
    for k in PREDICATES:
        lines.append(f"| {k} | {ours[k]['gold']} | {fmt(ours[k]['recall'])} | "
                     f"{fmt(ours[k]['recall_heldout'])} | {fmt(rnd[k]['recall'])} | "
                     f"{fmt(majority[k]['recall'])} | {fmt(boxonly[k]['recall'])} |")
    lines.append(f"| **mean** |  | **{ours['MEAN']['recall']:.2f}** | "
                 f"**{ours['MEAN']['recall_heldout']:.2f}** | {rnd['MEAN']['recall']:.2f} | "
                 f"{majority['MEAN']['recall']:.2f} | {boxonly['MEAN']['recall']:.2f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="outputs/pairs.csv")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rows = load_rows(args.pairs)
    print(f"loaded {len(rows)} ordered pairs "
          f"({sum(1 for r in rows if r['gold'])} human-annotated)")

    ours = recall_table(rows, lambda r: r["pred"])
    rnd = recall_table(rows, baseline_random(rows, cfg.get("seed", 42)))
    maj_fn, major_name = baseline_majority(rows)
    majority = recall_table(rows, maj_fn)
    print("rebuilding box-only baseline from cache ...")
    boxonly = recall_table(rows, baseline_box_only(rows, cfg))

    prf = restricted_prf(rows)
    conf = confusion(rows)
    groups = per_group_recall(rows)
    fb = front_behind_decomposition(rows)
    flags = flag_rates(f"{args.out}/annotations", len(rows))

    report = {"recall": {"ours": ours, "random": rnd, "majority": majority,
                         "majority_predicate": major_name, "box_only": boxonly},
              "restricted_prf": prf, "confusion": conf,
              "per_group_recall": groups,
              "front_behind_decomposition": fb, "flag_rates": flags}
    Path(f"{args.out}/fidelity_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    tables = Path(f"{args.out}/tables"); tables.mkdir(exist_ok=True)
    md = ["# RQ1 fidelity tables\n",
          "## T1 — Per-predicate recall of human triplets\n",
          md_recall_table(ours, rnd, majority, boxonly, major_name),
          "\n## T2 — P/R/F1 restricted to human-annotated pairs "
          f"(n={prf['n_annotated_pairs']})\n",
          "| Predicate | P | R | F1 | support |", "|---|---|---|---|---|"]
    for k in PREDICATES:
        v = prf[k]
        md.append(f"| {k} | {v['precision']:.2f} | {v['recall']:.2f} | "
                  f"{v['f1']:.2f} | {v['support']} |")
    md += ["\n## T3 — Per-annotator-group recall (tenth-annotator view)\n",
           "| Group | Gold triplets | Recall |", "|---|---|---|"]
    for g, v in groups.items():
        md.append(f"| {g} | {v['gold_triplets']} | "
                  + (f"{v['recall']:.2f}" if v["recall"] is not None else "—") + " |")
    md += ["\n## T5 — Front/behind decomposition per annotator group\n",
           "| Group | Gold | Emit rate | Agreement when committed | Convention |"
           " Raw recall | Aligned recall |",
           "|---|---|---|---|---|---|---|"]
    for g, v in fb.items():
        if g == "OVERALL":
            continue
        agr = (f"{v['direction_agreement_when_committed']:.2f}"
               if v['direction_agreement_when_committed'] is not None else "—")
        md.append(f"| {g} | {v['gold']} | {v['emit_rate']:.2f} | {agr} | "
                  f"{v['convention']} | {v['raw_recall']:.2f} | "
                  f"{v['convention_aligned_recall']:.2f} |")
    md.append(f"| **overall** | {fb['OVERALL']['gold']} |  |  |  | "
              f"**{fb['OVERALL']['raw_recall']:.2f}** | "
              f"**{fb['OVERALL']['convention_aligned_recall']:.2f}** |")

    md += ["\n## T4 — Flag rates\n",
           f"pairs flagged: {flags['pairs_flagged']} ({100*flags['rate']:.1f}%)\n",
           "| Flag | Count | Rate |", "|---|---|---|"]
    for k, v in flags["by_type"].items():
        md.append(f"| {k} | {v['count']} | {100*v['rate']:.1f}% |")
    Path(tables / "rq1_tables.md").write_text("\n".join(md), encoding="utf-8")

    print("\n" + md_recall_table(ours, rnd, majority, boxonly, major_name))
    print(f"\nreport -> {args.out}/fidelity_report.json")
    print(f"tables -> {tables/'rq1_tables.md'}")


if __name__ == "__main__":
    main()
