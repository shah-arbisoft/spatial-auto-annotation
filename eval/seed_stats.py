"""Aggregate the benchmark arms across seeds.

The Week-7 experiment ran one seed per arm, so chapter 6 reported its margins
as observed rather than tested. The replication adds seeds 43 and 44, giving
three runs per arm, and this script reports mean and spread for each metric
and each test slice.

Two input shapes are accepted:

1. `outputs/sgg_benchmark/reeval_results.json` - written by the
   re-evaluation notebook, one record per (run, slice), covering the full
   test set and each annotator group. Preferred, because its zero-shot
   numbers use one fixed reference set for both arms.
2. `outputs/sgg_benchmark/seeds/**/eval_results_top_100.json` - the raw
   per-run evaluation folders. Pooled metrics only.

Seed 42 is read from `outputs/sgg_benchmark/test_results.json`.

A note on zero-shot recall: it is scored against the triplet types seen in
whatever is staged as the training split at evaluation time. The first
replication run left the auto arm's labels staged, so every arm was scored
against its 213 seen types instead of the human arm's 94, and all four
zero-shot numbers collapsed to zero. Those values are ignored here unless
the re-evaluation output is present; R/mR/F1 never consult training
statistics and are always used.

    python eval/seed_stats.py
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("outputs/sgg_benchmark")
REEVAL = ROOT / "reeval_results.json"
SEED_DIR = ROOT / "seeds"
OUT_MD = Path("outputs/tables/seed_replication.md")
OUT_JSON = ROOT / "seed_replication.json"
METRICS = ["R@100", "mR@100", "F1@100", "zR@100"]


def load_original():
    """Seed 42, already parsed into test_results.json (pooled + per-group)."""
    p = ROOT / "test_results.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for arm in ("human", "auto"):
        if arm not in d:
            continue
        rows.append({"arm": arm, "seed": 42, "slice": "full",
                     **{m: d[arm].get(m) for m in METRICS if m in d[arm]}})
    # per-group sharpeners, if the original recorded them
    sharp = d.get("sharpeners", {})
    for key, val in sharp.items() if isinstance(sharp, dict) else []:
        g = re.search(r"group[_ ]?(\d)", str(key))
        if not g or not isinstance(val, dict):
            continue
        for arm in ("human", "auto"):
            if arm in val and isinstance(val[arm], (int, float)):
                rows.append({"arm": arm, "seed": 42, "slice": f"group_{g.group(1)}",
                             "mR@100": val[arm]})
    return rows


def load_reeval():
    if not REEVAL.exists():
        return []
    d = json.loads(REEVAL.read_text(encoding="utf-8"))
    rows = []
    for rec in d.values():
        rows.append({"arm": rec["arm"], "seed": rec["seed"], "slice": rec["slice"],
                     **{m: rec.get(m) for m in METRICS},
                     "n_zeroshot": rec.get("n_zeroshot")})
    return rows


def load_raw_seed_folders():
    """Fallback: the first replication's per-run folders (pooled, no valid zR)."""
    rows = []
    for f in sorted(SEED_DIR.rglob("eval_results_top_100.json")):
        m = re.search(r"eval_react_(human|auto)_s(\d+)", str(f))
        if not m:
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        rec = {"arm": m.group(1), "seed": int(m.group(2)), "slice": "full"}
        if d.get("sgdet_recall", {}).get("100"):
            rec["R@100"] = statistics.mean(d["sgdet_recall"]["100"])
        rec["mR@100"] = d.get("sgdet_mean_recall", {}).get("100")
        rec["F1@100"] = d.get("sgdet_f1_score", {}).get("100")
        # zR deliberately omitted: staged against the wrong reference set
        rows.append(rec)
    return rows


def summarise(rows, slice_name, metric):
    out = {}
    for arm in ("human", "auto"):
        xs = [r[metric] for r in rows
              if r["arm"] == arm and r["slice"] == slice_name
              and isinstance(r.get(metric), (int, float))]
        if xs:
            out[arm] = {"mean": statistics.mean(xs), "n": len(xs),
                        "min": min(xs), "max": max(xs),
                        "sd": statistics.pstdev(xs) if len(xs) > 1 else 0.0}
    return out


def main():
    rows = load_original()
    reeval = load_reeval()
    if reeval:
        # re-evaluation supersedes the raw folders for any (arm, seed, slice)
        seen = {(r["arm"], r["seed"], r["slice"]) for r in reeval}
        rows = [r for r in rows if (r["arm"], r["seed"], r["slice"]) not in seen] + reeval
        source = "re-evaluation (fixed zero-shot reference, per-group slices)"
    else:
        raw = load_raw_seed_folders()
        seen = {(r["arm"], r["seed"], r["slice"]) for r in raw}
        rows = [r for r in rows if (r["arm"], r["seed"], r["slice"]) not in seen] + raw
        source = "first replication (pooled only; zero-shot omitted, see docstring)"

    if not rows:
        sys.exit("no results found under outputs/sgg_benchmark/")

    slices = sorted({r["slice"] for r in rows},
                    key=lambda s: (s != "full", s))
    seeds = sorted({r["seed"] for r in rows})
    print(f"source: {source}\nseeds: {seeds}   slices: {slices}\n")

    md = ["# Benchmark replication across seeds\n",
          f"Seeds {seeds}. Identical data, split and frozen detector in every "
          "run, so the spread is the relation model's own training variance. "
          f"Source: {source}.\n"]
    summary = defaultdict(dict)

    for sl in slices:
        avail = [m for m in METRICS
                 if summarise(rows, sl, m).get("human") or summarise(rows, sl, m).get("auto")]
        if not avail:
            continue
        md += [f"## {sl}\n",
               "| metric | human-trained | auto-trained | ranges overlap |",
               "|---|---|---|---|"]
        for m in avail:
            s = summarise(rows, sl, m)
            summary[sl][m] = s
            if len(s) < 2:
                continue
            h, a = s["human"], s["auto"]
            ov = not (h["min"] > a["max"] or a["min"] > h["max"])
            md.append(
                f"| {m} | {h['mean']:.3f} ({h['min']:.3f}-{h['max']:.3f}, n={h['n']}) "
                f"| {a['mean']:.3f} ({a['min']:.3f}-{a['max']:.3f}, n={a['n']}) "
                f"| {'yes' if ov else 'no'} |")
        md.append("")

    # the two claims the replication exists to test
    notes = []
    full_mr = summary.get("full", {}).get("mR@100", {})
    if len(full_mr) == 2:
        h, a = full_mr["human"], full_mr["auto"]
        ov = not (h["min"] > a["max"] or a["min"] > h["max"])
        notes.append(
            f"Pooled mR@100: human {h['mean']:.3f} vs auto {a['mean']:.3f}; per-seed "
            f"ranges {'overlap' if ov else 'do not overlap'}, so the human arm's "
            f"headline advantage is {'within' if ov else 'larger than'} run-to-run "
            f"variation at n={h['n']} seeds per arm.")
    g7 = summary.get("group_7", {}).get("mR@100", {})
    if len(g7) == 2:
        h, a = g7["human"], g7["auto"]
        ov = not (h["min"] > a["max"] or a["min"] > h["max"])
        who = "auto" if a["mean"] > h["mean"] else "human"
        notes.append(
            f"Group 7 (the one test annotator with no measured convention defect): "
            f"human {h['mean']:.3f} vs auto {a['mean']:.3f}. The {who} arm leads on "
            f"the mean and the per-seed ranges {'overlap' if ov else 'do not overlap'}, "
            f"so the margin is {'not separable from' if ov else 'larger than'} "
            f"seed variance.")
    if notes:
        md += ["## What the replication settles\n"] + [f"- {n}" for n in notes]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(
        {"source": source, "per_run": rows, "summary": summary}, indent=2),
        encoding="utf-8")
    print("\n".join(md))
    print(f"\nreport -> {OUT_JSON} ; table -> {OUT_MD}")


if __name__ == "__main__":
    main()
