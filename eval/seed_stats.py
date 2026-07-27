"""Aggregate the benchmark arms across seeds.

The Week-7 experiment ran one seed per arm, so chapter 6 reports its margins
as observed rather than tested. The replication adds seeds 43 and 44 (see
scripts/kaggle/notebook_cells_seeds.md), giving three runs per arm. This
script reads them all and reports mean and spread, which is what turns
"the auto arm is ahead on group 7" into a claim with a stated uncertainty.

Input layout, after unzipping the Kaggle output:

    outputs/sgg_benchmark/test_results.json          seed 42 (the original run)
    outputs/sgg_benchmark/seeds/eval_react_human_s43/...
    outputs/sgg_benchmark/seeds/eval_react_auto_s43/...
    outputs/sgg_benchmark/seeds/eval_react_human_s44/...
    outputs/sgg_benchmark/seeds/eval_react_auto_s44/...

Each eval folder is searched for whatever the framework wrote (a json result
file, or the log text), so the parser is deliberately forgiving.

    python eval/seed_stats.py
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path("outputs/sgg_benchmark")
SEED_DIR = ROOT / "seeds"
OUT_MD = Path("outputs/tables/seed_replication.md")
OUT_JSON = ROOT / "seed_replication.json"
METRICS = ["R@100", "mR@100", "zR@100"]


def from_original():
    """Seed 42 numbers already parsed into test_results.json."""
    p = ROOT / "test_results.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for arm in ("human", "auto"):
        if arm in d:
            out[(arm, 42)] = {m: d[arm].get(m) for m in METRICS if m in d[arm]}
    return out


def parse_folder(folder: Path):
    """Pull R@100 / mR@100 / zR@100 out of whatever the framework left behind."""
    vals = {}
    # 1) a json result file, if the framework wrote one
    for jf in folder.rglob("*.json"):
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, dict):
            flat = json.dumps(d)
            for m in METRICS:
                key = m.replace("@", "_at_")
                for pat in (rf'"{re.escape(m)}"\s*:\s*([\d.]+)',
                            rf'"{key}"\s*:\s*([\d.]+)'):
                    hit = re.search(pat, flat)
                    if hit and m not in vals:
                        vals[m] = float(hit.group(1))
    # 2) otherwise scrape the log text
    if len(vals) < len(METRICS):
        for lf in list(folder.rglob("*.txt")) + list(folder.rglob("*.log")):
            text = lf.read_text(encoding="utf-8", errors="ignore")
            for m, pat in (("R@100", r"R @ 100:\s*([\d.]+)"),
                           ("mR@100", r"mR @ 100:\s*([\d.]+)"),
                           ("zR@100", r"zR @ 100:\s*([\d.]+)")):
                hit = re.findall(pat, text)
                if hit and m not in vals:
                    vals[m] = float(hit[-1])
    return vals


def main():
    results = from_original()
    if SEED_DIR.exists():
        for folder in sorted(SEED_DIR.glob("eval_react_*")):
            m = re.search(r"eval_react_(human|auto)_s(\d+)", folder.name)
            if not m:
                continue
            arm, seed = m.group(1), int(m.group(2))
            vals = parse_folder(folder)
            if vals:
                results[(arm, seed)] = vals
            else:
                print(f"WARNING: no metrics parsed from {folder}")
    else:
        print(f"note: {SEED_DIR} not present yet — reporting the original seed only")

    if not results:
        sys.exit("no results found; unzip the Kaggle output into outputs/sgg_benchmark/")

    seeds = sorted({s for _, s in results})
    print(f"arms found: {sorted({a for a, _ in results})}   seeds: {seeds}\n")

    summary, md = {}, [
        "# Benchmark replication across seeds\n",
        f"Seeds {seeds}; identical data, split and frozen detector in every "
        "run, so the spread is the relation model's own training variance.\n",
        "| metric | human-trained | auto-trained |", "|---|---|---|",
    ]
    for m in METRICS:
        row = {}
        for arm in ("human", "auto"):
            xs = [results[(a, s)][m] for (a, s) in results
                  if a == arm and m in results[(a, s)]]
            if xs:
                row[arm] = {"mean": statistics.mean(xs), "n": len(xs),
                            "min": min(xs), "max": max(xs),
                            "sd": statistics.pstdev(xs) if len(xs) > 1 else 0.0}
        summary[m] = row
        if len(row) == 2:
            h, a = row["human"], row["auto"]
            md.append(f"| {m} | {h['mean']:.3f} ({h['min']:.3f}-{h['max']:.3f}, n={h['n']}) "
                      f"| {a['mean']:.3f} ({a['min']:.3f}-{a['max']:.3f}, n={a['n']}) |")

    # the claim the replication exists to test
    if "mR@100" in summary and len(summary["mR@100"]) == 2:
        h, a = summary["mR@100"]["human"], summary["mR@100"]["auto"]
        overlap = not (a["min"] > h["max"] or h["min"] > a["max"])
        md += ["",
               f"Pooled mR@100: human {h['mean']:.3f} vs auto {a['mean']:.3f}. "
               f"The per-seed ranges {'overlap' if overlap else 'do not overlap'}, "
               f"so the pooled difference is "
               f"{'within run-to-run variation' if overlap else 'larger than run-to-run variation'} "
               f"at n={h['n']} seeds per arm."]
        summary["ranges_overlap"] = overlap

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(
        {"per_run": {f"{a}_s{s}": v for (a, s), v in results.items()},
         "summary": summary}, indent=2), encoding="utf-8")
    print("\n".join(md))
    print(f"\nreport -> {OUT_JSON} ; table -> {OUT_MD}")


if __name__ == "__main__":
    main()
