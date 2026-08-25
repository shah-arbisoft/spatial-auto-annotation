"""Check dissertation tables against the tables the runs generate.

Two tables have now gone stale in this project: Table 6.1, rebuilt after it
turned out to come from a superseded training session, and Table 4.3, which
predated the ground-plane fallback and disagreed with the fidelity run in
twenty-one of its forty-five cells while the sentence above it claimed the
fallback was included. Both were copied into the Markdown by hand and never
refreshed when the run changed.

This pairs each generated table in outputs/tables/ with the dissertation
table carrying the same header row, and reports any cell that differs.

    python scripts/check_tables_fresh.py

Exits non-zero if a table has drifted. Tables with no generated counterpart
are listed as unchecked rather than passed, so the report never implies more
coverage than it has.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISS = ROOT / "dissertation"
GEN = ROOT / "outputs" / "tables"


def tables(text: str):
    """Every pipe table in a Markdown file, as lists of stripped cells."""
    out, cur = [], []
    for line in text.splitlines():
        if line.lstrip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(set(c) <= set("-: ") for c in cells):
                cur.append(cells)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def norm(cell: str) -> str:
    """Compare on the value, not the emphasis the chapter adds."""
    return re.sub(r"[*`]", "", cell).strip()


def key(rows) -> str:
    return " | ".join(norm(c) for c in rows[0])


# Tables in outputs/tables/ are written by generators that read stored
# results. If a result is re-measured and its generator is not re-run, the
# table keeps numbers the data no longer supports, and anything the
# dissertation copied out of it goes stale silently. That is how the Gemini
# comparison in E.1 came to carry pre-refit pipeline columns for a fortnight.
# A table with no entry here is not age-checked.
SOURCES = {
    "vlm_models.md": ["outputs/vlm_pilot/scores.json",
                      "outputs/vlm_pilot/scores_pro.json"],
    "rq2.md": ["outputs/rq2_report.json"],
    "rq2_vlm.md": ["outputs/rq2_report_vlm.json"],
    "rq1_tables.md": ["outputs/fidelity_report.json"],
    "uncertainty.md": ["outputs/uncertainty.json"],
    "annotator_agreement.md": ["outputs/annotator_agreement.json"],
    "depth_ablation.md": ["outputs/depth_ablation.json"],
    "crowd_validation.md": ["outputs/crowd_validation.json"],
    "seed_replication.md": ["outputs/sgg_benchmark/seed_replication.json"],
}


def check_ages() -> int:
    """Report generated tables older than the results they are built from."""
    stale = 0
    for name, srcs in sorted(SOURCES.items()):
        tbl = GEN / name
        if not tbl.exists():
            continue
        for s in srcs:
            sp = ROOT / s
            if sp.exists() and sp.stat().st_mtime > tbl.stat().st_mtime:
                stale += 1
                print(f"  STALE {name} is older than {s}; re-run its "
                      "generator before trusting anything copied from it")
    print(f"  {len(SOURCES)} generated table(s) age-checked, "
          f"{stale} older than their source data")
    return stale


def main() -> int:
    generated = {}
    for f in sorted(GEN.glob("*.md")):
        for t in tables(f.read_text(encoding="utf-8")):
            if len(t) > 1:
                generated.setdefault(key(t), (f.name, t))

    drift = checked = 0
    unchecked = []
    for f in sorted(DISS.glob("*.md")):
        for t in tables(f.read_text(encoding="utf-8")):
            if len(t) < 2:
                continue
            k = key(t)
            if k not in generated:
                unchecked.append((f.name, k[:58]))
                continue
            checked += 1
            src, g = generated[k]
            gmap = {norm(r[0]): r for r in g[1:]}
            for row in t[1:]:
                label = norm(row[0])
                if label not in gmap:
                    continue
                for i, cell in enumerate(row):
                    if i >= len(gmap[label]):
                        continue
                    a, b = norm(cell), norm(gmap[label][i])
                    if a != b:
                        drift += 1
                        print(f"  DRIFT {f.name} :: {label} col {i}: "
                              f"text {a!r} vs {src} {b!r}")

    aged = check_ages()
    print(f"\n  {checked} table(s) checked against outputs/tables/, "
          f"{drift} cell(s) drifted")
    if unchecked:
        print(f"  {len(unchecked)} table(s) have no generated counterpart "
              "and were not checked:")
        for name, k in unchecked[:8]:
            print(f"     {name}: {k}")
    return 1 if (drift or aged) else 0


if __name__ == "__main__":
    raise SystemExit(main())
