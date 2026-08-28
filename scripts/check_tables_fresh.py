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

import json
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
    # the gallery and the recall table describe the same run, so the
    # gallery must not predate the fidelity report
    "failure_gallery.md": ["outputs/fidelity_report.json"],
}


def check_recall_reconciles() -> int:
    """Table D.3's misses must equal gold x (1 - recall) from the fidelity run.

    A stale failure gallery is invisible to both the cell check and the age
    check if someone edits the Markdown by hand, but it cannot survive this:
    the two tables are derived from one run and must agree by arithmetic.
    """
    rep = ROOT / "outputs" / "fidelity_report.json"
    app = DISS / "appendices.md"
    if not (rep.exists() and app.exists()):
        return 0
    rec = json.loads(rep.read_text(encoding="utf-8"))["recall"]["ours"]
    txt = app.read_text(encoding="utf-8")
    bad = 0
    seen = 0
    for m in re.finditer(r"^\| ([a-z ]+?) \| (\d+)/(\d+) \|", txt, re.M):
        pred, miss, gold = m.group(1), int(m.group(2)), int(m.group(3))
        if pred not in rec:
            continue
        seen += 1
        want = round(rec[pred]["gold"] * (1 - rec[pred]["recall"]))
        if gold != rec[pred]["gold"] or abs(miss - want) > 1:
            bad += 1
            print(f"  MISMATCH {pred}: appendix says {miss}/{gold}, the "
                  f"fidelity run implies {want}/{rec[pred]['gold']}")
    print(f"  {seen} miss/gold row(s) checked against the recall table, "
          f"{bad} disagree")
    return bad


BENCH = ROOT / "outputs" / "sgg_benchmark" / "reeval_all_arms_085.json"
BENCH_SEEDS = (42, 43, 44)
BENCH_ARMS = {"human-trained": "human", "auto-trained": "auto",
              "vision-language": "vlm"}
BENCH_SLICES = {"full test": "full", "group 6": "group_6",
                "group 7": "group_7", "group 8": "group_8",
                "aligned": "full_aligned"}


def check_benchmark() -> int:
    """Chapter 6's two tables against the run they are drawn from.

    Neither had a generated counterpart, so the cell check above skipped both
    and the chapter's central evidence was verified by nothing. That is the
    gap the whole script exists to close, and it matters more here than
    elsewhere: outputs/tables/seed_replication.md holds the *superseded*
    on_contact_min 0.60 figures (Appendix F.9 quotes them deliberately), so
    the artefact nearest to hand contradicts the chapter by design. These
    numbers come from the shipped 0.85 re-evaluation instead.
    """
    ch6 = DISS / "chapter6_benchmark.md"
    if not (BENCH.exists() and ch6.exists()):
        return 0
    run = json.loads(BENCH.read_text(encoding="utf-8"))

    def vals(arm, sl, met):
        out = []
        for s in BENCH_SEEDS:
            row = run.get(f"react_{arm}_s{s}|{sl}")
            if row and met in row:
                out.append(row[met])
        return out

    def num(cell):
        m = re.match(r"^([0-9]*\.?[0-9]+)", norm(cell))
        return float(m.group(1)) if m else None

    bad = seen = 0
    for rows in tables(ch6.read_text(encoding="utf-8")):
        head = [norm(c) for c in rows[0]]
        # Table 6.1: one seed, one slice, metrics down the rows.
        if head[:1] == ["metric (test, sgdet)"]:
            cols = [BENCH_ARMS.get(h) for h in head[1:]]
            for r in rows[1:]:
                met = norm(r[0]).split(",")[0]
                for i, arm in enumerate(cols, start=1):
                    if not arm or i >= len(r):
                        continue
                    got, want = num(r[i]), vals(arm, "full", met)
                    if got is None or len(want) != 3:
                        continue
                    seen += 1
                    if abs(got - want[0]) > 0.0006:   # seed 42 is want[0]
                        bad += 1
                        print(f"  MISMATCH 6.1 {met} {arm}: text {got}, "
                              f"run {want[0]:.4f}")
        # 6.3.1: slice x metric down the rows, "mean (min-max)" per arm.
        elif head[:2] == ["slice", "metric"]:
            cols = [BENCH_ARMS.get(h) for h in head]
            for r in rows[1:]:
                sl = BENCH_SLICES.get(norm(r[0]))
                met = norm(r[1])
                if not sl:
                    continue
                for i, arm in enumerate(cols):
                    if not arm or i >= len(r):
                        continue
                    cell = norm(r[i]).replace("–", "-")
                    m = re.match(r"([0-9.]+)\s*\(([0-9.]+)-([0-9.]+)\)", cell)
                    v = vals(arm, sl, met)
                    if not m or len(v) != 3:
                        continue
                    seen += 1
                    want = (sum(v) / 3, min(v), max(v))
                    got = tuple(float(x) for x in m.groups())
                    if any(abs(g - w) > 0.0006 for g, w in zip(got, want)):
                        bad += 1
                        print(f"  MISMATCH 6.3.1 {sl} {met} {arm}: "
                              f"text {got}, run "
                              f"({want[0]:.4f}, {want[1]:.4f}, {want[2]:.4f})")
    print(f"  {seen} benchmark cell(s) checked against "
          f"{BENCH.name}, {bad} disagree")
    return bad


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
    aged += check_recall_reconciles()
    aged += check_benchmark()
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
