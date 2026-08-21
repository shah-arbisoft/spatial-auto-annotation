"""Exact paired tests for the planner experiment (§5.7).

Twenty-five scenes is a small sample and the obvious objection is that nothing
can be concluded from it. That objection assumes independent groups. The design
is paired: every condition is put to the *same* 25 scenes, so the evidence is
carried by the scenes where two conditions disagree, and when that disagreement
runs entirely one way a handful of scenes is enough.

This computes, for each pair of conditions, the exact McNemar sign test on the
discordant scenes, plus a Clopper-Pearson interval for each condition's own
rate. It reports which comparisons 25 scenes can settle and which it cannot,
so the chapter can say so with a number instead of a hedge.

    python eval/planner_paired_tests.py
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "planner_scores_abcde.json"
OUT = ROOT / "outputs" / "planner_paired_tests.json"

NAMES = {"A": "objects only", "B": "human relations", "C": "pipeline relations",
         "D": "vision-language relations", "E": "pipeline + vision-language"}

PAIRS = [("B", "A"), ("C", "A"), ("E", "A"),
         ("B", "C"), ("E", "C"), ("E", "D"), ("D", "C")]


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    """Exact binomial interval, computed by bisection on the tail sums."""
    def tail_le(p, k, n):                     # P(X <= k)
        return sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k + 1))

    def tail_ge(p, k, n):                     # P(X >= k)
        return sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))

    lo, hi = 0.0, 1.0
    if k > 0:
        a, b = 0.0, 1.0
        for _ in range(200):
            m = (a + b) / 2
            if tail_ge(m, k, n) < alpha / 2: a = m
            else: b = m
        lo = (a + b) / 2
    if k < n:
        a, b = 0.0, 1.0
        for _ in range(200):
            m = (a + b) / 2
            if tail_le(m, k, n) < alpha / 2: b = m
            else: a = m
        hi = (a + b) / 2
    return lo, hi


def mcnemar_exact(only_x: int, only_y: int) -> float:
    """Two-sided exact sign test over the discordant scenes."""
    n = only_x + only_y
    if n == 0:
        return 1.0
    p = 2 * sum(comb(n, i) for i in range(min(only_x, only_y) + 1)) / 2**n
    return min(1.0, p)


def main() -> int:
    rows = json.loads(SRC.read_text())["rows"]
    ok = {}
    for r in rows:
        ok.setdefault(r["condition"], set())
        if r["clears_first"]:
            ok[r["condition"]].add(r["scene"])
    n = len({r["scene"] for r in rows if r["condition"] == "A"})

    out = {"n_scenes": n, "rates": {}, "pairs": {}}
    print(f"Planner conditions over {n} paired scenes\n")
    for c in sorted(ok):
        k = len(ok[c])
        lo, hi = clopper_pearson(k, n)
        out["rates"][c] = {"k": k, "n": n, "lo": lo, "hi": hi, "name": NAMES.get(c, c)}
        print(f"  {c} ({NAMES.get(c,c):28}) {k:2}/{n}  95% [{lo:.2f}, {hi:.2f}]")

    print("\nExact McNemar over discordant scenes:")
    for x, y in PAIRS:
        bx, by = len(ok[x] - ok[y]), len(ok[y] - ok[x])
        p = mcnemar_exact(bx, by)
        out["pairs"][f"{x}_vs_{y}"] = {"only_" + x: bx, "only_" + y: by,
                                       "discordant": bx + by, "p": p,
                                       "separates": bool(p < 0.05)}
        mark = "separates    " if p < 0.05 else "cannot settle"
        print(f"  {x} vs {y}: {bx} / {by} discordant  p = {p:<10.3g} {mark}")

    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nreport -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
