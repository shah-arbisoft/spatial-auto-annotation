"""Compare two vision-language runs of the §4.16 battery against each other
and against the pipeline.

The pilot's stated limit was that it ran one model at one prompt, so a larger
model was an untested explanation for the gap. This puts a second model
through the identical battery: same 30 images, same numbered boxes, same
written definitions, same scored pairs. Only the model differs.

    python eval/score_vlm_pilot.py --replies outputs/vlm_pilot/replies.jsonl \
        --out outputs/vlm_pilot/scores.json
    python eval/score_vlm_pilot.py --replies outputs/vlm_pilot/replies_pro.jsonl \
        --out outputs/vlm_pilot/scores_pro.json
    python eval/compare_vlm_models.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PREDICATES = ["on", "under", "to the left of", "to the right of",
              "in front of", "behind", "near"]


def load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"missing {p}; run eval/score_vlm_pilot.py first")
    return json.loads(p.read_text(encoding="utf-8"))


def name(scores: dict) -> str:
    return ", ".join(scores.get("models", ["?"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="outputs/vlm_pilot/scores.json")
    ap.add_argument("--b", default="outputs/vlm_pilot/scores_pro.json")
    ap.add_argument("--out", default="outputs/tables/vlm_models.md")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    na, nb = name(A), name(B)

    if A["images"] != B["images"]:
        raise SystemExit(f"different image counts ({A['images']} vs "
                         f"{B['images']}); the runs are not comparable")
    # the pipeline column is the same measurement in both files; if it moved,
    # something other than the model changed and the comparison is void
    if A["pipeline_recall"] != B["pipeline_recall"]:
        raise SystemExit("pipeline recall differs between the two score "
                         "files; the two runs were not scored against the "
                         "same pipeline output")

    md = ["# Two vision-language models on the same battery", "",
          f"Both runs: {A['images']} images, the same numbered boxes, the same "
          f"written predicate definitions, the same "
          f"{A.get('judged_pairs', '?')} human-judged pairs. Only the model "
          f"differs. The pipeline column is one measurement, identical in "
          f"both files by construction.", "",
          "## Recall of the human triplets", "",
          f"| predicate | gold | {na} | {nb} | pipeline |",
          "|---|---|---|---|---|"]
    for p in PREDICATES:
        md.append(f"| {p} | {A['gold'].get(p, 0)} | {A['vlm_recall'][p]:.3f} | "
                  f"{B['vlm_recall'][p]:.3f} | {A['pipeline_recall'][p]:.3f} |")
    md.append(f"| **mean** | {sum(A['gold'].values())} | "
              f"**{A['vlm_mean']:.3f}** | **{B['vlm_mean']:.3f}** | "
              f"**{A['pipeline_mean']:.3f}** |")

    md += ["", "## On the judged pairs, where precision is defined", "",
           f"| metric | {na} | {nb} | pipeline |", "|---|---|---|---|"]
    for label, key in (("precision (micro)", "precision"),
                       ("recall (micro)", "recall"),
                       ("F1 (micro)", "f1"),
                       ("assertions", "assertions")):
        fmt = (lambda v: f"{v:,}") if key == "assertions" else (lambda v: f"{v:.3f}")
        md.append(f"| {label} | {fmt(A['vlm_micro'][key])} | "
                  f"{fmt(B['vlm_micro'][key])} | {fmt(A['pipeline_micro'][key])} |")

    md += ["", "## Self-consistency: one direction of a symmetric pair "
               "without the other", "",
           f"| predicate | {na} | {nb} |", "|---|---|---|"]
    for p in ("to the left of", "to the right of", "in front of", "behind"):
        aa = A["asserted"].get(p, 0)
        ab = B["asserted"].get(p, 0)
        ia = A["inverse_absent"].get(p, 0)
        ib = B["inverse_absent"].get(p, 0)
        md.append(f"| {p} | {ia}/{aa} ({ia / aa:.2f}) | "
                  f"{ib}/{ab} ({ib / ab:.2f}) |" if aa and ab else
                  f"| {p} | {ia}/{aa} | {ib}/{ab} |")
    md.append(f"\nDirect contradictions (a pair given a predicate and its "
              f"opposite): {na} {A['contradictions']}, {nb} "
              f"{B['contradictions']}.")
    md.append(f"\nMalformed records dropped: {na} {A['malformed_dropped']}, "
              f"{nb} {B['malformed_dropped']}.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
