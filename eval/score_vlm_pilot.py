"""Score the VLM pilot with the RQ1 battery, beside the geometric pipeline.

Both label sources are scored on the *same* pairs of the *same* images
against the *same* human gold, so the only difference is what produced the
label. Recall of human triplets is the primary metric, matching §4.1.

The output answers the question the pilot was run to settle. If the VLM
matches or beats the pipeline it earns a fourth arm; if it is close on some
predicates and poor on others it earns an adjudicator role on those; if it
trails everywhere it earns a footnote.

    python eval/score_vlm_pilot.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PREDICATES = ["on", "under", "to the left of", "to the right of",
              "in front of", "behind", "near"]
INVERTED = {"group_6", "group_8"}


def load_pairs(path: str) -> dict:
    """{image_id: {(s, o): (gold_set, pipeline_set)}} from pairs.csv."""
    out: dict = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[r["image_id"]][(int(r["subj"]), int(r["obj"]))] = (
                {p for p in r["gold"].split(";") if p},
                {p for p in r["pred"].split(";") if p})
    return dict(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replies", default="outputs/vlm_pilot/replies.jsonl")
    ap.add_argument("--pairs", default="outputs/pairs.csv")
    ap.add_argument("--out", default="outputs/vlm_pilot/scores.json")
    args = ap.parse_args()

    rp = Path(args.replies)
    if not rp.exists():
        sys.exit(f"No replies at {rp}. Run scripts/run_vlm_pilot.py first.")
    replies = [json.loads(l) for l in
               rp.read_text(encoding="utf-8").splitlines() if l.strip()]
    pairs = load_pairs(args.pairs)

    gold_n = defaultdict(int)
    vlm_hit = defaultdict(int)
    pipe_hit = defaultdict(int)
    vlm_emitted = defaultdict(int)
    pipe_emitted = defaultdict(int)
    scored_images = 0
    skipped = []
    per_group = defaultdict(lambda: {"gold": 0, "vlm": 0, "pipe": 0})

    for r in replies:
        image_id = r["image_id"]
        if image_id not in pairs:
            skipped.append(image_id)
            continue
        scored_images += 1
        group = image_id.split("/")[0]
        vlm = defaultdict(set)
        for s, p, o in r["relations"]:
            vlm[(s, o)].add(p)

        for (s, o), (gold, pipe) in pairs[image_id].items():
            v = vlm.get((s, o), set())
            for p in PREDICATES:
                if p in v:
                    vlm_emitted[p] += 1
                if p in pipe:
                    pipe_emitted[p] += 1
                if p in gold:
                    gold_n[p] += 1
                    per_group[group]["gold"] += 1
                    if p in v:
                        vlm_hit[p] += 1
                        per_group[group]["vlm"] += 1
                    if p in pipe:
                        pipe_hit[p] += 1
                        per_group[group]["pipe"] += 1

    def rec(hit):
        return {p: (hit[p] / gold_n[p] if gold_n[p] else None)
                for p in PREDICATES}

    v_rec, p_rec = rec(vlm_hit), rec(pipe_hit)
    mean = lambda d: (sum(x for x in d.values() if x is not None)
                      / max(1, sum(1 for x in d.values() if x is not None)))

    print(f"{scored_images} images scored"
          + (f" ({len(skipped)} had no pairs record)" if skipped else ""))
    print(f"\n{'predicate':16s} {'gold':>6s} {'VLM':>8s} {'pipeline':>10s} "
          f"{'VLM emits':>10s} {'pipe emits':>11s}")
    for p in PREDICATES:
        f = lambda v: f"{v:.3f}" if v is not None else "    -"
        print(f"{p:16s} {gold_n[p]:6d} {f(v_rec[p]):>8s} {f(p_rec[p]):>10s} "
              f"{vlm_emitted[p]:10d} {pipe_emitted[p]:11d}")
    print(f"{'MEAN':16s} {sum(gold_n.values()):6d} {mean(v_rec):8.3f} "
          f"{mean(p_rec):10.3f}")

    print(f"\n{'group':10s} {'gold':>6s} {'VLM':>8s} {'pipeline':>10s}"
          "   (groups 6 and 8 use the inverted front/behind convention)")
    for g in sorted(per_group):
        d = per_group[g]
        tag = "  inverted" if g in INVERTED else ""
        print(f"{g:10s} {d['gold']:6d} {d['vlm'] / max(1, d['gold']):8.3f} "
              f"{d['pipe'] / max(1, d['gold']):10.3f}{tag}")

    bad = sum(len(r.get("problems", [])) for r in replies)
    print(f"\nmalformed records dropped from the VLM output: {bad}")
    print("\nverdict guide: a fourth arm needs the VLM to match or beat the")
    print("pipeline overall; an adjudicator role needs it to win on specific")
    print("predicates; otherwise it is a footnote.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "images": scored_images,
        "gold": dict(gold_n),
        "vlm_recall": v_rec,
        "pipeline_recall": p_rec,
        "vlm_mean": mean(v_rec),
        "pipeline_mean": mean(p_rec),
        "vlm_emitted": dict(vlm_emitted),
        "pipeline_emitted": dict(pipe_emitted),
        "per_group": {k: dict(v) for k, v in per_group.items()},
        "malformed_dropped": bad,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
