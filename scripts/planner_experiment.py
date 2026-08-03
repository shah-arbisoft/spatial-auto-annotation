"""Build the LLM-planner experiment: do spatial relations improve robot plans?

The source paper motivates its dataset with a planning demo: an LLM planner
fails to clear a cube before grasping the book beneath it until scene-graph
relations are supplied. This experiment turns that demo into a measurement,
and asks the question that matters for this project: do the AUTOMATIC
relations drive the planner as well as the human ones?

For ~25 held-out test scenes (groups 6-8) it generates a manipulation task
that genuinely requires spatial knowledge, then writes one prompt per
condition:

    A  objects only (no relations)          - the paper's failure case
    B  objects + human-labelled relations   - the manual gold standard
    C  objects + this tool's relations      - the automatic labels

Relations in B and C are filtered identically to a task-relevant subgraph
(support relations among mentioned objects, plus 1-hop relations of the
target, inverse duplicates removed) so the two conditions differ only in
label SOURCE, never in prompt size.

Outputs (outputs/planner/):
    prompts.jsonl            one prompt per scene x condition
    scoring_sheet_blind.csv  shuffled plan slots with a fixed checklist
    _key_do_not_share.csv    scene/condition per slot (for scoring later)
    README.txt               how to run the prompts and fill the sheet

    python scripts/planner_experiment.py --n 25 --seed 7
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import SpatialDataset
from src.pipeline import load_config

CANON = {"under": "on", "to the right of": "to the left of",
         "behind": "in front of"}


def canonical(s, k, o):
    """One direction per pair: map under/right/behind onto their inverses,
    and order the symmetric `near` so each pair appears once."""
    if k in CANON:
        return (o, CANON[k], s)
    if k == "near" and s > o:
        return (o, k, s)
    return (s, k, o)


def load_auto(pairs_csv):
    auto = defaultdict(list)
    with open(pairs_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["pred"]:
                for k in r["pred"].split(";"):
                    auto[r["image_id"]].append((int(r["subj"]), k, int(r["obj"])))
    return auto


def filter_relations(rels, target, stack_members):
    """Task-relevant subgraph: all support relations among mentioned objects
    plus 1-hop relations of the target; canonical direction; deduplicated."""
    keep = set()
    focus = {target} | set(stack_members)
    for s, k, o in rels:
        c = canonical(s, k, o)
        if c[1] == "on" and (s in focus or o in focus):
            keep.add(c)
        elif target in (s, o):
            keep.add(c)
    return sorted(keep, key=lambda t: (t[1] != "on", t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pairs", default="outputs/pairs.csv")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="outputs/planner")
    ap.add_argument("--vlm-replies", default=None, dest="vlm_replies",
                    help="a VLM replies file; adds condition D, the same "
                         "prompt built from that model's relations")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cfg = load_config(args.config)
    ds = SpatialDataset(cfg["dataset"]["root"])
    auto = load_auto(args.pairs)
    vlm = None
    if args.vlm_replies:
        vlm = defaultdict(list)
        for line in Path(args.vlm_replies).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            for rel in r.get("relations", []):
                s, pred, o = rel      # stored as [subject, predicate, object]
                vlm[r["image_id"]].append((int(s), pred, int(o)))

    # candidate scenes: held-out groups, with a gold stack (task needs the
    # relation) and at least 4 objects (room for distractors)
    candidates = []
    for gt in ds:
        if gt.image_id.split("/")[0] not in ("group_6", "group_7", "group_8"):
            continue
        if len(gt.objects) < 4 or gt.image_id not in auto:
            continue
        stacks = [(r.subject, r.object) for r in gt.relations if r.predicate == "on"]
        stacks += [(r.object, r.subject) for r in gt.relations if r.predicate == "under"]
        if stacks:
            candidates.append((gt, stacks))
    rng.shuffle(candidates)
    chosen = candidates[: args.n]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts, slots = [], []

    for scene_no, (gt, stacks) in enumerate(chosen, 1):
        names = {i: f"{o.label}{i}" for i, o in enumerate(gt.objects)}
        top, bottom = rng.choice(stacks)          # top rests on bottom
        target = bottom                           # grasping it requires clearing top
        task = (f"Pick up {names[target]} and hand it to me. "
                f"Do not knock anything over.")
        objects_line = ", ".join(f"{names[i]} (a {o.label})"
                                 for i, o in enumerate(gt.objects))

        human_rels = [(r.subject, r.predicate, r.object) for r in gt.relations]
        auto_rels = auto[gt.image_id]
        rel_sets = {
            "A": None,
            "B": filter_relations(human_rels, target, [top, bottom]),
            "C": filter_relations(auto_rels, target, [top, bottom]),
        }
        # Condition D is built by the identical filter from a vision-language
        # model's relations, so B, C and D differ only in who supplied them.
        if vlm is not None:
            rel_sets["D"] = filter_relations(vlm.get(gt.image_id, []),
                                             target, [top, bottom])

        for cond, rels in rel_sets.items():
            lines = [
                "You control a robot arm working at a table in a lab.",
                f"Visible objects: {objects_line}.",
            ]
            if rels is not None:
                rel_txt = "; ".join(f"{names[s]} is {k} {names[o]}"
                                    for s, k, o in rels)
                lines.append(f"Known spatial relationships: {rel_txt}.")
            lines += [
                f"Task: {task}",
                "Give the minimal numbered list of steps to complete the task "
                "safely. Mention every object you move.",
            ]
            prompts.append({"scene": scene_no, "image_id": gt.image_id,
                            "condition": cond, "target": names[target],
                            "must_clear": names[top],
                            "prompt": "\n".join(lines)})
            slots.append({"scene": scene_no, "condition": cond})

    rng.shuffle(slots)
    with open(out / "prompts.jsonl", "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")
    with open(out / "scoring_sheet_blind.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "slot", "plan_pasted_from_prompt_id",
            "clears the object on top before grasping (y/n)",
            "grasps the correct object (y/n)",
            "no invented objects or steps (y/n)", "notes"])
        w.writeheader()
        for i, s in enumerate(slots, 1):
            w.writerow({"slot": i, "plan_pasted_from_prompt_id":
                        f"scene{s['scene']}-{s['condition']}"})
    with open(out / "_key_do_not_share.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["slot", "scene", "condition"])
        w.writeheader()
        for i, s in enumerate(slots, 1):
            w.writerow({"slot": i, **s})

    (out / "README.txt").write_text(f"""Planner experiment - how to run

1. For each entry in prompts.jsonl, send the "prompt" text to the LLM
   (Gemini API or chat window; temperature 0 if you can set it). {len(prompts)}
   prompts in total: {len(chosen)} scenes x 3 conditions.
2. Save each reply. With an API key this is a 20-line loop; ask for the
   snippet if wanted.
3. Score the plans in scoring_sheet_blind.csv IN SLOT ORDER (the order is
   shuffled so conditions cannot be guessed). Three y/n columns per plan:
   - clears the top object before grasping the target
   - grasps the right object
   - invents nothing (no objects or constraints that are not in the prompt)
4. Return the filled sheet; scoring by condition is automatic from the key.

The comparison that matters: condition C (our labels) vs B (human labels),
with A showing what happens with no relations at all.
""", encoding="utf-8")

    print(f"scenes: {len(chosen)}  prompts: {len(prompts)} -> {out}")
    print("sample prompt (condition C):")
    print("-" * 60)
    print(next(p["prompt"] for p in prompts if p["condition"] == "C"))


if __name__ == "__main__":
    main()
