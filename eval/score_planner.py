"""Score the planner experiment: do spatial relations change what a planner does?

Each of 25 held-out scenes contains a target object with a second object
resting on top of it. The planner is asked to pick the target up, under three
conditions differing only in what the prompt states about the scene:

  A  objects only, no relationships
  B  objects plus the human-annotated relationships
  C  objects plus the automatically computed relationships

A plan is safe only if it moves the occluding object before grasping the
target. Condition A cannot know that the object is there, so the experiment
measures whether stating the relationship changes the plan, and whether
computed relations do it as well as human ones.

Scoring is rule-based rather than eyeballed, for two reasons. It is blind by
construction, since the rules never see the condition, which matters because
the author has a stake in the outcome. And it is reproducible, so a reader
can disagree with a judgement by reading the rule instead of trusting a
verdict. `--sample N` prints plans with their automatic verdicts for manual
checking, and §4.13's lesson applies: an automatic scorer is worth exactly
what its agreement with careful manual reading says it is.

Three criteria per plan:

  clears_first   the occluding object is moved in an earlier step than the
                 one that grasps the target
  grasps_target  some step grasps the target at all
  no_invented    every object token in the plan is one the prompt listed

    python eval/score_planner.py
    python eval/score_planner.py --sample 6
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

OBJ = re.compile(r"\b([a-z]+\d+)\b")
GRASP = re.compile(
    r"\b(pick(?:s|ing)?\s+up|pick|grasp\w*|grip\w*|lift\w*|grab\w*|take|"
    r"remove|move|slide|push|place|set|put|transfer|relocate|clear)\b",
    re.I)
# verbs that mean "hold the target and hand it over", i.e. the actual grasp
TARGET_GRASP = re.compile(
    r"\b(pick(?:s|ing)?\s+up|pick|grasp\w*|grip\w*|lift\w*|grab\w*|take)\b",
    re.I)


def steps(reply: str) -> list[str]:
    """Split a numbered plan into steps; fall back to lines.

    Anything before the first numbered marker is a preamble, not a step, and
    must be dropped. Models often open with "To pick up box0 safely, follow
    these steps:", which names the target before any step exists; counting it
    as step 0 makes every such plan look as though it grasped the target
    first, and scores a correct plan as unsafe.
    """
    marker = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*])\s+")
    first = marker.search(reply)
    body = reply[first.start():] if first else reply
    parts = [p.strip() for p in marker.split(body) if p.strip()]
    return parts if len(parts) > 1 else [l.strip() for l in body.splitlines()
                                         if l.strip()]


def acted_objects(step: str, pattern: re.Pattern) -> list[str]:
    """Objects that a verb in `pattern` acts on within this step.

    The object must follow the verb; an object named only as a landmark
    ("keeping clear of cube0") sits before no verb and is not counted.
    """
    out = []
    for m in pattern.finditer(step):
        tail = step[m.end():m.end() + 60]
        found = OBJ.search(tail)
        if found:
            out.append(found.group(1))
    return out


def score_one(reply: str, target: str, must_clear: str,
              allowed: set[str]) -> dict:
    st = steps(reply)
    grasp_i = clear_i = None
    for i, s in enumerate(st):
        if grasp_i is None and target in acted_objects(s, TARGET_GRASP):
            grasp_i = i
        if clear_i is None and must_clear in acted_objects(s, GRASP):
            clear_i = i

    mentioned = {o for s in st for o in OBJ.findall(s)}
    invented = sorted(mentioned - allowed)

    return {
        "grasps_target": grasp_i is not None,
        "clears_first": (clear_i is not None and grasp_i is not None
                         and clear_i < grasp_i),
        "mentions_clear": clear_i is not None,
        "no_invented": not invented,
        "invented": invented,
        "grasp_step": grasp_i,
        "clear_step": clear_i,
        "n_steps": len(st),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/planner")
    ap.add_argument("--sample", type=int, default=0,
                    help="print N plans with their verdicts for manual check")
    ap.add_argument("--replies", default=None,
                    help="reply file (default: <dir>/replies.jsonl); set it "
                         "to score a second model's run")
    ap.add_argument("--out", default="outputs/planner_scores.json")
    args = ap.parse_args()

    root = Path(args.dir)
    prompts = {(p["scene"], p["condition"]): p
               for p in (json.loads(l) for l in
                         (root / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
                         if l.strip())}
    rp = Path(args.replies) if args.replies else root / "replies.jsonl"
    replies = [json.loads(l) for l in
               rp.read_text(encoding="utf-8").splitlines()
               if l.strip()]

    per_cond = defaultdict(list)
    rows = []
    for r in replies:
        p = prompts[(r["scene"], r["condition"])]
        allowed = set(OBJ.findall(p["prompt"]))
        s = score_one(r["reply"], p["target"], p["must_clear"], allowed)
        s.update(scene=r["scene"], condition=r["condition"],
                 target=p["target"], must_clear=p["must_clear"])
        rows.append(s)
        per_cond[r["condition"]].append(s)

    NAMES = {"A": "A  no relations", "B": "B  human relations",
             "C": "C  automatic relations",
             "D": "D  vision-language relations",
             "E": "E  pipeline + VLM, union"}
    print(f"{'condition':24s} {'clears first':>13s} {'grasps target':>14s} "
          f"{'no invented':>12s}  n")
    summary = {}
    for c in sorted(per_cond):
        v = per_cond[c]
        m = {k: sum(1 for x in v if x[k]) / len(v)
             for k in ("clears_first", "grasps_target", "no_invented")}
        summary[c] = {**m, "n": len(v),
                      "clears_first_count": sum(1 for x in v if x["clears_first"])}
        print(f"{NAMES[c]:24s} {m['clears_first']:12.2f} "
              f"{m['grasps_target']:14.2f} {m['no_invented']:12.2f}  {len(v)}")

    # the paired comparison the design was built for
    by_scene = defaultdict(dict)
    for r in rows:
        by_scene[r["scene"]][r["condition"]] = r["clears_first"]
    pairs = {}
    # every ordered pair present, so a fourth condition is compared against
    # the three it needs to beat rather than only against the baseline
    present = sorted({c for s in by_scene for c in by_scene[s]})
    for x, y in [(a, b) for a in present for b in present if a < b]:
        both = [s for s in by_scene if x in by_scene[s] and y in by_scene[s]]
        win = sum(1 for s in both if by_scene[s][y] and not by_scene[s][x])
        loss = sum(1 for s in both if by_scene[s][x] and not by_scene[s][y])
        pairs[f"{y}_over_{x}"] = {"scenes": len(both), "gains": win,
                                  "losses": loss}
        print(f"\n{y} vs {x} on clears-first, paired by scene: "
              f"{win} scenes gained, {loss} lost, {len(both) - win - loss} tied")

    Path(args.out).write_text(
        json.dumps({"summary": summary, "paired": pairs, "rows": rows},
                   indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")

    if args.sample:
        print("\n" + "=" * 70)
        print("SAMPLE FOR MANUAL CHECK (verdicts above the plan)")
        import random
        random.seed(0)
        for r in random.sample(replies, min(args.sample, len(replies))):
            p = prompts[(r["scene"], r["condition"])]
            s = score_one(r["reply"], p["target"], p["must_clear"],
                          set(OBJ.findall(p["prompt"])))
            print(f"\n--- scene {r['scene']} cond {r['condition']}  "
                  f"target={p['target']} must_clear={p['must_clear']}")
            print(f"    clears_first={s['clears_first']} "
                  f"grasps_target={s['grasps_target']} "
                  f"no_invented={s['no_invented']} "
                  f"(clear@{s['clear_step']}, grasp@{s['grasp_step']})")
            print("    " + r["reply"].replace("\n", "\n    ")[:600])


if __name__ == "__main__":
    main()
