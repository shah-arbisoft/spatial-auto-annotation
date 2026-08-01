"""Build a pack for checking the §4.16 result by hand, in a browser.

The pilot's verdict rests on a script comparing three sets of triplets. That
is easy to accept and easy to get wrong, so this writes out everything
needed to redo the comparison manually: the exact image the model saw, the
exact prompt it was given, what the human annotators recorded, what the
pipeline computed, and what the model actually replied in the recorded run.

Paste the prompt and upload the image at aistudio.google.com, then read your
own answer against the three columns. A reader who suspects the scoring
script can check it without running any code.

    python scripts/vlm_manual_check.py            # 3 images: worst, middle, best
    python scripts/vlm_manual_check.py --all      # all 30
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import SpatialDataset
from src.pipeline import load_config

PILOT = ROOT / "outputs" / "vlm_pilot"
OUT = PILOT / "manual_check"
PREDICATES = ["on", "under", "to the left of", "to the right of",
              "in front of", "behind", "near"]


def load_pairs(path):
    out = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[r["image_id"]][(int(r["subj"]), int(r["obj"]))] = (
                {p for p in r["gold"].split(";") if p},
                {p for p in r["pred"].split(";") if p})
    return out


def triplets(names, mapping, wanted=None):
    """Readable 'book2 is on box1' lines for a {(s,o): set} mapping."""
    out = []
    for (s, o), preds in sorted(mapping.items()):
        for p in sorted(preds):
            if wanted is None or p in wanted:
                out.append(f"{names[s]} is {p} {names[o]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    replies = {r["image_id"]: r for r in
               (json.loads(l) for l in
                (PILOT / "replies.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip())}
    prompts = {r["image_id"]: r for r in
               (json.loads(l) for l in
                (PILOT / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip())}
    pairs = load_pairs(ROOT / "outputs" / "pairs.csv")
    cfg = load_config(args.config)
    ds = {im.image_id: im for im in SpatialDataset(cfg["dataset"]["root"])}

    # rank by how well the model did, so the pack spans the range rather than
    # showing only the flattering cases
    scored = []
    for iid, r in replies.items():
        if iid not in pairs:
            continue
        v = defaultdict(set)
        for s, p, o in r["relations"]:
            v[(s, o)].add(p)
        g = h = 0
        for k, (gold, _pipe) in pairs[iid].items():
            for p in gold:
                g += 1
                h += p in v.get(k, set())
        if g >= 4:
            scored.append((h / g, iid))
    scored.sort()

    if args.all:
        chosen = [i for _s, i in scored]
    else:
        n = min(args.n, len(scored))
        idx = [round(k * (len(scored) - 1) / max(1, n - 1)) for k in range(n)]
        chosen = [scored[i][1] for i in dict.fromkeys(idx)]

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    lines = [
        "# Check the vision-language result by hand",
        "",
        "Each section below is one image from the pilot. To redo it yourself:",
        "",
        "1. Open <https://aistudio.google.com/> and pick the same model,",
        "   `gemini-flash-latest`. Set temperature to 0 so the reply is",
        "   reproducible, as the recorded run did.",
        "2. Upload the image named in the section, from the `img/` folder",
        "   beside this file. It already has the boxes drawn and numbered,",
        "   which is what the model was given.",
        "3. Paste the prompt exactly as it appears. It carries the operational",
        "   definitions from Chapter 3, including that *in front of* means",
        "   nearer the camera, without which the answer measures a different",
        "   question.",
        "4. Compare your reply against the three lists. **Human** is what the",
        "   dataset's annotators recorded, and is the target both other",
        "   columns are scored against. **Pipeline** is this project's output.",
        "   **Model (recorded)** is what the pilot actually received.",
        "",
        "Your reply will not match the recorded one word for word even at",
        "temperature 0, since the models behind an alias change over time.",
        "What should reproduce is the shape of it: few relations emitted,",
        "silence rather than wrong answers on most missed pairs, and one",
        "direction of a left/right pair asserted without the other.",
        "",
    ]

    for iid in chosen:
        im = ds[iid]
        names = {i: f"{o.label}{i}" for i, o in enumerate(im.objects)}
        src = ROOT / prompts[iid]["image"]
        dst = OUT / "img" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)

        gold_map, pipe_map = defaultdict(set), defaultdict(set)
        for (s, o), (g, p) in pairs[iid].items():
            if g:
                gold_map[(s, o)] |= g
            if p:
                pipe_map[(s, o)] |= p
        vlm_map = defaultdict(set)
        for s, p, o in replies[iid]["relations"]:
            vlm_map[(s, o)].add(p)

        gold_t = triplets(names, gold_map)
        # the pipeline and the model both emit far more than the humans, so
        # showing only the pairs the humans judged keeps the comparison
        # readable; the full output is in the pilot's own files
        keys = set(gold_map)
        pipe_t = triplets(names, {k: v for k, v in pipe_map.items() if k in keys})
        vlm_t = triplets(names, {k: v for k, v in vlm_map.items() if k in keys})

        hit = sum(1 for t in gold_t if t in vlm_t)
        php = sum(1 for t in gold_t if t in pipe_t)

        lines += [
            f"## {iid}",
            "",
            f"Image: `img/{src.name}` ({len(im.objects)} objects)",
            "",
            f"On the {len(gold_t)} relationships the humans recorded here, "
            f"the model got {hit} and the pipeline got {php}.",
            "",
            "<details><summary>The prompt (click to expand, paste all of it)"
            "</summary>",
            "",
            "```",
            prompts[iid]["prompt"],
            "```",
            "",
            "</details>",
            "",
            "The relationships the humans recorded come first; below the rule",
            "are ones only the pipeline or the model proposed, which the human",
            "annotators did not record either way and which therefore cannot",
            "be scored (§4.3 explains why).",
            "",
            "| Relationship | Human | Pipeline | Model |",
            "|---|---|---|---|",
        ]
        mark = lambda t, s: "yes" if t in s else ""
        for t in sorted(gold_t):
            lines.append(f"| {t} | **yes** | {mark(t, pipe_t)} | "
                         f"{mark(t, vlm_t)} |")
        extra = sorted((set(pipe_t) | set(vlm_t)) - set(gold_t))
        if extra:
            lines.append("| *not recorded by the humans* | | | |")
            for t in extra:
                lines.append(f"| {t} | | {mark(t, pipe_t)} | "
                             f"{mark(t, vlm_t)} |")
        lines += ["", "Model's full reply as recorded:", "", "```",
                  replies[iid]["reply"].strip()[:2000], "```", ""]

    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(chosen)} images written to {OUT}")
    print(f"open {OUT / 'README.md'}")


if __name__ == "__main__":
    main()
