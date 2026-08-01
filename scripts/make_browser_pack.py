"""Prepare the §4.16 comparison to be run by hand in a browser.

The API arm uses `gemini-flash-latest` at temperature 0, which is
reproducible but not the strongest model available. Running the same 30
images through a consumer chat interface tests a stronger model, at the cost
of experimental control: the version behind the product is not pinned, the
sampling temperature is not settable, and the interface may resize the image
before the model sees it. That is a fair trade as long as the two arms are
reported separately, which is why this writes its own replies file rather
than adding to the API run's.

Produces a folder holding every image, one file with all the prompts, and a
template to paste the answers into. `ingest_browser_replies.py` turns the
filled template back into the same JSONL the scorer already reads.

    python scripts/make_browser_pack.py
    # ... work through PROMPTS.md in the browser, paste into REPLIES.md ...
    python scripts/ingest_browser_replies.py
    python eval/score_vlm_pilot.py --replies outputs/vlm_pilot/replies_browser.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PILOT = ROOT / "outputs" / "vlm_pilot"
OUT = PILOT / "browser_pack"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0,
                    help="limit to the first N images (0 = all)")
    args = ap.parse_args()

    prompts = [json.loads(l) for l in
               (PILOT / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
               if l.strip()]
    if args.n:
        prompts = prompts[:args.n]

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "img").mkdir(parents=True)

    guide = [
        "# Browser arm: how to run it",
        "",
        f"{len(prompts)} images. Budget about a minute each.",
        "",
        "**Start a new chat for every image.** Carrying one conversation",
        "across several scenes lets the model see earlier answers, and it",
        "will copy their shape, which would make the run measure its own",
        "consistency instead of its accuracy.",
        "",
        "For each numbered section in `PROMPTS.md`:",
        "",
        "1. Open a new chat.",
        "2. Attach the image named in the section, from `img/`.",
        "3. Paste the whole prompt block. Do not shorten it: the operational",
        "   definitions are what make this comparable to the other arms.",
        "4. Copy the model's reply.",
        "5. Paste it into `REPLIES.md` under the matching heading, between",
        "   the fences that are already there.",
        "",
        "Paste the reply exactly as given, fences and prose included. The",
        "parser strips those and reports anything it cannot read rather than",
        "guessing, so a messy paste is safe but a silently edited one is not.",
        "",
        "Stopping early is fine. Whatever is filled in gets scored, and the",
        "count is reported with the result.",
        "",
        "Record which model you used at the top of `REPLIES.md`, since the",
        "arm is only meaningful if the write-up can name it.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(guide), encoding="utf-8")

    plines = ["# Prompts, one per image", ""]
    rlines = [
        "# Replies",
        "",
        "model used: _______________________   (e.g. Gemini 2.5 Pro)",
        "date: _______________",
        "",
        "Paste each reply between the fences under its heading.",
        "",
    ]
    for i, p in enumerate(prompts, 1):
        src = ROOT / p["image"]
        shutil.copy(src, OUT / "img" / src.name)
        plines += [
            f"## {i}. {p['image_id']}",
            "",
            f"Attach: `img/{src.name}`  ({p['n_objects']} objects)",
            "",
            "```",
            p["prompt"],
            "```",
            "",
        ]
        rlines += [f"### {p['image_id']}", "", "```json", "", "```", ""]

    (OUT / "PROMPTS.md").write_text("\n".join(plines), encoding="utf-8")
    (OUT / "REPLIES.md").write_text("\n".join(rlines), encoding="utf-8")

    print(f"{len(prompts)} images -> {OUT}")
    print(f"  README.md   how to run it")
    print(f"  PROMPTS.md  the prompts, in order")
    print(f"  REPLIES.md  paste answers here")
    print(f"  img/        the images to attach")


if __name__ == "__main__":
    main()
