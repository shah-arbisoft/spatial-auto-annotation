"""Turn the hand-filled REPLIES.md into the JSONL the scorer reads.

Kept deliberately forgiving about formatting and strict about content: a
reply pasted with its fences, its prose preamble or its trailing commentary
parses fine, but a record naming an object index the image does not have is
dropped and counted rather than quietly scored.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_vlm_pilot import parse_relations  # noqa: E402

PILOT = ROOT / "outputs" / "vlm_pilot"
PACK = PILOT / "browser_pack"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replies", default=str(PACK / "REPLIES.md"))
    ap.add_argument("--out", default=str(PILOT / "replies_browser.jsonl"))
    args = ap.parse_args()

    src = Path(args.replies)
    if not src.exists():
        sys.exit(f"No {src}. Run scripts/make_browser_pack.py first.")
    text = src.read_text(encoding="utf-8")

    model = "unspecified"
    m = re.search(r"model used:\s*(.+)", text)
    if m:
        cand = m.group(1).strip().strip("_").strip()
        if cand and not cand.startswith("("):
            model = cand.split("(")[0].strip()

    n_objects = {r["image_id"]: r["n_objects"] for r in
                 (json.loads(l) for l in
                  (PILOT / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
                  if l.strip())}

    blocks = re.split(r"(?m)^###\s+(\S+)\s*$", text)
    rows, empty, problems = [], [], 0
    for image_id, body in zip(blocks[1::2], blocks[2::2]):
        if image_id not in n_objects:
            print(f"  ignoring unknown heading: {image_id}")
            continue
        fenced = re.findall(r"```(?:json)?\s*(.*?)```", body, re.S)
        inside = "\n".join(f.strip() for f in fenced).strip()
        # what is left once the fences themselves are removed, so a heading
        # that has only the empty template under it reads as "not done yet"
        # rather than as a reply that failed to parse
        outside = re.sub(r"```(?:json)?.*?```", "", body, flags=re.S).strip()
        if not inside and not outside:
            empty.append(image_id)
            continue
        payload = inside or outside
        rels, bad = parse_relations(payload, n_objects[image_id])
        problems += len(bad)
        if not rels and bad:
            print(f"  {image_id}: nothing usable ({bad[0]})")
        rows.append({"image_id": image_id, "model": model,
                     "relations": rels, "problems": bad, "reply": payload})

    Path(args.out).write_text(
        "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8")

    total = len(n_objects)
    print(f"model recorded as: {model}")
    print(f"parsed {len(rows)}/{total} images, "
          f"{sum(len(r['relations']) for r in rows)} relations, "
          f"{problems} records dropped as malformed")
    if empty:
        print(f"{len(empty)} still empty: {', '.join(empty[:5])}"
              + (" ..." if len(empty) > 5 else ""))
    print(f"\nwrote {args.out}")
    print(f"score it: python eval/score_vlm_pilot.py --replies {args.out}")


if __name__ == "__main__":
    main()
