"""Re-derive the parsed relations in a replies file from the stored reply text.

Every reply is recorded verbatim alongside its parse, so a parser fix can be
applied to work already paid for rather than re-running the model. That
matters here: gemini-3.5-flash writes `to_the_left_of` where the dataset
writes `to the left of`, and the original parser dropped those as unknown
predicates, discarding 2,033 relations on a 582-image run, concentrated in
the lateral and depth predicates the comparison turns on.

Reports what changed rather than rewriting silently, and refuses to touch a
file whose reply text it cannot reproduce a parse from.

    python scripts/reparse_vlm_replies.py outputs/vlm_pilot/replies_train_f35.jsonl
    python scripts/reparse_vlm_replies.py --check outputs/vlm_pilot/*.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "run_vlm_pilot", ROOT / "scripts" / "run_vlm_pilot.py")
_rvp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rvp)


def reparse(path: Path, write: bool) -> dict:
    rows = [json.loads(l) for l in
            path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return {"file": path.name, "records": 0}

    n_objects = {}
    for pf in ("prompts.jsonl", "prompts_train.jsonl", "prompts_planner.jsonl"):
        p = path.parent / pf
        if p.exists():
            for l in p.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    d = json.loads(l)
                    n_objects[d["image_id"]] = d["n_objects"]

    before = sum(len(r.get("relations", [])) for r in rows)
    before_bad = sum(len(r.get("problems", [])) for r in rows)
    gained = Counter()
    still_bad = Counter()
    out = []
    for r in rows:
        n = n_objects.get(r["image_id"])
        if n is None:            # no prompt for this image: leave untouched
            out.append(r)
            continue
        rels, bad = _rvp.parse_relations(r["reply"], n)
        old = {tuple(x) for x in r.get("relations", [])}
        for x in rels:
            if tuple(x) not in old:
                gained[x[1]] += 1
        for b in bad:
            still_bad[str(b)[:60]] += 1
        out.append({**r, "relations": rels, "problems": bad})

    after = sum(len(r["relations"]) for r in out)
    after_bad = sum(len(r.get("problems", [])) for r in out)
    if write and after != before:
        backup = path.with_suffix(path.suffix + ".prereparse")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text("".join(json.dumps(r) + "\n" for r in out),
                        encoding="utf-8")
    return {"file": path.name, "records": len(rows),
            "relations_before": before, "relations_after": after,
            "problems_before": before_bad, "problems_after": after_bad,
            "gained_by_predicate": dict(gained),
            "still_unparsed": dict(still_bad.most_common(4)),
            "written": bool(write and after != before)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--check", action="store_true",
                    help="report only; do not rewrite")
    args = ap.parse_args()

    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"{f}: missing")
            continue
        s = reparse(p, write=not args.check)
        if not s.get("records"):
            print(f"{s['file']}: empty")
            continue
        delta = s["relations_after"] - s["relations_before"]
        tag = "REWRITTEN" if s["written"] else ("would gain" if delta else "unchanged")
        print(f"{s['file']:42} {s['records']:4} records  "
              f"{s['relations_before']:6} -> {s['relations_after']:6} "
              f"relations ({delta:+}), dropped {s['problems_before']} -> "
              f"{s['problems_after']}   [{tag}]")
        if s["gained_by_predicate"]:
            print(f"    recovered: {s['gained_by_predicate']}")
        if s["still_unparsed"]:
            print(f"    still unparsed: {s['still_unparsed']}")


if __name__ == "__main__":
    main()
