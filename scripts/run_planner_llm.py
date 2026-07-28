"""Run the planner experiment end to end: prompts -> LLM -> blind sheet -> scores.

The experiment (scripts/planner_experiment.py) generated 75 prompts: 25
held-out scenes x 3 conditions (A no relations, B human relations, C this
tool's relations). This script does the three remaining steps.

1. Query the LLM (default):
       set GEMINI_API_KEY=...              (or export on Unix)
       python scripts/run_planner_llm.py
   Calls Gemini at temperature 0 for every prompt, resumably: replies land
   in outputs/planner/replies.jsonl and finished prompts are skipped on
   re-run, so an interrupted pass loses nothing.

2. Build the scorer's sheet:
       python scripts/run_planner_llm.py --make-sheet
   Writes outputs/planner/scoring_sheet_filled.csv with the PLAN TEXT
   embedded in shuffled slot order and no identifiers. This corrects a
   blinding flaw in the original sheet, whose prompt-id column
   ("scene10-B") leaked the condition letter to the scorer. Give the scorer
   ONLY this file; the slot-to-condition key stays in
   _key_do_not_share.csv.

3. Score the returned sheet:
       python scripts/run_planner_llm.py --score
   Joins the filled sheet with the key and reports, per condition, the rate
   at which plans clear the top object first, grasp the right target and
   invent nothing -> outputs/planner/results.json + a markdown table.

The scorer answers three y/n questions per plan, in slot order, without
knowing which condition produced which plan.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("outputs/planner")
PROMPTS = OUT / "prompts.jsonl"
REPLIES = OUT / "replies.jsonl"
BLIND = OUT / "scoring_sheet_blind.csv"
FILLED = OUT / "scoring_sheet_filled.csv"
KEY = OUT / "_key_do_not_share.csv"
CRITERIA = [
    "clears the object on top before grasping (y/n)",
    "grasps the correct object (y/n)",
    "no invented objects or steps (y/n)",
]


def load_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def api_key() -> str:
    """From the environment, else a KEY=VALUE line in .env at the repo root."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                if name.strip() == "GEMINI_API_KEY":
                    key = value.strip().strip("\"'")
                    break
    if not key:
        sys.exit("No GEMINI_API_KEY. Copy .env.example to .env and paste your "
                 "key, or set the environment variable.")
    if key.startswith("AQ."):
        sys.exit("That looks like a Gemini CLI OAuth token (keys of this shape "
                 "begin 'AQ.'). The REST endpoint used here needs a Google AI "
                 "Studio API key, which begins 'AIza'. Create one at "
                 "https://aistudio.google.com/apikey and put it in .env.")
    return key


def list_models(key: str):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.loads(r.read().decode("utf-8"))
    out = []
    for m in data.get("models", []):
        if "generateContent" in m.get("supportedGenerationMethods", []):
            out.append(m["name"].removeprefix("models/"))
    return out


def cmd_doctor(args):
    """Check the credential and report which models it can actually call."""
    key = api_key()
    print(f"key loaded: {len(key)} chars, starts {key[:4]}...")
    try:
        names = list_models(key)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        print(f"\nListModels failed: HTTP {e.code}")
        print(body)
        if e.code in (400, 403):
            print("\n-> the key is not valid for this API. Create an AI Studio "
                  "key at https://aistudio.google.com/apikey")
        sys.exit(1)
    print(f"\n{len(names)} models available to this key for generateContent:")
    for n in names:
        print("   ", n)
    flash = [n for n in names if "flash" in n and "thinking" not in n]
    if flash:
        pick = sorted(flash, key=len)[0]
        print(f"\nSuggested: --model {pick}")
        print(f"   python scripts/run_planner_llm.py --model {pick}")
    if args.model not in names:
        print(f"\nNote: the current default ({args.model}) is NOT in the list "
              "above, which is what produces HTTP 404.")


def call_gemini(prompt: str, model: str, api_key: str, retries: int = 4) -> str:
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={api_key}")
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0},
    }).encode("utf-8")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                data = json.loads(r.read().decode("utf-8"))
            parts = data["candidates"][0]["content"]["parts"]
            return "\n".join(p.get("text", "") for p in parts).strip()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < retries - 1:
                wait = 15 * (attempt + 1)
                print(f"    HTTP {e.code}, retrying in {wait}s ...", flush=True)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("unreachable")


def cmd_run(args):
    key = api_key()
    prompts = load_jsonl(PROMPTS)
    done = {(r["scene"], r["condition"]) for r in load_jsonl(REPLIES)} if REPLIES.exists() else set()
    todo = [p for p in prompts if (p["scene"], p["condition"]) not in done]
    print(f"{len(prompts)} prompts, {len(done)} already answered, {len(todo)} to go "
          f"(model {args.model})")
    with open(REPLIES, "a", encoding="utf-8") as f:
        for i, p in enumerate(todo, 1):
            tag = f"scene{p['scene']}-{p['condition']}"
            print(f"  [{i}/{len(todo)}] {tag} ...", flush=True)
            reply = call_gemini(p["prompt"], args.model, key)
            f.write(json.dumps({"scene": p["scene"], "condition": p["condition"],
                                "model": args.model, "reply": reply}) + "\n")
            f.flush()
            time.sleep(args.sleep)
    print(f"done -> {REPLIES}")
    print("next: python scripts/run_planner_llm.py --make-sheet")


def cmd_make_sheet(_args):
    replies = {(r["scene"], r["condition"]): r["reply"] for r in load_jsonl(REPLIES)}
    rows = list(csv.DictReader(open(BLIND, newline="", encoding="utf-8")))
    out_rows, missing = [], []
    for row in rows:
        pid = row["plan_pasted_from_prompt_id"]           # e.g. scene10-B
        scene = int(pid.split("-")[0].removeprefix("scene"))
        cond = pid.split("-")[1]
        reply = replies.get((scene, cond))
        if reply is None:
            missing.append(pid)
            continue
        out_rows.append({"slot": row["slot"], "plan": reply,
                         **{c: "" for c in CRITERIA}, "notes": ""})
    if missing:
        sys.exit(f"{len(missing)} replies missing (run without flags first): {missing[:5]}")
    with open(FILLED, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"{len(out_rows)} plans -> {FILLED}")
    print("Give the scorer ONLY this file (it contains no scene or condition "
          "identifiers). They fill the three y/n columns in slot order.")


def cmd_score(_args):
    key = {r["slot"]: r for r in csv.DictReader(open(KEY, newline="", encoding="utf-8"))}
    rows = list(csv.DictReader(open(FILLED, newline="", encoding="utf-8")))
    per = {c: {crit: [0, 0] for crit in CRITERIA} for c in "ABC"}
    per_all = {c: [0, 0] for c in "ABC"}
    unscored = 0
    for row in rows:
        cond = key[row["slot"]]["condition"]
        verdicts = [(row.get(c) or "").strip().lower() for c in CRITERIA]
        if any(v not in ("y", "n") for v in verdicts):
            unscored += 1
            continue
        for crit, v in zip(CRITERIA, verdicts):
            per[cond][crit][1] += 1
            per[cond][crit][0] += v == "y"
        per_all[cond][1] += 1
        per_all[cond][0] += all(v == "y" for v in verdicts)

    names = {"A": "no relations", "B": "human relations", "C": "automatic relations"}
    md = ["# Planner experiment results\n",
          "| criterion | A (none) | B (human) | C (automatic) |", "|---|---|---|---|"]
    report = {}
    for crit in CRITERIA + ["all three criteria"]:
        cells = []
        for c in "ABC":
            k, n = per_all[c] if crit == "all three criteria" else per[c][crit]
            cells.append(f"{k}/{n} ({k / n:.0%})" if n else "-")
            report.setdefault(names[c], {})[crit] = {"yes": k, "n": n}
        md.append(f"| {crit.removesuffix(' (y/n)')} | " + " | ".join(cells) + " |")
    if unscored:
        md.append(f"\n{unscored} plans not yet scored.")
    (OUT / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "results.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n-> {OUT / 'results.json'} ; {OUT / 'results.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--sleep", type=float, default=5.0,
                    help="seconds between API calls (free-tier friendly)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--doctor", action="store_true",
                      help="check the key and list callable models")
    mode.add_argument("--make-sheet", action="store_true")
    mode.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.doctor:
        cmd_doctor(args)
    elif args.make_sheet:
        cmd_make_sheet(args)
    elif args.score:
        cmd_score(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
