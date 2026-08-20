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
REPLIES = OUT / "replies.jsonl"   # default; --replies overrides so a second
                                  # model writes its own file


def replies_path(args) -> Path:
    """Where this invocation reads and writes replies."""
    return Path(getattr(args, "replies", None) or REPLIES)
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
    print(f"{len(names)} models are listed for generateContent.\n")

    # Listing is not permission: versioned names such as gemini-2.5-flash are
    # still catalogued but answer generateContent with 404 "no longer
    # available to new users". The only reliable check is to call each one.
    candidates = [args.model] + [n for n in names
                                 if "flash" in n and n != args.model][:6]
    print("probing generateContent (listed is not the same as callable):")
    working = []
    for name in candidates:
        try:
            call_gemini("Reply with the single word: ok", name, key, retries=1)
            print(f"   {name:28s} OK")
            working.append(name)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            try:
                msg = json.loads(body)["error"]["message"][:70]
            except Exception:
                msg = body[:70]
            print(f"   {name:28s} HTTP {e.code}  {msg}")

    if not working:
        print("\nNo model answered. If every line says 429 the key is over its "
              "quota (wait, or use a different key); if they say 404 the names "
              "have moved on again and the -latest aliases are the safe pick.")
        sys.exit(1)
    print(f"\nUse: python scripts/run_planner_llm.py --model {working[0]}")
    if working[0] != args.model:
        print(f"(the current default, {args.model}, is not callable with this key)")


class QuotaExhausted(Exception):
    """The key's daily allowance for this model is spent."""


def _quota_info(err_body: str):
    """(is_daily_quota, retry_delay_seconds) from a 429 body."""
    try:
        err = json.loads(err_body)["error"]
    except Exception:
        return False, None
    daily, delay = False, None
    for d in err.get("details", []):
        kind = d.get("@type", "").rsplit(".", 1)[-1]
        if kind == "QuotaFailure":
            for v in d.get("violations", []):
                if "PerDay" in (v.get("quotaId") or ""):
                    daily = True
        elif kind == "RetryInfo":
            raw = str(d.get("retryDelay", ""))
            if raw.endswith("s"):
                try:
                    delay = float(raw[:-1])
                except ValueError:
                    pass
    return daily, delay


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
            # NB: not `body`, which holds the encoded request. Overwriting it
            # here made every retry after a 429 fail with a TypeError, so the
            # backoff this function implements had never actually run.
            detail = e.read().decode("utf-8", "replace")
            if e.code == 429:
                daily, delay = _quota_info(detail)
                if daily:
                    # a per-day cap: no amount of waiting helps within this run
                    raise QuotaExhausted(model) from None
                if attempt < retries - 1:
                    wait = delay if delay else 15 * (attempt + 1)
                    print(f"    rate limited, waiting {wait:.0f}s ...", flush=True)
                    time.sleep(wait)
                    continue
            elif e.code in (500, 503) and attempt < retries - 1:
                time.sleep(10 * (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def cmd_run(args):
    """Answer prompts a whole scene at a time.

    Scene-atomic on purpose. The free tier caps requests per day *per model*,
    so a long run can stop midway and be resumed later with a different
    model. If that boundary fell inside a scene, that scene's conditions
    would no longer be comparable, since the whole design rests on A, B and C
    differing only in the relations supplied. A scene is therefore written
    only once all three of its conditions succeed with the same model.
    """
    key = api_key()
    prompts = load_jsonl(Path(args.prompts))
    rp = replies_path(args)
    done = load_jsonl(rp) if rp.exists() else []
    have = {(r["scene"], r["condition"]) for r in done}

    by_scene = {}
    for p in prompts:
        by_scene.setdefault(p["scene"], []).append(p)
    complete = [s for s, ps in by_scene.items()
                if all((s, p["condition"]) in have for p in ps)]
    todo = [s for s in sorted(by_scene) if s not in complete]

    print(f"{len(by_scene)} scenes: {len(complete)} complete, {len(todo)} to go "
          f"(model {args.model})")
    if not todo:
        print("nothing to do; next: python scripts/run_planner_llm.py --make-sheet")
        return

    written = 0
    try:
        for n, scene in enumerate(todo, 1):
            batch = []
            for p in by_scene[scene]:
                if (scene, p["condition"]) in have:
                    continue
                print(f"  [scene {scene}: {n}/{len(todo)}] {p['condition']} ...", flush=True)
                reply = call_gemini(p["prompt"], args.model, key)
                batch.append({"scene": scene, "condition": p["condition"],
                              "model": args.model, "reply": reply})
                time.sleep(args.sleep)
            with open(rp, "a", encoding="utf-8") as f:   # commit whole scene
                for rec in batch:
                    f.write(json.dumps(rec) + "\n")
            written += len(batch)
    except QuotaExhausted as e:
        print(f"\nDaily quota for {e} is spent. The partly-answered scene was "
              f"discarded so no scene mixes models.")
        print(f"{written} replies added this run; {len(complete)} of "
              f"{len(by_scene)} scenes complete in total.")
        print("\nOptions:")
        print("  * wait for the quota to reset and re-run the same command")
        print("  * use a model with its own allowance, e.g. "
              "--model gemini-flash-lite-latest (record it: scenes then differ "
              "in model, though A/B/C within a scene never do)")
        print("  * enable billing on the key's project to lift the cap")
        sys.exit(2)

    print(f"done: {written} replies added -> {rp}")
    print("next: python scripts/run_planner_llm.py --make-sheet")


def cmd_make_sheet(args):
    replies = {(r["scene"], r["condition"]): r["reply"]
               for r in load_jsonl(replies_path(args))}
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


def cmd_score(args):
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


def cmd_status(args):
    """Progress, and drop any scene left half-answered by an interrupted run."""
    prompts = load_jsonl(Path(args.prompts))
    by_scene = {}
    for p in prompts:
        by_scene.setdefault(p["scene"], set()).add(p["condition"])
    rp = replies_path(args)
    recs = load_jsonl(rp) if rp.exists() else []

    got = {}
    for r in recs:
        got.setdefault(r["scene"], set()).add(r["condition"])
    partial = [s for s, cs in got.items() if cs != by_scene.get(s, set())]

    if partial:
        keep = [r for r in recs if r["scene"] not in partial]
        rp.write_text("".join(json.dumps(r) + "\n" for r in keep), encoding="utf-8")
        print(f"dropped {len(recs) - len(keep)} replies from partly-answered "
              f"scene(s) {partial}: a scene must be answered by one model or "
              f"its conditions are not comparable.")
        recs = keep
        got = {}
        for r in recs:
            got.setdefault(r["scene"], set()).add(r["condition"])

    complete = sorted(s for s, cs in got.items() if cs == by_scene.get(s, set()))
    models = sorted({r["model"] for r in recs})
    print(f"scenes complete: {len(complete)}/{len(by_scene)}  "
          f"({len(recs)}/{len(prompts)} replies)")
    print(f"models used: {', '.join(models) if models else 'none yet'}")
    if len(complete) < len(by_scene):
        print("\nresume with: python scripts/run_planner_llm.py")
    else:
        print("\nnext: python scripts/run_planner_llm.py --make-sheet")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # A model can appear in ListModels and still refuse generateContent with
    # "no longer available to new users" (a 404), which is what versioned
    # names like gemini-2.5-flash now do. The floating -latest aliases keep
    # working, so one is the default and --doctor verifies before a long run.
    ap.add_argument("--model", default="gemini-flash-latest")
    ap.add_argument("--prompts", default=str(PROMPTS),
                    help="prompt file to answer")
    ap.add_argument("--replies", default=None,
                    help="reply file; give each model its own so runs stay "
                         "separable and neither can overwrite the other")
    ap.add_argument("--sleep", type=float, default=5.0,
                    help="seconds between API calls (free-tier friendly)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--doctor", action="store_true",
                      help="check the key and list callable models")
    mode.add_argument("--status", action="store_true",
                      help="progress so far; clears any half-answered scene")
    mode.add_argument("--make-sheet", action="store_true")
    mode.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.doctor:
        cmd_doctor(args)
    elif args.status:
        cmd_status(args)
    elif args.make_sheet:
        cmd_make_sheet(args)
    elif args.score:
        cmd_score(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
