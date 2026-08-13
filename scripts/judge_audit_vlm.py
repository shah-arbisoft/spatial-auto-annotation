"""Have a vision-language model verdict the audit pack, as a second judge.

The audits behind RQ1's precision figures are the author's own, which 2.9
raises as an objection and 7.4 concedes. More author verdicts tighten the
interval without touching the circularity: the same person is still judging
the same tool. What removes it is a judge with no stake, and there are two
available -- the crowd study, which is slow, and a model, which is not.

This asks the model exactly what the human sheet asks, one item at a time, on
the identical rendered image: is this claim true of this picture? It sees the
same red subject box, the same blue object box and the same caption, and it
gets Chapter 3's definitions and the same conservative instruction to answer
WRONG when unsure. It is not told which items are decoys, and the decoys score
it the same way they score a human.

The model's own bias profile is already measured in 4.13, so its verdicts are
interpretable rather than taken on faith: it falls silent rather than
contradicting itself, and where it does speak it is more precise than the
pipeline. A judge that conservative is the right kind for a lower bound.

Image-atomic and resumable, like run_vlm_pilot.py: each answer is appended as
it arrives, so a quota stop loses nothing and re-running continues.

    python scripts/judge_audit_vlm.py --pack outputs/audit_v3
    python scripts/judge_audit_vlm.py --pack outputs/audit_v3 --model gemini-3.1-pro-preview

Writes <pack>/vlm_verdicts.jsonl and, when complete, <pack>/vlm_verdicts.csv
in the same shape as the human sheet so the two can be scored side by side.
"""

from __future__ import annotations

import argparse
import base64
import collections
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_vlm_pilot import api_key, load_jsonl, append_jsonl  # noqa: E402

DEFINITIONS = """You are checking one spatial claim about one photograph.

The SUBJECT is outlined in red and the OBJECT in blue. The claim is printed
along the bottom of the image.

Use these definitions exactly:
- "on" / "under": physically resting on, taking the other's weight. An object
  merely overlapping in the picture, or held by a person, is NOT on it.
- "to the left of" / "to the right of": from the camera's point of view.
- "in front of" / "behind": nearer to / further from the camera.
- "near": close together relative to the objects' own size, and not touching.

Answer with one word, TRUE or WRONG.
Answer WRONG if the claim is false, and also if you cannot tell. Do not guess:
answering WRONG when unsure is what the instruction requires."""


def detail_of(payload: str) -> str:
    """The human-readable half of an API error, not 200 bytes of raw JSON.

    A depleted-credits 429 and an invalid-key 401 look identical when the body
    is printed as-is, and the difference is the whole diagnosis.
    """
    try:
        return json.loads(payload)["error"]["message"].strip()
    except Exception:                                     # noqa: BLE001
        return payload.replace("\n", " ")[:160]


def ask(image_path: Path, claim: str, model: str, key: str,
        retries: int = 4, note=None) -> tuple[str, str, str]:
    """Returns (verdict, raw reply, model version that actually answered).

    The alias in --model is not what answered: Appendix E.1 records that the
    model behind `gemini-flash-latest` moved mid-project, so the reply's own
    modelVersion is captured per item rather than the name we asked for.
    """
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    prompt = f"{DEFINITIONS}\n\nThe claim: {claim}\n\nTRUE or WRONG?"
    body = json.dumps({
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/png", "data": b64}},
            {"text": prompt},
        ]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 2048},
    }).encode("utf-8")

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read())
            ver = d.get("modelVersion", "")
            parts = d["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts).strip()
            up = text.upper()
            # look for the verdict anywhere: a reasoning model may deliberate
            if "WRONG" in up and "TRUE" not in up:
                return "n", text, ver
            if "TRUE" in up and "WRONG" not in up:
                return "y", text, ver
            # both or neither: unparseable, and an unparseable answer is not a
            # verdict. Recorded rather than coerced.
            return "", text, ver
        except urllib.error.HTTPError as e:
            msg = f"HTTP {e.code}: {detail_of(e.read().decode('utf-8', 'replace'))}"
            if e.code in (429, 500, 503) and attempt < retries - 1:
                wait = min(60, 5 * 2 ** attempt)
                if note:
                    note(f"{msg} -- retry {attempt + 1}/{retries - 1} in {wait}s")
                time.sleep(wait)
                continue
            return "", msg, ""
        except Exception as e:                                # noqa: BLE001
            if attempt < retries - 1:
                wait = 5 * 2 ** attempt
                if note:
                    note(f"{type(e).__name__}: {e} -- retry "
                         f"{attempt + 1}/{retries - 1} in {wait}s")
                time.sleep(wait)
                continue
            return "", f"error: {e}", ""
    return "", "retries exhausted", ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="outputs/audit_v3")
    ap.add_argument("--model", default="gemini-flash-latest")
    ap.add_argument("--limit", type=int, default=0, help="0 = all remaining")
    ap.add_argument("--sleep", type=float, default=0.5)
    a = ap.parse_args()

    pack = Path(a.pack)
    sheet = list(csv.DictReader(
        open(pack / "audit_sheet_blind.csv", encoding="utf-8")))
    out = pack / "vlm_verdicts.jsonl"
    # only a parsed y/n counts as done. An HTTP error or an unparseable
    # reply must be retried on the next run, not silently treated as
    # judged, or a bad key would quietly produce an empty audit.
    done = {r["id"] for r in load_jsonl(out) if r.get("verdict") in ("y", "n")}
    todo = [r for r in sheet if r["id"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"  {len(sheet)} items, {len(done)} already judged, {len(todo)} to do")
    if not todo:
        print("  nothing to do")
    key = api_key() if todo else ""

    width = max((len(f"{r['subject']} is {r['predicate']} {r['object']}")
                 for r in todo), default=0)
    tally, versions, t0 = collections.Counter(), collections.Counter(), time.time()

    for i, r in enumerate(todo, 1):
        claim = f"{r['subject']} is {r['predicate']} {r['object']}"
        lead = f"    [{i:>4}/{len(todo)}] #{r['id']:>3}  {claim:<{width}}"
        print(f"{lead}  ... ", end="", flush=True)
        broke = []                    # a retry prints its own line, breaking this one

        def note(msg):
            broke.append(1)
            print(f"\n      !! {msg}", flush=True)

        v, raw, ver = ask(pack / "img" / r["image"], claim, a.model, key,
                          note=note)
        append_jsonl(out, [{"id": r["id"], "image": r["image"],
                            "predicate": r["predicate"], "claim": claim,
                            "verdict": v, "raw": raw[:400], "model": a.model,
                            "model_version": ver}])
        tally[v or "error"] += 1
        if ver:
            versions[ver] += 1
        shown = {"y": "TRUE", "n": "WRONG"}.get(v) or f"** {raw[:90]}"
        print(f"{lead}  {shown}" if broke else shown, flush=True)
        # the running tally is what tells you a run has quietly stopped working
        if i % 25 == 0:
            done_n, err_n = tally["y"] + tally["n"], tally["error"]
            rate = i / max(1e-9, time.time() - t0)
            print(f"      -- {done_n} judged, {err_n} failed, "
                  f"{rate * 60:.0f}/min, ~{(len(todo) - i) / rate / 60:.0f} min left")
        time.sleep(a.sleep)

    # a failed item that later succeeded has two rows; the file is an append
    # log, not a table, so everything below counts items rather than rows
    rows = load_jsonl(out)
    ok = {r["id"]: r for r in rows if r.get("verdict") in ("y", "n")}
    stuck = sorted({r["id"] for r in rows} - set(ok), key=int)
    versions = collections.Counter(r["model_version"] for r in ok.values()
                                   if r.get("model_version"))
    if versions:
        print("\n  answered by: " + ", ".join(f"{k} x{v}"
                                              for k, v in versions.most_common()))
    print(f"\n  {len(ok)}/{len(sheet)} items judged")
    if stuck:
        print(f"  {len(stuck)} still unresolved, re-run to retry: "
              f"{', '.join(stuck[:12])}{' ...' if len(stuck) > 12 else ''}")
    if len(ok) == len(sheet):
        with open(pack / "vlm_verdicts.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "image", "subject",
                                              "predicate", "object",
                                              "verdict (y/n)", "notes"])
            w.writeheader()
            by = ok
            for s in sheet:
                w.writerow({**{k: s[k] for k in
                               ("id", "image", "subject", "predicate", "object")},
                            "verdict (y/n)": by[s["id"]]["verdict"],
                            # the version that answered, not the alias asked for
                            "notes": "vlm:" + (by[s["id"]].get("model_version")
                                               or a.model)})
        print(f"  -> {pack}/vlm_verdicts.csv, same shape as the human sheet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
