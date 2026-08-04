"""Measure what each vision-language model costs to run as an annotator.

Accuracy is only half of a deployment decision; the other half is what a pass
over a dataset costs in wall time and tokens. Reasoning models spend most of
their output budget thinking before they answer, which does not show up in
the answer text at all but is billed and waited for, so it has to be measured
rather than inferred from response length.

Same images, same prompts, same settings for every model, so the only
variable is the model. Tokens are reported rather than currency because
per-token prices change; multiply by whatever the current rate is.

    python eval/vlm_cost_benchmark.py --n 8 --models gemini-3.5-flash,gemini-3.1-pro-preview
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import statistics as st
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
       "{model}:generateContent?key={key}")


def api_key() -> str:
    env = (ROOT / ".env").read_text(encoding="utf-8")
    m = re.search(r"GEMINI_API_KEY=(\S+)", env)
    if not m:
        raise SystemExit("no GEMINI_API_KEY in .env")
    return m.group(1)


def one_call(model: str, prompt: str, image: Path, key: str, max_tokens: int):
    b64 = base64.b64encode(image.read_bytes()).decode("ascii")
    body = json.dumps({
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
            {"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens},
    }).encode("utf-8")
    req = urllib.request.Request(URL.format(model=model, key=key), data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read().decode("utf-8"))
    dt = time.time() - t0
    u = d.get("usageMetadata", {})
    cand = d["candidates"][0]
    txt = "\n".join(p.get("text", "")
                    for p in cand.get("content", {}).get("parts", []))
    return {
        "seconds": dt,
        "prompt_tokens": u.get("promptTokenCount", 0),
        "thought_tokens": u.get("thoughtsTokenCount", 0),
        "answer_tokens": u.get("candidatesTokenCount", 0),
        "total_tokens": u.get("totalTokenCount", 0),
        "finish": cand.get("finishReason"),
        "model_version": d.get("modelVersion"),
        "chars": len(txt),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models",
                    default="gemini-3.5-flash,gemini-3.1-pro-preview")
    ap.add_argument("--prompts", default="outputs/vlm_pilot/prompts.jsonl")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=65536,
                    dest="max_tokens")
    ap.add_argument("--dataset-images", type=int, default=836,
                    dest="dataset_images",
                    help="scale the per-image cost to a full pass")
    ap.add_argument("--out", default="outputs/vlm_cost_benchmark.json")
    args = ap.parse_args()

    key = api_key()
    prompts = [json.loads(l) for l in
               Path(args.prompts).read_text(encoding="utf-8").splitlines()
               if l.strip()][:args.n]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    results = {}
    for model in models:
        rows = []
        print(f"\n=== {model} on {len(prompts)} images ===", flush=True)
        for i, p in enumerate(prompts, 1):
            try:
                r = one_call(model, p["prompt"], ROOT / p["image"], key,
                             args.max_tokens)
            except urllib.error.HTTPError as e:
                print(f"  [{i}] HTTP {e.code}: {e.read().decode()[:120]}")
                continue
            rows.append(r)
            print(f"  [{i}/{len(prompts)}] {r['seconds']:5.1f}s  "
                  f"think {r['thought_tokens']:>6}  answer {r['answer_tokens']:>5}  "
                  f"{r['finish']}", flush=True)
        if not rows:
            continue
        med = lambda k: st.median(x[k] for x in rows)
        tot = lambda k: sum(x[k] for x in rows)
        per_image_tokens = tot("total_tokens") / len(rows)
        results[model] = {
            "n": len(rows),
            "model_version": rows[0]["model_version"],
            "median_seconds": med("seconds"),
            "mean_seconds": st.mean(x["seconds"] for x in rows),
            "median_thought_tokens": med("thought_tokens"),
            "median_answer_tokens": med("answer_tokens"),
            "median_prompt_tokens": med("prompt_tokens"),
            "tokens_per_image": per_image_tokens,
            "truncated": sum(1 for x in rows if x["finish"] == "MAX_TOKENS"),
            "projected_hours_for_dataset":
                st.mean(x["seconds"] for x in rows) * args.dataset_images / 3600,
            "projected_tokens_for_dataset":
                per_image_tokens * args.dataset_images,
        }

    print(f"\n{'model':28}{'s/img':>8}{'think':>8}{'answer':>8}"
          f"{'tok/img':>9}{'hours':>8}{'Mtok':>8}")
    for m, r in results.items():
        print(f"{m:28}{r['median_seconds']:8.1f}{r['median_thought_tokens']:8.0f}"
              f"{r['median_answer_tokens']:8.0f}{r['tokens_per_image']:9.0f}"
              f"{r['projected_hours_for_dataset']:8.2f}"
              f"{r['projected_tokens_for_dataset']/1e6:8.2f}")
    print(f"\nprojections are for a {args.dataset_images}-image pass, "
          f"sequential, no concurrency")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"dataset_images": args.dataset_images, "images_sampled": args.n,
         "models": results}, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
