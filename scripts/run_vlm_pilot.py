"""Week-9 pilot: measure a VLM as a spatial-relationship annotator.

The question this decides is what role a vision-language model should play in
this dissertation: a footnote, an adjudicator for ambiguous pairs, or a
fourth label source alongside human, self-trained and automatic. That
decision needs a measurement, not an opinion, so the VLM is put through the
same battery the geometric pipeline faces in Chapter 4 and scored on
identical pairs.

The comparison is made fair by construction:

  * The VLM is given the ground-truth boxes, drawn on the image and numbered,
    and is asked to name objects by index. This is the PredCls setting the
    pipeline is evaluated in (§3.3), so neither side is being scored on
    detection.
  * It is asked for the same seven predicates, with the operational
    definitions Chapter 3 fixes, including that "in front of" means nearer
    the camera. Without that the reference-frame ambiguity of §2.5 would be
    measured instead of accuracy.
  * Only pairs the human annotators labelled are scored, and the pipeline's
    own answer on those same pairs is reported beside it.

Free-tier quota is a daily per-model cap, so this is image-atomic and
resumable in exactly the way `run_planner_llm.py` is: an image is recorded
only when its reply parses, and a stopped run resumes where it left off.

    python scripts/run_vlm_pilot.py --make            # renders + prompts
    python scripts/run_vlm_pilot.py                   # answer them
    python scripts/run_vlm_pilot.py --status
    python eval/score_vlm_pilot.py                    # score against RQ1
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import SpatialDataset, load_rgb
from src.pipeline import load_config

OUT = ROOT / "outputs" / "vlm_pilot"
IMG = OUT / "img"
PROMPTS = OUT / "prompts.jsonl"
REPLIES = OUT / "replies.jsonl"   # default; --replies overrides so a
                                  # second model writes its own file

PREDICATES = ["on", "under", "to the left of", "to the right of",
              "in front of", "behind", "near"]

INSTRUCTIONS = """You are annotating spatial relationships in a photograph.

The objects have been outlined and numbered for you. Use only these numbers.

Objects:
{objects}

For every ORDERED pair of objects that stands in one of these relationships,
output one record. The seven allowed predicates, with the definitions you
must apply:

  on              subject physically rests on object, in contact and supported
  under           subject is beneath object and supports it, the inverse of on
  to the left of  subject appears further left than object from the camera
  to the right of subject appears further right than object from the camera
  in front of     subject is NEARER TO THE CAMERA than object
  behind          subject is FURTHER FROM THE CAMERA than object
  near            subject and object are close together relative to their size

"in front of" and "behind" are judged from the camera's position, not from
any object's own facing direction, and not from the viewer's left or right.

Reply with JSON only, no commentary, in exactly this form:

{{"relations": [{{"s": 0, "p": "on", "o": 3}}, {{"s": 3, "p": "under", "o": 0}}]}}

Omit any pair you are unsure about rather than guessing."""


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


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in
            path.read_text(encoding="utf-8").splitlines() if l.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


class QuotaExhausted(Exception):
    """The key's daily allowance for this model is spent."""


def _quota_info(err_body: str):
    """(is_daily_quota, retry_delay_seconds) from a 429 body."""
    daily, delay = False, 0.0
    try:
        details = json.loads(err_body).get("error", {}).get("details", [])
    except Exception:
        return daily, delay
    for d in details:
        for v in d.get("violations", []):
            if "PerDay" in (v.get("quotaId") or ""):
                daily = True
        raw = d.get("retryDelay") or ""
        if raw.endswith("s"):
            try:
                delay = float(raw[:-1])
            except ValueError:
                pass
    return daily, delay


def render_numbered(image_rgb, objects, path: Path) -> None:
    """Draw each ground-truth box with its index, so the VLM can refer to it."""
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415

    img = Image.fromarray(image_rgb).convert("RGB")
    W, H = img.size
    # The source is 640x480 greyscale. Upscaling before drawing keeps the
    # index labels legible to the model; an integer scale factor rounds to 1
    # at this size and leaves them at about 14 px, which is the difference
    # between a readable tag and a guess.
    target = 1280
    if max(W, H) < target:
        s = target / max(W, H)
        img = img.resize((int(W * s), int(H * s)), Image.LANCZOS)
        W, H = img.size
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", max(16, W // 45))
    except OSError:
        font = ImageFont.load_default()
    for i, o in enumerate(objects):
        x1, y1, x2, y2 = (o.box[0] * W, o.box[1] * H, o.box[2] * W, o.box[3] * H)
        d.rectangle([x1, y1, x2, y2], outline=(255, 40, 40), width=3)
        tag = str(i)
        tb = d.textbbox((0, 0), tag, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        bx, by = x1, max(0, y1 - th - 6)
        d.rectangle([bx, by, bx + tw + 8, by + th + 6], fill=(255, 40, 40))
        d.text((bx + 4, by + 2), tag, fill=(255, 255, 255), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=92)


def cmd_make(args):
    """Choose the sample, render numbered images, write the prompts."""
    cfg = load_config(args.config)
    ds = {im.image_id: im for im in SpatialDataset(cfg["dataset"]["root"])}

    # only images the human annotators gave at least one relation, since the
    # battery scores recall of human triplets
    scored = [i for i, im in ds.items() if im.relations]
    if args.only:
        wanted = Path(args.only)
        ids = [l.strip() for l in
               (wanted.read_text(encoding="utf-8").splitlines()
                if wanted.exists() else args.only.split(","))
               if l.strip()]
        missing_ids = [i for i in ids if i not in ds]
        if missing_ids:
            sys.exit(f"unknown image ids: {missing_ids[:5]}")
        scored = [i for i in ids if ds[i].relations]
    if args.groups:
        want = {g.strip() for g in args.groups.split(",") if g.strip()}
        scored = [i for i in scored if i.split("/")[0] in want]
        if not scored:
            sys.exit(f"no annotated images in groups {sorted(want)}")
    rng = random.Random(args.seed)
    rng.shuffle(scored)
    # stratify across annotator groups so no single convention dominates
    by_group: dict[str, list[str]] = defaultdict(list)
    for i in scored:
        by_group[i.split("/")[0]].append(i)
    chosen: list[str] = []
    groups = sorted(by_group)
    target = len(scored) if args.n < 0 else args.n
    while len(chosen) < target:
        added = False
        for g in groups:
            if by_group[g] and len(chosen) < target:
                chosen.append(by_group[g].pop())
                added = True
        if not added:
            break

    rows = []
    missing = []
    for image_id in chosen:
        im = ds[image_id]
        # A handful of annotation files have no matching image (§4.2 excludes
        # the same records from the fidelity totals). Skip rather than abort:
        # a stray record should not stop a 600-image run.
        if not Path(im.image_path).exists():
            missing.append(image_id)
            continue
        rgb = load_rgb(im.image_path)
        png = IMG / f"{image_id.replace('/', '_')}.jpg"
        render_numbered(rgb, im.objects, png)
        objects = "\n".join(f"  {i}: {o.label}" for i, o in enumerate(im.objects))
        rows.append({"image_id": image_id, "image": str(png.relative_to(ROOT)),
                     "n_objects": len(im.objects),
                     "prompt": INSTRUCTIONS.format(objects=objects)})

    prompts_out = Path(args.prompts)
    prompts_out.parent.mkdir(parents=True, exist_ok=True)
    prompts_out.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                           encoding="utf-8")
    if missing:
        print(f"skipped {len(missing)} annotation records with no image file: "
              f"{', '.join(missing[:4])}{' ...' if len(missing) > 4 else ''}")
    print(f"{len(rows)} images rendered to {IMG}")
    print(f"prompts -> {prompts_out}")
    print(f"\nnext: python scripts/run_vlm_pilot.py   (needs {len(rows)} calls; "
          f"the free tier allows 20 per day per model)")


class Truncated(RuntimeError):
    """The model ran out of output budget mid-answer.

    Worth its own exception because the symptom is otherwise indistinguishable
    from a model that writes bad JSON: the reply simply stops mid-token and
    fails to parse. Reasoning models spend part of this budget on thinking
    before any answer is emitted, so a ceiling that never bound a
    non-reasoning model can bind a reasoning one on the same prompt.
    """


def call_gemini(prompt: str, image_path: Path, model: str, key: str,
                retries: int = 4, max_output_tokens: int = 8192) -> str:
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    body = json.dumps({
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
            {"text": prompt},
        ]}],
        "generationConfig": {"temperature": 0,
                             "maxOutputTokens": max_output_tokens},
    }).encode("utf-8")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.loads(r.read().decode("utf-8"))
            cand = data["candidates"][0]
            if cand.get("finishReason") == "MAX_TOKENS":
                used = data.get("usageMetadata", {})
                raise Truncated(
                    f"{model} hit maxOutputTokens={max_output_tokens} "
                    f"(thoughts {used.get('thoughtsTokenCount', '?')}, "
                    f"answer {used.get('candidatesTokenCount', '?')})")
            parts = cand.get("content", {}).get("parts", [])
            return "\n".join(p.get("text", "") for p in parts).strip()
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", "replace")
            if e.code == 429:
                daily, delay = _quota_info(err)
                if daily:
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


def parse_relations(reply: str, n_objects: int) -> tuple[list, list]:
    """(relations, problems) from a JSON reply, dropping anything malformed.

    Models wrap JSON in fences and occasionally emit an index that does not
    exist, so both are handled here rather than corrupting the score.
    """
    txt = reply.strip()
    txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.M).strip()
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return [], ["no JSON object in reply"]
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return [], [f"unparseable JSON: {e}"]

    out, bad = [], []
    for r in data.get("relations", []):
        try:
            s, p, o = int(r["s"]), str(r["p"]).strip().lower(), int(r["o"])
        except (KeyError, TypeError, ValueError):
            bad.append(f"malformed record {r}")
            continue
        if p not in PREDICATES:
            bad.append(f"unknown predicate {p!r}")
            continue
        if not (0 <= s < n_objects and 0 <= o < n_objects) or s == o:
            bad.append(f"index out of range or self-pair ({s},{o})")
            continue
        out.append([s, p, o])
    return out, bad


def cmd_run(args):
    prompts = load_jsonl(Path(args.prompts))
    if not prompts:
        sys.exit("No prompts. Run with --make first.")
    replies_path = Path(args.replies)
    done = {r["image_id"] for r in load_jsonl(replies_path)}
    todo = [p for p in prompts if p["image_id"] not in done]
    if not todo:
        print(f"all {len(prompts)} images already answered. "
              f"next: python eval/score_vlm_pilot.py")
        return
    key = api_key()
    print(f"{len(done)}/{len(prompts)} done; asking {len(todo)} more "
          f"with {args.model}")

    for n, p in enumerate(todo, 1):
        print(f"  [{n}/{len(todo)}] {p['image_id']} ...", flush=True)
        try:
            reply = call_gemini(p["prompt"], ROOT / p["image"], args.model, key,
                                max_output_tokens=args.max_output_tokens)
        except Truncated as e:
            # Not recorded. A truncated answer is an artefact of the budget,
            # not evidence about the model's spatial judgement, and recording
            # it would both understate the model and block the retry, since
            # resumption keys on image_id.
            print(f"\n  {e}\n"
                  f"  {len(done) + n - 1}/{len(prompts)} images done. Re-run "
                  f"with a larger --max-output-tokens to finish the rest.")
            return
        except QuotaExhausted:
            print(f"\nDaily quota for {args.model} is spent. "
                  f"{len(done) + n - 1}/{len(prompts)} images done.\n"
                  f"Resume tomorrow with the same command, or use a different "
                  f"model with --model (its allowance is separate, and the "
                  f"model used is recorded per image).")
            return
        rels, bad = parse_relations(reply, p["n_objects"])
        append_jsonl(replies_path, [{"image_id": p["image_id"], "model": args.model,
                                "relations": rels, "problems": bad,
                                "reply": reply}])
        if bad:
            print(f"      {len(rels)} relations, {len(bad)} dropped: {bad[0]}")
        else:
            print(f"      {len(rels)} relations")
        time.sleep(args.sleep)

    print(f"\ndone. next: python eval/score_vlm_pilot.py")


def cmd_status(args):
    prompts = load_jsonl(Path(args.prompts))
    replies = load_jsonl(Path(args.replies))
    print(f"images answered: {len(replies)}/{len(prompts)}")
    if replies:
        models = sorted({r.get("model", "?") for r in replies})
        rels = sum(len(r["relations"]) for r in replies)
        bad = sum(len(r.get("problems", [])) for r in replies)
        print(f"models used: {', '.join(models)}")
        print(f"relations parsed: {rels} ({rels / len(replies):.1f} per image)")
        print(f"records dropped as malformed: {bad}")
    if len(replies) < len(prompts):
        print("\nresume with: python scripts/run_vlm_pilot.py")
    else:
        print("\nnext: python eval/score_vlm_pilot.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--make", action="store_true",
                    help="render numbered images and build the prompts")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--n", type=int, default=30,
                    help="how many images; -1 for every annotated image in "
                         "the selected groups")
    ap.add_argument("--only", default=None,
                    help="comma-separated image ids, or a file of them one "
                         "per line; overrides the sampling")
    ap.add_argument("--groups", default=None,
                    help="comma-separated annotator groups to draw from, e.g. "
                         "group_0,group_1. Default: all")
    ap.add_argument("--prompts", default=str(PROMPTS),
                    help="prompt file to write with --make and read otherwise")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--model", default="gemini-flash-latest")
    ap.add_argument("--max-output-tokens", type=int, default=8192,
                    dest="max_output_tokens",
                    help="output budget per call. Reasoning models spend part "
                         "of it thinking, so they need more than the 8192 that "
                         "sufficed for the non-reasoning run")
    ap.add_argument("--replies", default=str(REPLIES),
                    help="reply file; give each model its own so runs stay "
                         "separable and neither can overwrite the other")
    ap.add_argument("--sleep", type=float, default=2.0)
    args = ap.parse_args()

    if args.make:
        cmd_make(args)
    elif args.status:
        cmd_status(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
