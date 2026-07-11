"""Video demonstration: per-frame annotation + temporal smoothing.

Runs the full detector-in-the-loop stack (GroundingDINO + SAM2 + Depth
Anything + the shipped rules) on every sampled frame of a video, tracks
objects across frames (greedy IoU, same class), applies a temporal majority
vote to each object pair's predicates, and writes:

  outputs/video/frames.jsonl   one record per processed frame (raw + smoothed)
  outputs/video/annotated.gif  overlay: boxes + the smoothed triplet list
  outputs/video/annotated.mp4  same overlay at full quality
  console                      stability summary (raw vs smoothed)

This is a demonstration, not an experiment: the dataset has no video ground
truth. It shows the annotator is video-ready (SAM2 is natively a video model;
here each frame is processed independently and smoothed after the fact).

    python scripts/run_video.py demo_video.mp4 --only-prompts \
        --prompts "computer monitor,keyboard,mouse,cup,glasses,potted plant,lamp,notebook"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import annotate_objects, load_config, objects_from
from src.contact import pair_contacts
from run_sgdet import PROMPTS, detect

COLOURS = ["#e6194b", "#4363d8", "#3cb44b", "#f58231", "#911eb4", "#46f0f0",
           "#f032e6", "#bcf60c", "#008080", "#9a6324", "#800000", "#000075"]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Tracker:
    """Greedy same-class IoU matching frame to frame — stable object ids."""

    def __init__(self, min_iou=0.30):
        self.min_iou = min_iou
        self.next_id = 0
        self.prev: list[tuple[int, str, list[float]]] = []  # (tid, label, box)

    def assign(self, dets) -> list[int]:
        ids = [-1] * len(dets)
        used = set()
        order = sorted(range(len(dets)), key=lambda i: -dets[i]["score"])
        for i in order:
            best, best_v = None, self.min_iou
            for j, (tid, label, box) in enumerate(self.prev):
                if j in used or label != dets[i]["label"]:
                    continue
                v = iou(dets[i]["box_px"], box)
                if v > best_v:
                    best, best_v = j, v
            if best is not None:
                used.add(best)
                ids[i] = self.prev[best][0]
            else:
                ids[i] = self.next_id
                self.next_id += 1
        self.prev = [(ids[i], dets[i]["label"], dets[i]["box_px"])
                     for i in range(len(dets))]
        return ids


def smooth(per_frame: list[dict], window: int = 2) -> list[set]:
    """Majority vote per (subject-track, predicate, object-track) over ±window
    frames, counting only frames where both tracks are visible."""
    visible = [set(f["tracks"].values()) for f in per_frame]
    raw = [set(map(tuple, f["triplets_raw"])) for f in per_frame]
    out = []
    for i in range(len(per_frame)):
        lo, hi = max(0, i - window), min(len(per_frame), i + window + 1)
        votes, denom = defaultdict(int), defaultdict(int)
        cand = set().union(*raw[lo:hi])
        for s, k, o in cand:
            for j in range(lo, hi):
                if s in visible[j] and o in visible[j]:
                    denom[(s, k, o)] += 1
                    if (s, k, o) in raw[j]:
                        votes[(s, k, o)] += 1
        keep = {t for t in cand
                if visible[i] >= {t[0], t[2]}
                and denom[t] > 0 and votes[t] * 2 > denom[t]}
        out.append(keep)
    return out


def render(frame_rgb, dets, ids, triplets, names, max_lines=10):
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415

    img = Image.fromarray(frame_rgb)
    W, H = img.size
    try:
        font = ImageFont.truetype("arial.ttf", max(14, W // 60))
    except OSError:
        font = ImageFont.load_default()
    lines = [f"{names[s]}  -{k}->  {names[o]}" for s, k, o in sorted(triplets)]
    strip_h = (min(len(lines), max_lines) + (1 if len(lines) > max_lines else 0)) \
        * (font.size + 6) + 8 if lines else 0
    canvas = Image.new("RGB", (W, H + strip_h), "black")
    canvas.paste(img, (0, 0))
    dr = ImageDraw.Draw(canvas)
    for d, tid in zip(dets, ids):
        c = COLOURS[tid % len(COLOURS)]
        x1, y1, x2, y2 = d["box_px"]
        dr.rectangle([x1, y1, x2, y2], outline=c, width=3)
        tag = names[tid]
        ty = max(0, y1 - font.size - 6)
        dr.rectangle([x1, ty, x1 + dr.textlength(tag, font=font) + 8,
                      ty + font.size + 4], fill=c)
        dr.text((x1 + 4, ty + 2), tag, fill="white", font=font)
    for j, line in enumerate(lines[:max_lines]):
        dr.text((6, H + 4 + (font.size + 6) * j), line, fill="white", font=font)
    if len(lines) > max_lines:
        dr.text((6, H + 4 + (font.size + 6) * max_lines),
                f"(+{len(lines) - max_lines} more)", fill="gray", font=font)
    return canvas


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="path to a video file")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--prompts", default=None,
                    help="comma-separated objects to look for")
    ap.add_argument("--only-prompts", action="store_true",
                    help="use --prompts INSTEAD of the six dataset classes")
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--stride", type=int, default=2,
                    help="process every Nth frame")
    ap.add_argument("--width", type=int, default=1280,
                    help="downscale frames to this width for inference")
    ap.add_argument("--window", type=int, default=2,
                    help="temporal majority window: +/- this many frames")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/video")
    args = ap.parse_args()

    import cv2  # noqa: PLC0415

    if args.prompts:
        extra = {p.strip(): p.strip().replace(" ", "_")
                 for p in args.prompts.split(",") if p.strip()}
        if args.only_prompts:
            PROMPTS.clear()
        PROMPTS.update(extra)

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import pipeline as hf_pipeline  # noqa: PLC0415
    print("loading models (GroundingDINO + SAM2 + Depth Anything) ...")
    detector = hf_pipeline("zero-shot-object-detection",
                           model="IDEA-Research/grounding-dino-tiny",
                           device=0 if args.device == "cuda" else -1)
    from src.depth import DepthEstimator  # noqa: PLC0415
    from src.segment import Segmenter  # noqa: PLC0415
    depther = DepthEstimator(hf_model=cfg["depth"]["hf_model"], device=args.device).load()
    segmenter = Segmenter(hf_model=cfg["segmentation"]["hf_model"], device=args.device).load()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    from PIL import Image  # noqa: PLC0415
    from tqdm import tqdm  # noqa: PLC0415

    tracker = Tracker()
    per_frame, frames_rgb = [], []
    z_scale = cfg.get("geometry", {}).get("z_scale", 1.0)

    idxs = list(range(0, n_total, args.stride))
    for fi in tqdm(idxs, desc="frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, bgr = cap.read()
        if not ok:
            break
        scale = args.width / bgr.shape[1]
        if scale < 1.0:
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        dets = detect(detector, rgb, args.threshold)
        ids = tracker.assign(dets)
        triplets = []
        if dets:
            boxes_px = [d["box_px"] for d in dets]
            labels = [d["label"] for d in dets]
            masks = segmenter.masks_from_boxes(rgb, boxes_px)
            depth_map = depther.estimate(Image.fromarray(rgb))
            contact = pair_contacts(masks)
            objs = objects_from(boxes_px, labels, masks, depth_map, w, h, z_scale)
            det2track = {i: tid for i, tid in enumerate(ids)}
            for p in annotate_objects(objs, cfg, contact):
                for k in p.predicates:
                    triplets.append([det2track[p.subject], k, det2track[p.object]])
        per_frame.append({
            "frame": fi, "time": round(fi / fps, 3),
            "tracks": {str(i): tid for i, tid in enumerate(ids)},
            "detections": dets, "triplets_raw": triplets,
        })
        frames_rgb.append((rgb, dets, ids))

    cap.release()

    # temporal smoothing + stability summary
    smoothed = smooth(per_frame, window=args.window)
    raw_sets = [set(map(tuple, f["triplets_raw"])) for f in per_frame]
    jr = [jaccard(raw_sets[i], raw_sets[i + 1]) for i in range(len(raw_sets) - 1)]
    js = [jaccard(smoothed[i], smoothed[i + 1]) for i in range(len(smoothed) - 1)]
    print(f"\nframe-to-frame triplet stability (mean Jaccard): "
          f"raw {np.mean(jr):.3f} -> smoothed {np.mean(js):.3f}")

    # names: label + track id, stable across the video
    names = {}
    for f, (rgb, dets, ids) in zip(per_frame, frames_rgb):
        for d, tid in zip(dets, ids):
            names.setdefault(tid, f"{d['label']}{tid}")

    for f, sm in zip(per_frame, smoothed):
        f["triplets_smoothed"] = sorted(list(t) for t in sm)
        f["triplets_named"] = sorted(f"{names[s]} -{k}-> {names[o]}"
                                     for s, k, o in sm)
    with open(out / "frames.jsonl", "w", encoding="utf-8") as fh:
        for f in per_frame:
            fh.write(json.dumps(f) + "\n")

    # overlay video + gif (smoothed triplets)
    print("rendering overlays ...")
    rendered = [np.array(render(rgb, dets, ids, sm, names))
                for (rgb, dets, ids), sm in zip(frames_rgb, smoothed)]
    hh, ww = rendered[0].shape[:2]
    vw = cv2.VideoWriter(str(out / "annotated.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"),
                         fps / args.stride, (ww, hh))
    for fr in rendered:
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    vw.release()

    gif_scale = 640 / ww
    from PIL import Image  # noqa: PLC0415
    gif = [Image.fromarray(fr).resize((640, int(hh * gif_scale)))
           for fr in rendered]
    gif[0].save(out / "annotated.gif", save_all=True, append_images=gif[1:],
                duration=int(1000 * args.stride / fps), loop=0)

    print(f"done: {len(per_frame)} frames -> {out}\\annotated.gif / .mp4 / frames.jsonl")


if __name__ == "__main__":
    main()
