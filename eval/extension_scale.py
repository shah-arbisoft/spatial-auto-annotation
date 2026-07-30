"""What the pipeline produces on robot captures nobody has labelled.

Chapter 9 listed scale as demonstrated in principle rather than in practice:
throughput and density were measured on the 836 annotated images, and the
claim that the method extends to new captures was an extrapolation. The
supervising group subsequently supplied the full 2,650-frame sequence those
images were cut from, of which frames 000884 onward carry no annotation of
any kind. Running the annotator over them turns the extrapolation into a
measurement.

There is no ground truth here, so nothing about accuracy can be claimed. What
can be measured is the operational half: how many frames a content-adaptive
pass actually has to process, what label density results, and whether the
predicate distribution resembles the annotated portion or drifts, drift being
the signature of a pipeline that has quietly stopped working on unfamiliar
input.

    python eval/extension_scale.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path

PREDICATES = ["on", "under", "to the left of", "to the right of",
              "in front of", "behind", "near"]


def summarise(paths: list[str]) -> dict:
    preds: Counter = Counter()
    labels: Counter = Counter()
    per_image, dets_per_image = [], []
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        trips = d.get("triplets", [])
        per_image.append(len(trips))
        dets_per_image.append(len(d.get("detections", [])))
        for det in d.get("detections", []):
            labels[det["label"]] += 1
        for t in trips:
            preds[t[1] if isinstance(t, (list, tuple)) else t.get("predicate")] += 1
    total = sum(preds.values())
    return {
        "images": len(paths),
        "triplets": total,
        "triplets_per_image": total / len(paths) if paths else 0.0,
        "detections_per_image": (sum(dets_per_image) / len(paths)
                                 if paths else 0.0),
        "images_with_no_triplet": sum(1 for n in per_image if n == 0),
        "predicate_share": {p: preds.get(p, 0) / total if total else 0.0
                            for p in PREDICATES},
        "predicate_counts": {p: preds.get(p, 0) for p in PREDICATES},
        "class_counts": dict(labels.most_common()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/extension")
    ap.add_argument("--total-frames", type=int, default=1766,
                    help="frames the keyframe pass was selected from")
    ap.add_argument("--out", default="outputs/extension_scale.json")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "*.json")))
    if not paths:
        raise SystemExit(f"no annotations in {args.dir}")
    s = summarise(paths)
    s["source_frames"] = args.total_frames
    s["frames_skipped"] = args.total_frames - s["images"]
    s["compression"] = args.total_frames / s["images"] if s["images"] else None

    # wall-clock from the written files; the first has no predecessor to time
    ts = sorted(os.path.getmtime(p) for p in paths)
    if len(ts) > 1:
        s["seconds_per_image"] = (ts[-1] - ts[0]) / (len(ts) - 1)
        s["images_per_hour"] = 3600 / s["seconds_per_image"]

    print(f"annotated {s['images']} keyframes of {s['source_frames']} raw "
          f"frames ({s['compression']:.1f}x fewer perception passes)")
    print(f"{s['triplets']} triplets, {s['triplets_per_image']:.1f} per image, "
          f"{s['detections_per_image']:.1f} detections per image")
    print(f"{s['images_with_no_triplet']} images produced no triplet")
    if "seconds_per_image" in s:
        print(f"{s['seconds_per_image']:.1f} s/image including overlay "
              f"rendering ({s['images_per_hour']:.0f} images/hour)")
    print("\npredicate share:")
    for p in PREDICATES:
        print(f"  {p:16s} {s['predicate_share'][p]:6.3f}  "
              f"({s['predicate_counts'][p]})")
    print("\ndetected classes:", s["class_counts"])

    Path(args.out).write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
