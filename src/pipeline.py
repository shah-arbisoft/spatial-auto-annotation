"""End-to-end orchestration: RGB image -> scene-graph triplets.

detect -> segment -> depth -> lift -> predicates(+correction+flags) -> writers.

This is the thin glue the project is *not* graded on; the value is in
predicates.py and the rules. Full wiring lands in Week 2 once the perception
models are confirmed running (scripts/smoke_test.py). The structure below shows
the intended data flow and the single config that controls every threshold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from . import geometry
from .predicates import Thresholds, evaluate_scene


def load_config(path: str = "configs/default.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def thresholds_from_config(cfg: dict) -> Thresholds:
    p = cfg["predicates"]
    c = cfg["confidence"]
    return Thresholds(
        near_T=p["near_T"],
        on_vertical_gap=p["on_vertical_gap"],
        on_horizontal_overlap=p["on_horizontal_overlap"],
        lateral_center_eps=p["lateral_center_eps"],
        depth_eps=p["depth_eps"],
        flag_near_band=c["flag_near_band"],
    )


def annotate_image(image: np.ndarray, detector, segmenter, depther, cfg: dict):
    """Run the full pipeline on one image and return (objects, pair_results).

    Wiring sketch (completed Week 2):
        dets   = detector.detect(image)
        masks  = segmenter.masks_from_boxes(image, [d.box for d in dets])
        depth  = depther.estimate(image)
        objs   = [geometry.lift(i, d.label, d.box, m, depth, W, H, z_scale)
                  for i, (d, m) in enumerate(zip(dets, masks))]
        pairs  = evaluate_scene(objs, thresholds_from_config(cfg),
                                correct=cfg["correction"]["enabled"])
    """
    raise NotImplementedError("end-to-end wiring — Week 2")
