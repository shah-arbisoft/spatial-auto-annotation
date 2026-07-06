"""End-to-end orchestration: image + objects -> scene-graph triplets.

Perception (SAM2 masks + Depth Anything depth) runs on the GPU; the geometry and
predicate rules are pure NumPy and testable offline. The split is deliberate so
the graded core (geometry -> predicates) can be unit-tested without a GPU.

Two entry points:
  - run_perception(): GPU. image + pixel boxes -> (masks, depth_map)
  - objects_from() + annotate_objects(): CPU. masks + depth -> Obj -> predicates
  - annotate_image(): ties them together for one image.

For the fidelity study we feed ground-truth boxes/labels (PredCls setting), so an
Obj's index equals the dataset object index and pairs align with human triplets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from . import geometry
from .predicates import Obj, Thresholds, evaluate_scene


def load_config(path: str = "configs/default.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def thresholds_from_config(cfg: dict) -> Thresholds:
    p = cfg["predicates"]
    c = cfg["confidence"]
    return Thresholds(
        near_T=p["near_T"],
        on_vertical_gap=p["on_vertical_gap"],
        on_horizontal_overlap=p["on_horizontal_overlap"],
        on_depth_eps=p.get("on_depth_eps", 0.06),
        lateral_center_eps=p["lateral_center_eps"],
        depth_eps=p["depth_eps"],
        flag_near_band=c["flag_near_band"],
    )


# --------------------------------------------------------------------------- #
# CPU: geometry + rules (testable without a GPU)
# --------------------------------------------------------------------------- #
def rect_mask(box_px, height: int, width: int) -> np.ndarray:
    """Boolean mask filled inside a pixel box — the no-SAM2 fallback."""
    x1, y1, x2, y2 = (int(round(v)) for v in box_px)
    m = np.zeros((height, width), dtype=bool)
    m[max(0, y1):min(height, y2), max(0, x1):min(width, x2)] = True
    return m


def objects_from(boxes_px, labels, masks, depth_map, width, height, z_scale=1.0) -> list[Obj]:
    """Lift each (box, label, mask) to an Obj using the depth map."""
    return [
        geometry.lift(i, lab, box, mask, depth_map, width, height, z_scale=z_scale)
        for i, (box, lab, mask) in enumerate(zip(boxes_px, labels, masks))
    ]


def annotate_objects(objs: list[Obj], cfg: dict):
    """Run the predicate rules + correction over every ordered pair."""
    return evaluate_scene(
        objs, thresholds_from_config(cfg), correct=cfg["correction"]["enabled"]
    )


# --------------------------------------------------------------------------- #
# GPU: perception
# --------------------------------------------------------------------------- #
def run_perception(image_rgb: np.ndarray, boxes_px, segmenter, depther, use_sam2=True):
    """image (H,W,3 uint8 RGB) + pixel boxes -> (masks, depth_map)."""
    from PIL import Image  # noqa: PLC0415

    h, w = image_rgb.shape[:2]
    depth_map = depther.estimate(Image.fromarray(image_rgb))
    if use_sam2:
        masks = segmenter.masks_from_boxes(image_rgb, boxes_px)
    else:
        masks = [rect_mask(b, h, w) for b in boxes_px]
    return masks, depth_map


def annotate_image(image_rgb, boxes_px, labels, segmenter, depther, cfg, use_sam2=True):
    """Full pipeline for one image. Returns (objs, pair_results, depth_map)."""
    h, w = image_rgb.shape[:2]
    masks, depth_map = run_perception(image_rgb, boxes_px, segmenter, depther, use_sam2)
    z_scale = cfg.get("geometry", {}).get("z_scale", 1.0)
    objs = objects_from(boxes_px, labels, masks, depth_map, w, h, z_scale)
    pairs = annotate_objects(objs, cfg)
    return objs, pairs, depth_map
