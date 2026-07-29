"""Object detectors: the pipeline's one replaceable input.

The rule layer never sees an image. It consumes boxes, labels and the masks
and depth derived from them, so any detector that can produce boxes with
class names can drive it. This module states that contract explicitly and
ships three implementations of it, so swapping in a different detector is a
configuration choice rather than a code change.

The contract is one method:

    detect(image_rgb) -> list[dict]

Each dict has:
    box_px : [x1, y1, x2, y2]  pixel coordinates, top-left origin
    label  : str               a dataset class name
    score  : float             confidence in [0, 1]

Boxes must be in the same upright frame as the image passed in. That is the
only requirement; nothing downstream inspects the detector.

Shipped implementations:

    GroundingDinoDetector  open-vocabulary, text-prompted (the deployment
                           default; no training needed, weakest per-class
                           recall, which is why chapter 4 treats it as the
                           worst case)
    YoloDetector           any ultralytics checkpoint, e.g. the YOLOv10m
                           weights reported in the source paper at 0.93
                           mAP@50; strongest option when weights exist
    JsonDetector           reads pre-computed detections from disk, so a
                           detector this project has never heard of can be
                           run separately and its output dropped in

Writing a fourth takes about ten lines: implement detect(), return the
dicts above. See docs/CUSTOM_DETECTOR.md for a worked example.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Detector(Protocol):
    """Anything that turns an RGB image into labelled boxes."""

    def detect(self, image_rgb: np.ndarray) -> list[dict]:
        """RGB uint8 (H, W, 3) -> [{'box_px': [x1,y1,x2,y2], 'label': str,
        'score': float}, ...]"""
        ...


# The dataset's six classes as text prompts. Short noun phrases work best
# with open-vocabulary detectors; the value is the class name to record, so
# "person" is stored as the dataset's "human".
DATASET_PROMPTS = {
    "book": "book", "bottle": "bottle", "box": "box",
    "cube": "cube", "person": "human", "remote control": "remote",
}


def _clip(box, w: int, h: int) -> list[float]:
    x1, y1, x2, y2 = box
    return [max(0.0, min(float(x1), w)), max(0.0, min(float(y1), h)),
            max(0.0, min(float(x2), w)), max(0.0, min(float(y2), h))]


class GroundingDinoDetector:
    """Open-vocabulary detection from text prompts (no training required).

    `prompts` maps the phrase given to the model onto the dataset class name
    it should be recorded as, which is how "person" becomes "human".
    """

    def __init__(self, prompts: dict[str, str] | None = None,
                 threshold: float = 0.30,
                 hf_model: str = "IDEA-Research/grounding-dino-tiny",
                 device: str = "cuda"):
        self.prompts = dict(prompts) if prompts else dict(DATASET_PROMPTS)
        self.threshold = threshold
        self.hf_model = hf_model
        self.device = device
        self._pipe = None

    def load(self):
        from transformers import pipeline as hf_pipeline  # noqa: PLC0415
        self._pipe = hf_pipeline("zero-shot-object-detection", model=self.hf_model,
                                 device=0 if self.device == "cuda" else -1)
        return self

    def detect(self, image_rgb: np.ndarray) -> list[dict]:
        from PIL import Image  # noqa: PLC0415
        if self._pipe is None:
            self.load()
        h, w = image_rgb.shape[:2]
        results = self._pipe(Image.fromarray(image_rgb),
                             candidate_labels=[f"{p}." for p in self.prompts],
                             threshold=self.threshold)
        out = []
        for r in results:
            label = self.prompts.get(r["label"].rstrip(".").strip())
            if label is None:
                continue
            b = r["box"]
            out.append({"box_px": _clip([b["xmin"], b["ymin"], b["xmax"], b["ymax"]], w, h),
                        "label": label, "score": round(float(r["score"]), 4)})
        return out


class YoloDetector:
    """Any ultralytics checkpoint, including the source paper's own weights.

    `class_map` renames the checkpoint's classes onto this dataset's six if
    they differ; omit it when the names already match.
    """

    def __init__(self, weights: str, threshold: float = 0.25,
                 class_map: dict[str, str] | None = None, device: str = "cuda"):
        self.weights = weights
        self.threshold = threshold
        self.class_map = class_map or {}
        self.device = device
        self._model = None

    def load(self):
        from ultralytics import YOLO  # noqa: PLC0415
        self._model = YOLO(self.weights)
        return self

    def detect(self, image_rgb: np.ndarray) -> list[dict]:
        if self._model is None:
            self.load()
        h, w = image_rgb.shape[:2]
        res = self._model.predict(image_rgb, conf=self.threshold, verbose=False,
                                  device=0 if self.device == "cuda" else "cpu")[0]
        names = res.names
        out = []
        for b in res.boxes:
            raw = names[int(b.cls)]
            out.append({"box_px": _clip(b.xyxy[0].tolist(), w, h),
                        "label": self.class_map.get(raw, raw),
                        "score": round(float(b.conf), 4)})
        return out


class JsonDetector:
    """Pre-computed detections, for a detector outside this codebase.

    Expects one JSON file per image under `root`, named after the image stem,
    holding either a bare list of detection dicts or {"detections": [...]}.
    Run any detector you like, write that file, and the pipeline consumes it
    with no knowledge of what produced it.
    """

    def __init__(self, root: str | Path, stem: str | None = None):
        self.root = Path(root)
        self.stem = stem

    def for_image(self, stem: str) -> "JsonDetector":
        return JsonDetector(self.root, stem)

    def detect(self, image_rgb: np.ndarray) -> list[dict]:
        if self.stem is None:
            raise ValueError("call for_image(stem) first so the detector knows "
                             "which file to read")
        path = self.root / f"{self.stem}.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        dets = data.get("detections", data) if isinstance(data, dict) else data
        h, w = image_rgb.shape[:2]
        return [{"box_px": _clip(d["box_px"], w, h),
                 "label": d["label"], "score": float(d.get("score", 1.0))}
                for d in dets]


# the config file predates this module and names things its own way
_LEGACY_KIND = {"yolov10m": "yolo", "yolov8m": "yolo", "groundingdino": "grounding_dino"}
_LEGACY_KEYS = {"conf_threshold": "threshold"}
_IGNORED = {"iou_nms"}          # handled inside ultralytics, not our concern


def from_config(cfg: dict):
    """Build the detector described by a config's `detector` block.

    detector:
      kind: grounding_dino | yolo | json
      ...kind-specific keys...

    The older spelling (`backend: yolov10m`, `conf_threshold`) is still
    accepted so existing configs keep working.
    """
    spec = dict(cfg.get("detector") or {})
    kind = spec.pop("kind", None) or _LEGACY_KIND.get(spec.pop("backend", ""), None) \
        or "grounding_dino"
    spec = {_LEGACY_KEYS.get(k, k): v for k, v in spec.items() if k not in _IGNORED}

    builders = {"grounding_dino": GroundingDinoDetector,
                "yolo": YoloDetector, "json": JsonDetector}
    if kind not in builders:
        raise ValueError(f"unknown detector kind {kind!r}; expected one of "
                         f"{', '.join(builders)}, or pass your own object "
                         "implementing detect(image_rgb)")
    if kind == "grounding_dino":
        spec.pop("weights", None)   # not meaningful for a prompt-driven model
    return builders[kind](**spec)
