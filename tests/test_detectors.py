"""The detector contract: anything with detect() can drive the pipeline.

These tests pin the extension point documented in docs/CUSTOM_DETECTOR.md.
They deliberately use a hand-written fake detector rather than a real model,
because the point being tested is that the pipeline does not care what
produced the boxes.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.detectors import (DATASET_PROMPTS, Detector, GroundingDinoDetector,
                           JsonDetector, YoloDetector, from_config)


class FakeDetector:
    """A detector implemented from the documentation alone."""

    def __init__(self, dets):
        self._dets = dets

    def detect(self, image_rgb):
        return list(self._dets)


def test_a_hand_written_class_satisfies_the_protocol():
    assert isinstance(FakeDetector([]), Detector)


def test_shipped_detectors_satisfy_the_protocol():
    # constructed but not loaded: no weights are downloaded here
    assert isinstance(GroundingDinoDetector(), Detector)
    assert isinstance(YoloDetector("weights.pt"), Detector)
    assert isinstance(JsonDetector("."), Detector)


def test_grounding_dino_defaults_to_the_dataset_classes():
    d = GroundingDinoDetector()
    assert d.prompts == DATASET_PROMPTS
    assert d.prompts["person"] == "human", "person must be recorded as human"


def test_from_config_selects_the_named_kind():
    assert isinstance(from_config({}), GroundingDinoDetector)  # default
    assert isinstance(from_config({"detector": {"kind": "grounding_dino"}}),
                      GroundingDinoDetector)
    assert isinstance(from_config({"detector": {"kind": "yolo", "weights": "w.pt"}}),
                      YoloDetector)
    assert isinstance(from_config({"detector": {"kind": "json", "root": "."}}),
                      JsonDetector)


def test_from_config_rejects_an_unknown_kind_with_a_useful_message():
    with pytest.raises(ValueError, match="grounding_dino"):
        from_config({"detector": {"kind": "not-a-detector"}})


def test_json_detector_reads_precomputed_boxes(tmp_path):
    (tmp_path / "frame.json").write_text(json.dumps(
        {"detections": [{"box_px": [1, 2, 30, 40], "label": "cube", "score": 0.7}]}),
        encoding="utf-8")
    dets = JsonDetector(tmp_path).for_image("frame").detect(np.zeros((80, 80, 3), np.uint8))
    assert dets == [{"box_px": [1.0, 2.0, 30.0, 40.0], "label": "cube", "score": 0.7}]


def test_json_detector_accepts_a_bare_list(tmp_path):
    (tmp_path / "frame.json").write_text(json.dumps(
        [{"box_px": [0, 0, 5, 5], "label": "book"}]), encoding="utf-8")
    dets = JsonDetector(tmp_path).for_image("frame").detect(np.zeros((10, 10, 3), np.uint8))
    assert dets[0]["score"] == 1.0, "score should default when absent"


def test_json_detector_is_silent_about_missing_files(tmp_path):
    assert JsonDetector(tmp_path).for_image("nope").detect(
        np.zeros((4, 4, 3), np.uint8)) == []


def test_boxes_are_clipped_to_the_image():
    """A detector reporting boxes past the edge must not move objects."""
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp())
    (d / "f.json").write_text(json.dumps(
        [{"box_px": [-5, -5, 500, 500], "label": "box", "score": 1.0}]), encoding="utf-8")
    got = JsonDetector(d).for_image("f").detect(np.zeros((20, 30, 3), np.uint8))
    assert got[0]["box_px"] == [0.0, 0.0, 30.0, 20.0]


def test_a_custom_detector_drives_the_rule_layer():
    """The end the contract exists for: foreign boxes produce triplets."""
    from src.pipeline import annotate_objects, load_config, objects_from
    from src.predicates import PREDICATES

    cfg = load_config("configs/default.yaml")
    # a cube resting on a book, invented by a detector this project never saw
    detector = FakeDetector([
        {"box_px": [40, 60, 60, 80], "label": "cube", "score": 0.9},
        {"box_px": [30, 80, 70, 95], "label": "book", "score": 0.9},
    ])
    image = np.zeros((100, 100, 3), np.uint8)
    dets = detector.detect(image)
    depth = np.full((100, 100), 0.5, dtype=float)
    masks = []
    for d in dets:                     # rectangular stand-ins for real masks
        m = np.zeros((100, 100), bool)
        x1, y1, x2, y2 = (int(v) for v in d["box_px"])
        m[y1:y2, x1:x2] = True
        masks.append(m)
    objs = objects_from([d["box_px"] for d in dets], [d["label"] for d in dets],
                        masks, depth, 100, 100)
    pairs = annotate_objects(objs, cfg)
    emitted = {p for pair in pairs for p in pair.predicates}
    assert emitted, "a conforming detector must yield relations"
    assert emitted <= set(PREDICATES), "only the seven predicates may be emitted"


def test_the_projects_own_config_builds_a_detector():
    """The shipped config predates this module; it must still work."""
    from src.pipeline import load_config
    d = from_config(load_config("configs/default.yaml"))
    assert isinstance(d, YoloDetector), "backend: yolov10m should map to the YOLO adapter"
    assert d.threshold == 0.25, "conf_threshold should map onto threshold"


def test_legacy_backend_names_are_accepted():
    assert isinstance(from_config({"detector": {"backend": "groundingdino"}}),
                      GroundingDinoDetector)
    assert isinstance(from_config({"detector": {"backend": "yolov8m", "weights": "w.pt"}}),
                      YoloDetector)
