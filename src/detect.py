"""Object detection wrapper: YOLOv10m (trained on the 900) or GroundingDINO.

Thin orchestration over the model's own Python API. Returns a uniform list of
Detection records so the rest of the pipeline is detector-agnostic (this also
makes the YOLOv10m-vs-GroundingDINO ablation a one-line backend swap).

Implementation is deferred to Week 2; the interface and the YOLO path are
sketched here so pipeline.py can be wired against a stable contract.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]  # pixel (x1, y1, x2, y2)


class Detector:
    def __init__(self, backend: str, weights: str, conf_threshold: float = 0.25,
                 iou_nms: float = 0.7, device: str = "cuda"):
        self.backend = backend
        self.weights = weights
        self.conf_threshold = conf_threshold
        self.iou_nms = iou_nms
        self.device = device
        self._model = None

    def load(self):
        if self.backend == "yolov10m":
            from ultralytics import YOLO  # noqa: PLC0415
            self._model = YOLO(self.weights)
        elif self.backend == "groundingdino":
            raise NotImplementedError("GroundingDINO backend — Week 4 detector swap")
        else:
            raise ValueError(f"unknown detector backend: {self.backend}")
        return self

    def detect(self, image) -> list[Detection]:
        """Run detection on a single image (np.ndarray or path)."""
        if self._model is None:
            self.load()
        if self.backend == "yolov10m":
            results = self._model.predict(
                image, conf=self.conf_threshold, iou=self.iou_nms,
                device=self.device, verbose=False,
            )[0]
            dets: list[Detection] = []
            names = results.names
            for b in results.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                dets.append(Detection(
                    label=names[int(b.cls)],
                    confidence=float(b.conf),
                    box=(x1, y1, x2, y2),
                ))
            return dets
        raise NotImplementedError(self.backend)
