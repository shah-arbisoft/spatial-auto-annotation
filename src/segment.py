"""Box-prompted segmentation with SAM2.

Given the detector's boxes, SAM2 returns one precise mask per object. We use a
small SAM2 variant to fit the 6 GB RTX 2060. Implementation deferred to Week 2;
interface fixed here.
"""

from __future__ import annotations

import numpy as np


class Segmenter:
    def __init__(self, variant: str, checkpoint: str, device: str = "cuda"):
        self.variant = variant
        self.checkpoint = checkpoint
        self.device = device
        self._predictor = None

    def load(self):
        # from sam2.sam2_image_predictor import SAM2ImagePredictor
        # from sam2.build_sam import build_sam2
        # model = build_sam2(<config>, self.checkpoint, device=self.device)
        # self._predictor = SAM2ImagePredictor(model)
        raise NotImplementedError("SAM2 load — Week 2 (see scripts/smoke_test.py)")

    def masks_from_boxes(
        self, image: np.ndarray, boxes: list[tuple[float, float, float, float]]
    ) -> list[np.ndarray]:
        """Return one boolean mask per input box, in input order.

        Args:
            image: H x W x 3 RGB array.
            boxes: pixel (x1, y1, x2, y2) boxes.
        Returns:
            list of H x W boolean masks.
        """
        if self._predictor is None:
            self.load()
        # self._predictor.set_image(image)
        # masks, _, _ = self._predictor.predict(box=np.array(boxes), multimask_output=False)
        raise NotImplementedError("SAM2 predict — Week 2")
