"""Box-prompted segmentation with SAM2.

Given the detector's boxes, SAM2 returns one precise mask per object. We load a
small SAM2 variant from the Hugging Face hub (auto-downloads checkpoint + config,
no manual files) to fit the 6 GB RTX 2060.
"""

from __future__ import annotations

import numpy as np


class Segmenter:
    def __init__(self, hf_model: str = "facebook/sam2.1-hiera-small", device: str = "cuda"):
        self.hf_model = hf_model
        self.device = device
        self._predictor = None

    def load(self):
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: PLC0415
        try:
            self._predictor = SAM2ImagePredictor.from_pretrained(self.hf_model, device=self.device)
        except TypeError:
            # older signature without a device kwarg
            self._predictor = SAM2ImagePredictor.from_pretrained(self.hf_model)
        return self

    def masks_from_boxes(self, image, boxes) -> list[np.ndarray]:
        """Return one boolean mask per input box, in input order.

        Args:
            image: H x W x 3 RGB array (uint8) or PIL image.
            boxes: pixel (x1, y1, x2, y2) boxes; a single box or a list of boxes.
        Returns:
            list of H x W boolean masks.
        """
        import torch  # noqa: PLC0415

        if self._predictor is None:
            self.load()

        img = np.asarray(image)
        boxes_arr = np.asarray(boxes, dtype=float)
        if boxes_arr.ndim == 1:
            boxes_arr = boxes_arr[None, :]

        # Request multiple candidate masks per box and keep the highest-scoring
        # one. SAM2's single-mask mode can return an empty mask for a loose box;
        # multimask + best-by-score is the robust, recommended choice.
        with torch.inference_mode():
            self._predictor.set_image(img)
            masks, scores, _ = self._predictor.predict(
                point_coords=None, point_labels=None,
                box=boxes_arr, multimask_output=True,
            )

        masks = np.asarray(masks)
        scores = np.asarray(scores)
        if masks.ndim == 3:           # single box: (C, H, W) -> (1, C, H, W)
            masks = masks[None]
            scores = scores[None]

        out = []
        for i in range(masks.shape[0]):          # one entry per input box
            best = int(np.argmax(scores[i]))
            out.append(masks[i, best] > 0)        # threshold handles 0/1 or logits
        return out
