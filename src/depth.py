"""Monocular relative depth with Depth Anything v2 (Small variant).

Small variant is Apache-2.0 and fits the 6 GB GPU. Returns a relative
(per-image, ordinal) depth map. We treat smaller values as nearer the camera;
callers must keep all depth comparisons within a single image.

The HF `transformers` pipeline path is the simplest way to run the Small model
and is what scripts/smoke_test.py uses. Full integration is Week 2.
"""

from __future__ import annotations

import numpy as np


class DepthEstimator:
    def __init__(self, hf_model: str = "depth-anything/Depth-Anything-V2-Small-hf",
                 device: str = "cuda"):
        self.hf_model = hf_model
        self.device = device
        self._pipe = None

    def load(self):
        from transformers import pipeline  # noqa: PLC0415
        self._pipe = pipeline(
            task="depth-estimation",
            model=self.hf_model,
            device=0 if self.device == "cuda" else -1,
        )
        return self

    def estimate(self, image) -> np.ndarray:
        """Return a relative depth map (H x W float array), SMALLER = NEARER.

        The HF `depth` output has LARGER = nearer the camera, so we invert it to
        match the smaller = nearer convention in docs/predicate_spec.md and
        src/predicates.py, then min-max normalise to [0, 1] (order preserved).

        Verified on EXIF-corrected (upright) images: without inversion,
        front/behind agreement with the human labels is ~26%; with inversion,
        ~74%. (An earlier check that suggested the opposite was run on images fed
        to the model upside-down — see src.dataset.load_rgb — and was wrong.)
        """
        if self._pipe is None:
            self.load()
        out = self._pipe(image)
        depth = np.asarray(out["depth"], dtype=float)
        depth = depth.max() - depth          # HF output is larger=nearer -> invert
        rng = depth.max() - depth.min()
        if rng > 0:
            depth = (depth - depth.min()) / rng
        return depth
