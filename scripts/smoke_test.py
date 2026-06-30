"""Week 1 smoke test: confirm Depth Anything v2 Small and SAM2 run on a sample.

Run this after creating a 3.11/3.12 venv and installing requirements.txt
(including the CUDA torch build and SAM2 from git). It is intentionally minimal:
it loads each perception model, runs it on one image, and prints output shapes
and GPU memory so you know the 6 GB RTX 2060 is sufficient.

    python scripts/smoke_test.py --image assets/sample.jpg

Exit code 0 means both models loaded and produced output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def log(msg: str) -> None:
    print(f"[smoke] {msg}", flush=True)


def check_torch_cuda() -> str:
    import torch
    log(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    log("WARNING: CUDA not available — falling back to CPU (slow).")
    return "cpu"


def check_depth(image_path: str, device: str) -> None:
    from PIL import Image
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.depth import DepthEstimator

    log("loading Depth Anything v2 Small ...")
    est = DepthEstimator(device=device).load()
    img = Image.open(image_path).convert("RGB")
    depth = est.estimate(img)
    log(f"depth map: shape={depth.shape}, min={depth.min():.3f}, max={depth.max():.3f} OK")


def check_sam2(image_path: str, device: str) -> None:
    from PIL import Image
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.segment import Segmenter

    log("loading SAM2 (facebook/sam2.1-hiera-small) ...")
    seg = Segmenter(device=device).load()

    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    # a central box prompt (no detector in the smoke test) just to confirm output
    box = [w * 0.3, h * 0.3, w * 0.7, h * 0.85]
    masks = seg.masks_from_boxes(img, box)
    m = masks[0]
    area = int(m.sum())
    log(f"SAM2 mask: shape={m.shape}, dtype={m.dtype}, covered px={area}")
    if area == 0:
        # diagnostic: empty mask — inspect raw predictor output (shapes + scores)
        import torch
        log("  mask empty; running diagnostic predict (multimask) ...")
        with torch.inference_mode():
            seg._predictor.set_image(img)
            raw, scores, _ = seg._predictor.predict(
                box=np.array([box], dtype=float), multimask_output=True
            )
        raw = np.asarray(raw)
        log(f"  raw shape={raw.shape}, dtype={raw.dtype}, scores={np.asarray(scores).ravel()}")
        flat = raw.reshape(-1, raw.shape[-2], raw.shape[-1])
        log(f"  per-mask covered px={[int((flat[i] > 0).sum()) for i in range(flat.shape[0])]}")
    else:
        log("  SAM2 OK")
    if device == "cuda":
        import torch
        log(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="assets/sample.jpg")
    args = ap.parse_args()

    if not Path(args.image).exists():
        log(f"ERROR: sample image not found: {args.image}")
        log("Drop any indoor RGB image at that path and re-run.")
        return 2

    device = check_torch_cuda()
    check_depth(args.image, device)
    check_sam2(args.image, device)
    log("smoke test complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
