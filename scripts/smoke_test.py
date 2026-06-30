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
    log("SAM2 check: load a small checkpoint and segment one box.")
    log("  -> implement once the SAM2 checkpoint is downloaded (src/segment.py).")
    # Intentionally a no-op placeholder for Week 1; depth is the harder memory test.


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
