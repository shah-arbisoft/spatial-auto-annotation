# Container verification

Image `spatial-annotator:latest`, built 2026-08-02, 7.02 GB, 11 layers,
sha256:98376b6a5299. Verified after the build host ran out of disk during
the export stage.

## Integrity
- 84,972 files read in full, 0 read errors (no truncated layers)
- `pip check` clean except `ninja` platform-metadata note (SAM2 build dep)

## Build-time
- pytest inside the build: 66 passed
- rule layer import: OK

## Runtime (finished image)
- python 3.11.10
- pytest inside the finished image: 66 passed
- torch 2.5.1+cu121, torchvision 0.20.1+cu121, transformers 4.44.2
- CUDA build 12.1 (not the CPU wheel the SAM2 pitfall produces)
- sam2, cv2, numpy, scipy, sklearn, PIL all import

## Contents
- /app = 2.0 MB; 60 .py/.yaml files hash-identical to the working tree
- absent: .env, .env.local, votes.csv, datasets/, .git/, outputs/cache,
  checkpoints/, .venv/, __pycache__/
- present and intended: .env.example, data/README.md

## GPU passthrough
`docker run --gpus all`: CUDA available True, 1 device, NVIDIA GeForce
RTX 2060, matmul executes on device (21.0 MB peak).

## End-to-end
Dataset and geometry cache mounted read-only, `reannotate_from_cache.py`:
- 836 images re-annotated
- pairs.csv byte-identical to repo (84,881 rows, matching SHA-256)
- 2,508 annotation files, matching repo count

## Not verified
- The full GPU annotation pass (`run_annotator.py`) inside the container.
  The GPU is reachable and torch computes on it; the pass itself was run on
  the host, which is how the cached geometry was produced.
- The build has not been re-run since the base image and SAM2 commit were
  pinned. The pins were read out of the verified image, so they record what
  was installed rather than changing it.
