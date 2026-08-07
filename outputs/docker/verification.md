# Container verification

Image `spatial-annotator:latest`, rebuilt 2026-08-07, 7.02 GB content,
sha256:ce49e9ed1cdc. This supersedes the 2026-08-02 image
(sha256:98376b6a5299), which was verified in the same way but predates the
predicate-specification corrections and did not carry the geometry cache.

Both builds are recorded because the first attempt at this rebuild failed in
a way worth documenting: the layers exported and the tag was written, then
the builder died during `unpacking` when the host disk filled. The tag
existed and `docker images` listed a plausible size, so the failure was only
visible in the build's own exit status. It is the same failure the 2026-08-02
build hit during export, and the reason this file verifies a finished image
rather than trusting a build log.

## Build-time
- pytest inside the build: 66 passed
- rule layer import: OK
- no `ERROR:` after the naming step (checked; this is what the first attempt
  failed)

## Runtime (finished image)
- python 3.11.10
- pytest inside the finished image: **66 passed**
- torch 2.5.1+cu121, CUDA build 12.1 (not the CPU wheel the SAM2 pitfall
  produces)
- sam2, cv2, numpy, scipy, sklearn, PIL, transformers all import

## Contents
- `/app` = 19.3 MB, of which the geometry cache is 1,672 files
- present and intended: `outputs/geometry/`, `outputs/pairs.csv`,
  `outputs/fidelity_report.json`, `.env.example`
- absent: `.env`, `.env.local`, `votes.csv`, `outputs/annotations/`,
  `outputs/failure_gallery/`, `outputs/figures/`, `outputs/video/`,
  `datasets/`, `.git/`
- the current predicate specification is inside the image, checked on three
  strings that changed in the last revision

## Reproduction, which is the point of the image
With the dataset mounted read-only and nothing else:

    docker run --rm -v /path/to/SpatialAwareRobotDataset-main:/ds:ro \
      spatial-annotator sh -c \
      'sed -i "s|root: .*|root: /ds/SpatialAwareRobotDataset-main|" configs/default.yaml
       python scripts/reannotate_from_cache.py'

produces `outputs/pairs.csv` with **84,881 rows** and SHA-256
`60281435122944fc243f3ca252be8f800867cb8a17413f9e43ccd1a4e220e1bd`,
**byte-identical to the file committed in this repository**. The container
therefore does not merely install; it reproduces the annotations this
dissertation reports.

With **nothing mounted at all**, `eval/fidelity.py`, `eval/seed_stats.py`,
`eval/score_planner.py` and `eval/compare_vlm_models.py` all run to
completion from the committed caches. The four commands that iterate the
released images rather than the cache index still need `annotated_data/`
mounted; they never open a JPEG (Appendix B, step 3).

## GPU passthrough
`docker run --gpus all`: CUDA available True, NVIDIA GeForce RTX 2060
enumerated, 512x512 matmul executes on the device (10.6 MB peak).
