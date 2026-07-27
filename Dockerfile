# Reproducibility image for the spatial-relationship annotator.
#
# Mirrors the environment every number in the dissertation was produced on:
# Python 3.11, torch 2.5.1 + cu121, SAM2 from GitHub, and the pinned
# requirements. Model weights auto-download from Hugging Face on first use,
# so the image stays small; mount a cache volume to keep them between runs.
#
# Build:
#   docker build -t spatial-annotator .
#
# GPU run (full annotation pass, ~5 min on a 6 GB card):
#   docker run --gpus all \
#     -v /path/to/SpatialAwareRobotDataset:/data \
#     -v $(pwd)/outputs:/app/outputs \
#     -v hf-cache:/root/.cache/huggingface \
#     spatial-annotator python scripts/run_annotator.py
#
# CPU-only is enough for everything downstream of the caches (the rule layer,
# every eval script, the figures, the tests):
#   docker run -v $(pwd)/outputs:/app/outputs spatial-annotator pytest -q
#   docker run -v $(pwd)/outputs:/app/outputs spatial-annotator \
#     python scripts/reannotate_from_cache.py
#
# The dataset itself is not baked in (CC-BY, ~1 GB): mount it at /data and
# point configs/default.yaml's dataset.root there, or override on the CLI.

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# git for the SAM2 install; the OpenCV runtime libraries for cv2
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 zip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# SAM2 first, then force the CUDA torch back if its resolver replaced it
# (the known pitfall documented in requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir "git+https://github.com/facebookresearch/sam2.git" \
    && pip install --no-cache-dir --no-deps --force-reinstall \
        torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# quick self-check at build time: the rule layer needs no GPU
RUN python -m pytest -q tests/ && python -c "import src.predicates, src.pipeline; print('rule layer OK')"

CMD ["python", "scripts/smoke_test.py", "--image", "assets/sample.jpg"]
