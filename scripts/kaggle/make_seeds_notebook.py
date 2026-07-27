"""Generate the seed-replication Kaggle notebook as a real .ipynb.

Pasting cells out of a markdown runbook invites copying the ``` fences with
them, which fails immediately with a SyntaxError. Uploading a notebook file
removes that whole class of mistake, so this script writes the cells straight
into seed_replication.ipynb.

    python scripts/kaggle/make_seeds_notebook.py

Then in Kaggle: File > Import Notebook (or Create > New Notebook > File >
Upload), attach the two inputs, set the accelerator to GPU T4 x2, and use
Save Version > Save & Run All (Commit).
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).with_name("seed_replication.ipynb")

MD_INTRO = """\
# Benchmark seed replication (seeds 43 and 44, both arms)

The first benchmark run used one seed per arm, so the margins in chapter 6
are reported as observed rather than tested. This notebook adds two more
seeds to each arm, giving three runs per arm and a spread for every number.

**The detector is deliberately not retrained.** Both arms share one frozen
YOLOv8m backbone, which is what guarantees that only the relation labels
differ between them. Retraining it here would change that shared component
between the original seed and the new ones. Cell 4 copies it from the first
run's output instead.

## Before running

1. Accelerator: **GPU T4 x2** (P100 fails; Kaggle's torch dropped sm_60).
2. Add Input: the dataset **shah9212/spatial-sgg**.
3. Add Input: the committed **output of the original training notebook**
   (notebook-ssg), which holds `checkpoints/BACKBONES/yolov8m_spatial.pt`.
4. **Save Version > Save & Run All (Commit)**. Do not run cell by cell: an
   interactive session is wiped when it closes and takes the checkpoints
   with it.

Expected runtime ~1 h 45 m.
"""

CELLS: list[tuple[str, str]] = [
    ("markdown", MD_INTRO),

    ("markdown", "## Cell 1 — install the framework"),
    ("code", """\
%%bash
cd /kaggle/working && rm -rf SGG-Benchmark
git clone -q https://github.com/Maelic/SGG-Benchmark.git
cd SGG-Benchmark && pip install -e . -q
pip install -q ultralytics hydra-core omegaconf
echo "INSTALL DONE"
"""),

    ("markdown", "## Cell 2 — copy the dataset and config (auto-finds the upload)"),
    ("code", """\
%%bash
set -e
BASE=/kaggle/working/SGG-Benchmark
INPUT=$(dirname $(find /kaggle/input -name spatial_sgg_react.yaml | head -1))
echo "found data at: $INPUT"
mkdir -p $BASE/datasets $BASE/configs/hydra/Spatial
cp -r $INPUT/spatial_sgg $BASE/datasets/
cp -r $INPUT/spatial_sgg_yolo $BASE/datasets/
cp $INPUT/spatial_sgg_react.yaml $BASE/configs/hydra/Spatial/react.yaml
echo "DATA COPIED"; ls $BASE/datasets
"""),

    ("markdown",
     "## Cell 3 — background-index patch\\n\\n"
     "Index 0 is reserved for `__background__` / `__no_relation__`. Without "
     "this the framework silently eats class 0 and every mR is zero. "
     "Idempotent, so it is harmless if the upload is already fixed."),
    ("code", """\
import json, glob, os
os.chdir("/kaggle/working/SGG-Benchmark")
for p in sorted(glob.glob("datasets/spatial_sgg/*/_annotations.human.coco.json") +
                glob.glob("datasets/spatial_sgg/*/_annotations.auto.coco.json")):
    d = json.load(open(p))
    if any(c["name"] == "__background__" for c in d["categories"]):
        continue
    for c in d["categories"]:      c["id"] += 1
    for c in d["rel_categories"]:  c["id"] += 1
    for a in d["annotations"]:     a["category_id"]  += 1
    for r in d["rel_annotations"]: r["predicate_id"] += 1
    d["categories"].insert(0, {"id": 0, "name": "__background__", "supercategory": "none"})
    d["rel_categories"].insert(0, {"id": 0, "name": "__no_relation__"})
    json.dump(d, open(p, "w"))

d = json.load(open("datasets/spatial_sgg/test/_annotations.human.coco.json"))
assert d["categories"][0]["name"] == "__background__" and len(d["categories"]) == 7
assert d["rel_categories"][0]["name"] == "__no_relation__" and len(d["rel_categories"]) == 8
print("PATCH OK - 7 object classes (bg+6), 8 relation classes (norel+7)")
"""),

    ("markdown",
     "## Cell 4 — reuse the frozen detector from the first run\\n\\n"
     "No retraining: this is the shared component that makes the two arms "
     "comparable, so it must be the same weights the original run used."),
    ("code", """\
import glob, os, shutil
os.chdir("/kaggle/working/SGG-Benchmark")

hits = glob.glob("/kaggle/input/**/yolov8m_spatial.pt", recursive=True)
assert hits, ("detector checkpoint not found under /kaggle/input - attach the "
              "committed output of the original training notebook as an input")
os.makedirs("checkpoints/BACKBONES", exist_ok=True)
shutil.copy(hits[0], "checkpoints/BACKBONES/yolov8m_spatial.pt")
size = os.path.getsize("checkpoints/BACKBONES/yolov8m_spatial.pt") / 1e6
print(f"DETECTOR REUSED from {hits[0]}  ({size:.1f} MB)")
print("all arms and seeds share this frozen backbone, as in the first run")
"""),

    ("markdown",
     "## Cell 5 — train four runs: {human, auto} x {seed 43, seed 44}\\n\\n"
     "~12 min per human arm, ~26 min per auto arm. Each run asserts that "
     "validation mR is non-zero, so a broken setup fails in minutes rather "
     "than after the whole schedule."),
    ("code", """\
import subprocess, shutil, os, re, time
os.chdir("/kaggle/working/SGG-Benchmark")

def stage(variant):
    \"\"\"Point _annotations.coco.json at the chosen label source.
    Test is ALWAYS human gold in both arms: that is the design guarantee.\"\"\"
    for split in ["train", "val"]:
        shutil.copy(f"datasets/spatial_sgg/{split}/_annotations.{variant}.coco.json",
                    f"datasets/spatial_sgg/{split}/_annotations.coco.json")
    shutil.copy("datasets/spatial_sgg/test/_annotations.human.coco.json",
                "datasets/spatial_sgg/test/_annotations.coco.json")

for seed in [43, 44]:
    for variant in ["human", "auto"]:
        tag = f"react_{variant}_s{seed}"
        stage(variant)
        os.system(f"rm -rf checkpoints/spatial/{tag}")
        cmd = ("python -u tools/relation_train_net_hydra.py "
               "--config-path ../configs/hydra/Spatial --config-name react "
               f"--task sgdet --save-best seed={seed} "
               f"output_dir=./checkpoints/spatial/{tag}")
        t0 = time.time()
        print("=" * 78, f"\\nTRAIN {tag}\\n", "=" * 78, flush=True)
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        out = r.stdout + "\\n" + r.stderr
        print(out[-1500:], flush=True)
        mrs = [float(x) for x in re.findall(r"Result for mR:\\s*([\\d.]+)", out)]
        best = max(mrs) if mrs else 0.0
        print(f">>> {tag}: best val mR {best:.4f} in {(time.time()-t0)/60:.1f} min",
              flush=True)
        assert best > 0, f"{tag} produced mR=0 - check the class-index patch"

print("\\nALL FOUR TRAINING RUNS DONE")
"""),

    ("markdown",
     "## Cell 6 — evaluate all four checkpoints on the test set\\n\\n"
     "`--eval-only` is silently a no-op with this config: `eval_only()` zips "
     "`datasets.test`, which is undeclared here, and unlike the train/val "
     "path it applies no name fallback, so the loop never runs. "
     "`inference()` is therefore called directly, which is the recipe that "
     "produced the chapter 6 numbers."),
    ("code", """\
import subprocess, os, torch, logging, glob, shutil
os.chdir("/kaggle/working/SGG-Benchmark")
from omegaconf import OmegaConf
from sgg_benchmark.modeling.detector import build_detection_model
from sgg_benchmark.utils.checkpoint import DetectronCheckpointer
from sgg_benchmark.data import make_data_loader
from sgg_benchmark.engine.inference import inference

try:
    from sgg_benchmark.utils.logger import setup_logger
    logger = setup_logger("sgg_benchmark", ".", 0, verbose="INFO", steps=True)
except Exception:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("sgg_benchmark")

# every arm is scored against human gold
shutil.copy("datasets/spatial_sgg/test/_annotations.human.coco.json",
            "datasets/spatial_sgg/test/_annotations.coco.json")

for seed in [43, 44]:
    for variant in ["human", "auto"]:
        tag = f"react_{variant}_s{seed}"
        cfg = OmegaConf.load(f"checkpoints/spatial/{tag}/hydra_config.yaml")
        out = f"./checkpoints/spatial/eval_{tag}"
        os.makedirs(out, exist_ok=True)
        cfg.output_dir = out
        ckpt = sorted(glob.glob(f"checkpoints/spatial/{tag}/best_model_epoch_*.pth"))[-1]

        model = build_detection_model(cfg).to(cfg.model.device)
        DetectronCheckpointer(cfg, model).load(ckpt)
        model.eval()
        loader = make_data_loader(cfg, mode="test")[0]

        print("=" * 80, flush=True)
        print(f"EVAL {tag} | {os.path.basename(ckpt)} | test images: {len(loader.dataset)}",
              flush=True)
        print("=" * 80, flush=True)
        with torch.no_grad():
            inference(cfg, model, loader, dataset_name="SpatialRobot_test",
                      iou_types=("bbox", "relations"), box_only=False,
                      device=cfg.model.device, expected_results=[],
                      expected_results_sigma_tol=4, output_folder=out, logger=logger)

print("\\nALL EVALUATIONS DONE")
"""),

    ("markdown",
     "## Cell 7 — bundle the results for download\\n\\n"
     "Checkpoints are excluded to keep the archive small; the logs and "
     "result files are what the analysis needs."),
    ("code", """\
%%bash
cd /kaggle/working/SGG-Benchmark
zip -rq /kaggle/working/seed_results.zip checkpoints/spatial -x "*.pth" -x "*.pt"
echo "RESULTS -> /kaggle/working/seed_results.zip"
ls -la /kaggle/working/seed_results.zip
"""),

    ("markdown",
     "## After the run\\n\\n"
     "1. Committed version > **Output** tab > download `seed_results.zip`.\\n"
     "2. Unzip into `outputs/sgg_benchmark/seeds/` in the repository.\\n"
     "3. `python eval/seed_stats.py` prints mean and spread per arm and "
     "states whether the per-seed ranges overlap."),
]


def main():
    cells = []
    for kind, src in CELLS:
        lines = src.splitlines(keepends=True)
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            cells.append({"cell_type": "code", "execution_count": None,
                          "metadata": {}, "outputs": [], "source": lines})
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    n_code = sum(1 for k, _ in CELLS if k == "code")
    print(f"wrote {OUT}  ({len(cells)} cells, {n_code} code)")


if __name__ == "__main__":
    main()
