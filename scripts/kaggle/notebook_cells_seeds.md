# Seed replication — two extra seeds per arm (Save & Run All ready)

The Week-7 benchmark ran one seed per arm, so the per-group margins in
chapter 6 (notably group 7, 0.334 vs 0.323) are reported as observed rather
than tested. This notebook adds seeds 43 and 44 to both arms, giving three
runs per arm and therefore a mean and a spread for every number in §6.3.

**The detector is not retrained.** Both arms share one frozen YOLOv8m
backbone; that is the design guarantee that only the relation labels differ.
Retraining it here would silently change the shared component between the old
seed and the new ones, so the previous run's checkpoint is reused. That also
saves ~15 minutes.

Runtime: ~1 h 45 m on **T4 x2** (~12 min per human arm, ~26 min per auto arm,
~4 min per evaluation). Well inside the session limit.

## Setup before pasting cells

1. New notebook, **Accelerator: GPU T4 x2** (P100 fails: Kaggle's torch
   dropped Pascal sm_60).
2. **Add Input** twice:
   - the dataset `shah9212/spatial-sgg` (the 71.5 MB upload), and
   - the **committed output of the original training notebook** (`notebook-ssg`),
     which holds `checkpoints/BACKBONES/yolov8m_spatial.pt`.
3. Paste the cells below in order.
4. **Save Version → Save & Run All (Commit).** Do not run cell-by-cell: an
   interactive session is wiped when it closes and takes the checkpoints with
   it.

---

## Cell 1 — install

```bash
%%bash
cd /kaggle/working && rm -rf SGG-Benchmark
git clone -q https://github.com/Maelic/SGG-Benchmark.git
cd SGG-Benchmark && pip install -e . -q
pip install -q ultralytics hydra-core omegaconf
echo "INSTALL DONE"
```

## Cell 2 — copy data + config

```bash
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
```

## Cell 3 — background-index patch (idempotent)

```python
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
print("PATCH OK — 7 object classes (bg+6), 8 relation classes (norel+7)")
```

## Cell 4 — reuse the frozen detector from the first run (no retraining)

```python
import glob, os, shutil
os.chdir("/kaggle/working/SGG-Benchmark")
hits = glob.glob("/kaggle/input/**/yolov8m_spatial.pt", recursive=True)
assert hits, ("detector checkpoint not found under /kaggle/input — attach the "
              "committed output of the original training notebook as an input")
os.makedirs("checkpoints/BACKBONES", exist_ok=True)
shutil.copy(hits[0], "checkpoints/BACKBONES/yolov8m_spatial.pt")
size = os.path.getsize("checkpoints/BACKBONES/yolov8m_spatial.pt") / 1e6
print(f"DETECTOR REUSED from {hits[0]}  ({size:.1f} MB)")
print("both arms and all seeds share this frozen backbone, as in the first run")
```

## Cell 5 — train four runs: {human, auto} x {seed 43, seed 44}

```python
import subprocess, shutil, os, re, time
os.chdir("/kaggle/working/SGG-Benchmark")

def stage(variant):
    """Point _annotations.coco.json at the chosen label source.
    Test is ALWAYS human gold, in both arms: that is the design guarantee."""
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
        print("=" * 78, f"\nTRAIN {tag}\n", "=" * 78, flush=True)
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        out = r.stdout + "\n" + r.stderr
        print(out[-1500:], flush=True)
        mrs = [float(x) for x in re.findall(r"Result for mR:\s*([\d.]+)", out)]
        print(f">>> {tag}: best val mR {max(mrs) if mrs else 0:.4f} "
              f"in {(time.time()-t0)/60:.1f} min", flush=True)
        assert mrs and max(mrs) > 0, f"{tag} produced mR=0 — check the class-index patch"
print("\nALL FOUR TRAINING RUNS DONE")
```

## Cell 6 — evaluate all four checkpoints on the test set

The framework's `--eval-only` path is silently a no-op with this config
(`datasets.test` is undeclared and, unlike the train/val path, gets no name
fallback), so `inference()` is called directly. This is the recipe that
produced the chapter 6 numbers.

```python
import subprocess, os, torch, logging, glob, json
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

# test set is human gold for every arm
import shutil
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
        print(f"EVAL {tag} | {os.path.basename(ckpt)} | test images: {len(loader.dataset)}", flush=True)
        print("=" * 80, flush=True)
        with torch.no_grad():
            inference(cfg, model, loader, dataset_name="SpatialRobot_test",
                      iou_types=("bbox", "relations"), box_only=False,
                      device=cfg.model.device, expected_results=[],
                      expected_results_sigma_tol=4, output_folder=out, logger=logger)
print("\nALL EVALUATIONS DONE")
```

## Cell 7 — bundle everything for download

```bash
%%bash
cd /kaggle/working/SGG-Benchmark
zip -rq /kaggle/working/seed_results.zip checkpoints/spatial -x "*.pth" -x "*.pt"
echo "RESULTS -> /kaggle/working/seed_results.zip"
ls -la /kaggle/working/seed_results.zip
```

---

## After the run

1. Download `seed_results.zip` from the committed version's **Output** tab.
2. Unzip it into `outputs/sgg_benchmark/seeds/` in the repository.
3. Run `python eval/seed_stats.py`, which reads every arm and seed, prints
   mean ± spread for R@100 / mR@100 / zR@100 and for the per-group slices, and
   writes `outputs/tables/seed_replication.md`.

The number that matters most is the group-7 margin. If the auto arm's lead
survives across three seeds it becomes a finding; if the seeds straddle the
human arm, §6.4 says so plainly and the zero-shot gap carries the argument
instead. Either outcome is reportable, which is why the run is worth doing.
