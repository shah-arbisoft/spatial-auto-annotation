# Third arm — REACT++ trained on vision-language labels (Save & Run All ready)

Chapters 5 and 6 compare human labels against this tool's. A vision-language
model is a third source that a reader will reasonably ask about, and the local
RQ2 classifier already answers it at small scale (mean recall 0.38 against the
pipeline's 0.76). This notebook runs the same question through the source
paper's own benchmark, so the answer is stated in the metric Chapter 6 uses.

Three seeds, matching the other two arms, so the arms are comparable as
distributions rather than as single runs.

**The detector is not retrained.** All three arms share one frozen YOLOv8m
backbone. That is the design guarantee that only the relation labels differ,
so this notebook reuses the checkpoint the first run committed rather than
building a new one.

Relation counts the arms train on, for expectation setting:

| variant | train | val | test |
|---|---|---|---|
| human | 5,421 | 687 | 2,818 (human) |
| auto (pipeline) | 119,020 | 18,084 | 2,818 (human) |
| **vlm (Gemini)** | **14,626** | **4,465** | 2,818 (human) |

The VLM arm is about 2.7× the human arm's supervision and about an eighth of
the pipeline's, so expect a runtime near the human arm's: roughly 15 minutes
per seed on T4 x2, about an hour for all three including evaluation.

## Setup before pasting cells

1. **Re-export and re-upload the dataset.** The existing Kaggle dataset has
   only the human and auto variants; the VLM one is new.

       python scripts/export_sgg_benchmark.py --vlm-replies outputs/vlm_pilot/replies_train_f35.jsonl
       python scripts/export_yolo_det.py
       cd datasets
       python -c "import zipfile; from pathlib import Path; z = zipfile.ZipFile('spatial_sgg_upload.zip','w',zipfile.ZIP_DEFLATED); [z.write(p, p.as_posix()) for root in ('spatial_sgg','spatial_sgg_yolo') for p in sorted(Path(root).rglob('*')) if p.is_file()]; z.write('spatial_sgg_react.yaml','spatial_sgg_react.yaml'); z.close()"

   Confirm the export prints no coverage warning and that
   `_annotations.vlm.coco.json` exists in all three splits. Upload as a new
   version of the `spatial-sgg` dataset.

2. New notebook, **Accelerator: GPU T4 x2** (P100 fails: Kaggle's torch
   dropped Pascal sm_60).

3. **Add Input** twice: the updated `spatial-sgg` dataset, and the committed
   output of the **original** training notebook for
   `checkpoints/BACKBONES/yolov8m_spatial.pt`. That is the notebook whose
   last cell writes `results.zip`, not `seed_results.zip` or
   `reeval_results.zip`; the later two reuse the detector rather than
   producing it.

   Inputs may mount at `/kaggle/input/<name>/` or at
   `/kaggle/input/datasets/<user>/<name>/` depending on account age. Check
   with `ls /kaggle/input/` and `find /kaggle/input -maxdepth 3` before
   assuming either. Cell 2 below takes the path as a variable for this
   reason; Cell 3 globs recursively and needs no adjustment.

4. **Save Version → Save & Run All (Commit).** Not cell-by-cell: an
   interactive session is wiped when it closes and takes the checkpoints with
   it.

---

## Cell 1 — framework

```bash
%%bash
git clone https://github.com/Maelic/SGG-Benchmark.git
cd SGG-Benchmark && pip install -e . -q
pip install -q ultralytics hydra-core omegaconf
```

## Cell 1b — patch an import the current ultralytics no longer provides

Neither SGG-Benchmark's `requirements.txt` nor the install above pins
`ultralytics`, so a fresh clone picks up whatever is current. Recent releases
dropped `feature_visualization`, which several backbone modules import at the
top of the file. The function is only ever called inside an `if visualize:`
branch that training never enters, so the import is the whole problem, not the
absence of the function.

Idempotent by design: it skips any file already carrying its marker, which
matters because a naive re-run would otherwise wrap an already-wrapped import
and produce an `IndentationError`.

```python
import pathlib
root = pathlib.Path("/kaggle/working/SGG-Benchmark/sgg_benchmark")
old = "from ultralytics.utils.plotting import feature_visualization"
marker = "feature_visualization = None  # removed in newer ultralytics"
new = (
    "try:\n"
    "    from ultralytics.utils.plotting import feature_visualization\n"
    "except ImportError:\n"
    "    feature_visualization = None  # removed in newer ultralytics; only\n"
    "    # used by an optional debug path this run never enables"
)

patched = []
for f in root.rglob("*.py"):
    src = f.read_text(encoding="utf-8")
    if marker in src:
        continue                      # already patched, leave it alone
    if old in src:
        f.write_text(src.replace(old, new, 1), encoding="utf-8")
        patched.append(str(f.relative_to(root)))
print(f"patched {len(patched)} file(s):")
for name in patched:
    print(" ", name)
```

## Cell 2 — data and config into a writable tree

```bash
%%bash
cd /kaggle/working/SGG-Benchmark
cp -r /kaggle/input/spatial-sgg/spatial_sgg datasets/
mkdir -p configs/hydra/Spatial
cp /kaggle/input/spatial-sgg/spatial_sgg_react.yaml configs/hydra/Spatial/react.yaml

# the variant must be present, or the run below silently trains on whatever
# _annotations.coco.json happens to be there
for s in train val test; do
  test -f datasets/spatial_sgg/$s/_annotations.vlm.coco.json \
    || { echo "MISSING vlm variant in $s - re-export and re-upload"; exit 1; }
done
echo "vlm variant present in all three splits"
```

## Cell 3 — reuse the frozen detector

```python
import os, shutil, glob
os.chdir("/kaggle/working/SGG-Benchmark")
os.makedirs("checkpoints/BACKBONES", exist_ok=True)
src = glob.glob("/kaggle/input/**/yolov8m_spatial.pt", recursive=True)
assert src, "add the original notebook's committed output as an Input"
shutil.copy(src[0], "checkpoints/BACKBONES/yolov8m_spatial.pt")
print("frozen backbone from", src[0])
print("all three arms and all seeds share this detector, as in the first run")
```

## Cell 4 — train the VLM arm at three seeds

```bash
%%bash
cd /kaggle/working/SGG-Benchmark
for seed in 42 43 44; do
  # train/val carry the VLM's relations; test is ALWAYS human gold, for
  # every arm, or the arms are not being scored against the same yardstick
  for s in train val; do
    cp datasets/spatial_sgg/$s/_annotations.vlm.coco.json \
       datasets/spatial_sgg/$s/_annotations.coco.json
  done
  cp datasets/spatial_sgg/test/_annotations.human.coco.json \
     datasets/spatial_sgg/test/_annotations.coco.json

  echo "=== react_vlm_s${seed} ==="
  python tools/relation_train_net_hydra.py \
    --config-path ../configs/hydra/Spatial --config-name react \
    --task sgdet --save-best seed=${seed} \
    output_dir=./checkpoints/spatial/react_vlm_s${seed}
done
```

## Cell 5 — evaluate each seed against the human test gold

`inference()` is called directly rather than through a command-line script.
This is not a style preference. The framework's `--eval-only` path is silently
a no-op with this config, because `datasets.test` is undeclared and, unlike
the train/val path, gets no name fallback; it exits cleanly having evaluated
nothing. This is the recipe that produced the Chapter 6 numbers for the human
and auto arms, reused verbatim with the arm name changed.

```python
import os, torch, logging, glob, shutil
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

# fail on a missing checkpoint before loading any model, not three-quarters
# of the way through
for seed in [42, 43, 44]:
    tag = f"react_vlm_s{seed}"
    assert os.path.exists(f"checkpoints/spatial/{tag}/hydra_config.yaml"), tag
    ck = glob.glob(f"checkpoints/spatial/{tag}/best_model_epoch_*.pth")
    assert ck, f"no best_model_epoch_*.pth under checkpoints/spatial/{tag}/"
    print(f"{tag}: OK - {os.path.basename(sorted(ck)[-1])}")

# test set is human gold for every arm
shutil.copy("datasets/spatial_sgg/test/_annotations.human.coco.json",
            "datasets/spatial_sgg/test/_annotations.coco.json")

for seed in [42, 43, 44]:
    tag = f"react_vlm_s{seed}"
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
    print(f"EVAL {tag} | {os.path.basename(ckpt)} | "
          f"test images: {len(loader.dataset)}", flush=True)
    print("=" * 80, flush=True)
    with torch.no_grad():
        inference(cfg, model, loader, dataset_name="SpatialRobot_test",
                  iou_types=("bbox", "relations"), box_only=False,
                  device=cfg.model.device, expected_results=[],
                  expected_results_sigma_tol=4, output_folder=out, logger=logger)
print("\nALL EVALUATIONS DONE")
```

## Cell 6 — collect

```bash
%%bash
cd /kaggle/working/SGG-Benchmark
zip -rq /kaggle/working/vlm_results.zip checkpoints/spatial -x "*.pth" -x "*.pt"
echo "RESULTS -> /kaggle/working/vlm_results.zip"
ls -la /kaggle/working/vlm_results.zip
```

---

## Back on this machine

1. Download `vlm_results.zip` from the committed version's **Output** tab.
2. Unpack into `outputs/sgg_benchmark/seeds/` alongside the existing arms.
3. `python eval/seed_stats.py` picks the new arm up and rewrites
   `outputs/tables/seed_replication.md` with three arms instead of two.
4. `python scripts/make_figures.py` redraws `sgg_training_curves.png`, which
   then carries the vision-language arm and answers the epoch-curve question
   in the metric Chapter 6 reports.

## What to expect, stated before the run

The local classifier puts this arm between self-training and the pipeline,
nearer the former. If the benchmark agrees, the ordering on mR@100 should be
human > vlm > auto, mirroring Chapter 6's existing finding that ranked recall
against sparse human annotation rewards annotation habits; and on zR@100,
which isolates unseen compositions, the pipeline should stay far ahead of
both while the VLM arm sits above the human arm's near-zero.

That is a prediction, not a result. Record it either way: two of Chapter 5's
three predictions were refuted by the first benchmark run and reporting that
is what made the chapter worth reading.
