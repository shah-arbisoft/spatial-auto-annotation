# Week-7 experiment: REACT++ trained on human vs automatic labels (Kaggle)

The direct test of the dissertation's three predictions (ch5 §5.5): train the
same SGG model twice — once on the human relations, once on this tool's — and
compare mR@100 on the identical human test set (groups 6–8). The detector is
trained once on the ground-truth boxes and shared, so the arms differ in the
relation labels only.

## One-time setup

1. Kaggle account, phone-verified (Settings → Phone verification) — unlocks
   GPU (30 h/week). Pick **GPU T4 x2**. (Avoid P100: the current Kaggle
   PyTorch build dropped Pascal/sm_60 support, so P100 throws
   "no kernel image is available for execution on the device". T4 is
   Turing/sm_75 and works.)
2. Zip and upload the exported data as ONE Kaggle Dataset (slug `spatial-sgg`):

       cd datasets
       python -c "import zipfile; from pathlib import Path; z = zipfile.ZipFile('spatial_sgg_upload.zip','w',zipfile.ZIP_DEFLATED); [z.write(p, p.as_posix()) for root in ('spatial_sgg','spatial_sgg_yolo') for p in sorted(Path(root).rglob('*')) if p.is_file()]; z.write('spatial_sgg_react.yaml','spatial_sgg_react.yaml'); z.close()"

   (Windows tar does not write real zip files. Regenerate the data anytime:
   `python scripts/export_sgg_benchmark.py` and
   `python scripts/export_yolo_det.py`; the config copy lives at
   `datasets/spatial_sgg_react.yaml`.)
3. New Kaggle notebook → Add data → your `spatial-sgg` dataset → enable GPU +
   internet.

## Notebook cells

**Cell 1 — framework**
```bash
%%bash
git clone https://github.com/Maelic/SGG-Benchmark.git
cd SGG-Benchmark && pip install -e . -q
pip install -q ultralytics hydra-core omegaconf
# hydra-core/omegaconf are not pulled by the framework's install but its
# config layer imports them; ultralytics is for the detector in Cell 3.
```

**Cell 2 — data into a writable tree**
```bash
%%bash
cd SGG-Benchmark
cp -r /kaggle/input/spatial-sgg/spatial_sgg datasets/
cp -r /kaggle/input/spatial-sgg/spatial_sgg_yolo datasets/
```

**Cell 3 — detector, trained once (~15 min)**
```python
import os, yaml, shutil
os.chdir("/kaggle/working/SGG-Benchmark")

# sanity: the images must have copied in Cell 2
assert os.path.isdir("datasets/spatial_sgg_yolo/images/train"), "run Cell 2 first"
print("train images:", len(os.listdir("datasets/spatial_sgg_yolo/images/train")))

# Ultralytics resolves a relative `path:` against the CWD, not the yaml's
# folder — so rewrite it to the absolute dataset root.
yp = "datasets/spatial_sgg_yolo/data.yaml"
d = yaml.safe_load(open(yp))
d["path"] = os.path.abspath("datasets/spatial_sgg_yolo")
yaml.safe_dump(d, open(yp, "w"))
print("data root ->", d["path"])

from ultralytics import YOLO
m = YOLO("yolov8m.pt")
m.train(data=yp, epochs=60, imgsz=640, batch=16,
        project="det", name="yolov8m_spatial")
# mAP50 lands ~0.65-0.70 on the 500-image train split (bottle/book/human
# strong, cube/box weak). Lower than the paper's 0.93 because we train on the
# train split only; both experiment arms share this frozen detector, so the
# label-source comparison stays fair (detection is a shared ceiling).
import glob
os.makedirs("checkpoints/BACKBONES", exist_ok=True)
# Ultralytics saves under runs/detect/<project>/ and auto-numbers on rerun.
src = max(glob.glob("runs/detect/det/yolov8m_spatial*/weights/best.pt"),
          key=os.path.getmtime)
shutil.copy(src, "checkpoints/BACKBONES/yolov8m_spatial.pt")
print("detector copied from", src)
```

**Cell 4 — experiment config**
```bash
%%bash
cd SGG-Benchmark
mkdir -p configs/hydra/Spatial
cp /kaggle/input/spatial-sgg/spatial_sgg_react.yaml configs/hydra/Spatial/react.yaml
```
(Also add `spatial_sgg_react.yaml` to the uploaded dataset zip, or paste it
with `%%writefile configs/hydra/Spatial/react.yaml`.)

**Cell 5 — smoke test (1 epoch, human arm)**
```bash
%%bash
cd SGG-Benchmark
for s in train val test; do
  cp datasets/spatial_sgg/$s/_annotations.human.coco.json \
     datasets/spatial_sgg/$s/_annotations.coco.json
done
python tools/relation_train_net_hydra.py --config-path ../configs/hydra/Spatial \
  --config-name react --task sgdet --save-best \
  solver.max_epoch=1 output_dir=./checkpoints/spatial/smoke
```
Fix anything this surfaces before the real runs (likely suspects: class-count
fields, dataset name registration).

**Cell 6 — arm A: human labels (~1.5–2 h)**
```bash
%%bash
cd SGG-Benchmark
for s in train val test; do
  cp datasets/spatial_sgg/$s/_annotations.human.coco.json \
     datasets/spatial_sgg/$s/_annotations.coco.json
done
python tools/relation_train_net_hydra.py --config-path ../configs/hydra/Spatial \
  --config-name react --task sgdet --save-best \
  output_dir=./checkpoints/spatial/react_human
```

**Cell 7 — arm B: automatic labels (~2–3 h; 22× the relations)**
```bash
%%bash
cd SGG-Benchmark
for s in train val; do
  cp datasets/spatial_sgg/$s/_annotations.auto.coco.json \
     datasets/spatial_sgg/$s/_annotations.coco.json
done
cp datasets/spatial_sgg/test/_annotations.human.coco.json \
   datasets/spatial_sgg/test/_annotations.coco.json   # test is ALWAYS human gold
python tools/relation_train_net_hydra.py --config-path ../configs/hydra/Spatial \
  --config-name react --task sgdet --save-best \
  output_dir=./checkpoints/spatial/react_auto
```

**Cell 8 — collect**: zip both `checkpoints/spatial/react_*` folders (logs +
best checkpoints + per-epoch validation metrics) and download; the offline
curve/figure script lives in this repo and runs on the logs.

## What is in this folder

Three kinds of file, and the difference matters.

**Executed notebooks** carry their outputs and are the recorded provenance of
Chapter 6. They ran on Kaggle (T4) in this order:

| file | what it ran | writes |
|---|---|---|
| `executed_1_train_arms_seed42.ipynb` | YOLOv8m detector, the background-index patch, a one-epoch smoke test, then the human and auto arms at seed 42 | the first-run checkpoints and logs |
| `executed_2_seed_replication.ipynb` | the same two arms at seeds 43 and 44, on the frozen detector | `seed_results.zip` |
| `executed_3_vlm_arm.ipynb` | the third arm, trained on `_annotations.vlm.coco.json` at seeds 42/43/44 | `vlm_results.zip` |
| `executed_4_reeval_group_slices.ipynb` | re-evaluates every checkpoint against the corrected zero-shot reference and the per-group slices, all three arms | `reeval_results.json` |
| `eval_notebook.ipynb` | the first run's evaluation pass | `test_results.json` |

Run them in that order: 2 and 3 need the detector 1 freezes, and 4 needs the
checkpoints 1 to 3 produce.

**Generators** (`make_seeds_notebook.py`, `make_reeval_notebook.py`) write
clean, unexecuted copies of notebooks 2 and 4. Use these to regenerate a run
rather than editing an executed notebook, which would destroy the record.

**Recipes** (`notebook_cells.md`, `notebook_cells_seeds.md`,
`notebook_cells_vlm.md`) are the cell-by-cell instructions, useful if you are
pasting into a fresh Kaggle notebook by hand.

The dataset the notebooks consume (`spatial_sgg` and `spatial_sgg_yolo`,
about 71 MB) is **not** committed. It is built from this repository by
`scripts/export_sgg_benchmark.py` and `scripts/export_yolo_det.py` and
uploaded to Kaggle as a dataset; regenerate it rather than looking for it here.

## What we predict (already in writing, ch5 §5.5)

1. **Later saturation** — the human arm peaks within ~2–6 epochs (the source
   paper saw all six models saturate early); the auto arm keeps improving.
2. **Higher plateau** — the auto arm's final mR@100 exceeds the human arm's.
3. **`near` recovery** — per-predicate: human-arm `near` stays near the
   paper's 0.22–0.25; the auto arm's rises substantially.

## Honest notes

- First run on a fresh framework rarely works untouched — that is what the
  smoke test is for. Budget one debugging session.
- The val/test gold of groups 6–8 carries the measured annotation defects
  (inverted front/behind conventions) — both arms are penalised identically,
  as in RQ2.
- Total GPU budget: roughly 4–6 h, well inside one Kaggle session.
