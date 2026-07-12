# Consolidated Kaggle notebook — Week-7 experiment (Save & Run All ready)

Paste these 8 cells in order, then **Save Version → Save & Run All (Commit)**
(do NOT press play cell-by-cell — an interactive session is wiped when it ends;
a committed run saves its output permanently). Auto-finds the uploaded dataset,
patches the background-index offset at run time, and gates on a 1-epoch smoke so
a broken run aborts in ~20 min instead of wasting the full ~5 h.

See README.md for the per-cell explanation and expected numbers. The cells below
are the exact, path-robust versions.

## Cell 1 — install
```bash
%%bash
cd /kaggle/working && rm -rf SGG-Benchmark
git clone -q https://github.com/Maelic/SGG-Benchmark.git
cd SGG-Benchmark && pip install -e . -q
pip install -q ultralytics hydra-core omegaconf
echo "INSTALL DONE"
```

## Cell 2 — copy data + config (auto-finds the upload under /kaggle/input)
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

## Cell 3 — background-index patch + verify (idempotent; harmless if the upload is already fixed)
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

## Cell 4 — detector (~15 min)
```python
import os, yaml, shutil, glob
os.chdir("/kaggle/working/SGG-Benchmark")
yp = "datasets/spatial_sgg_yolo/data.yaml"
d = yaml.safe_load(open(yp)); d["path"] = os.path.abspath("datasets/spatial_sgg_yolo")
yaml.safe_dump(d, open(yp, "w"))
from ultralytics import YOLO
YOLO("yolov8m.pt").train(data=yp, epochs=60, imgsz=640, batch=16,
                         project="det", name="yolov8m_spatial", verbose=False)
os.makedirs("checkpoints/BACKBONES", exist_ok=True)
src = max(glob.glob("runs/detect/det/yolov8m_spatial*/weights/best.pt"), key=os.path.getmtime)
shutil.copy(src, "checkpoints/BACKBONES/yolov8m_spatial.pt")
print("DETECTOR DONE ->", src)
```

## Cell 5 — smoke gate (1 epoch, human; aborts if mR is still zero)
```python
import subprocess, os, shutil, re
os.chdir("/kaggle/working/SGG-Benchmark")
for s in ["train","val","test"]:
    shutil.copy(f"datasets/spatial_sgg/{s}/_annotations.human.coco.json",
                f"datasets/spatial_sgg/{s}/_annotations.coco.json")
os.system("rm -rf checkpoints/spatial/smoke")
cmd = ("python tools/relation_train_net_hydra.py --config-path ../configs/hydra/Spatial "
       "--config-name react --task sgdet --save-best "
       "solver.max_epoch=1 output_dir=./checkpoints/spatial/smoke")
r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
out = r.stdout + "\n" + r.stderr
print(out[-2500:])
mrs = [float(x) for x in re.findall(r"Result for mR:\s*([\d.]+)", out)]
print("\n>>> smoke mR:", mrs)
assert mrs and max(mrs) > 0, "SMOKE FAILED (mR still zero)"
print(">>> SMOKE PASSED — full arms will run")
```

## Cell 6 — arm A: human labels (~2 h)
```bash
%%bash
cd /kaggle/working/SGG-Benchmark
for s in train val test; do
  cp datasets/spatial_sgg/$s/_annotations.human.coco.json datasets/spatial_sgg/$s/_annotations.coco.json
done
rm -rf checkpoints/spatial/react_human
python tools/relation_train_net_hydra.py --config-path ../configs/hydra/Spatial \
  --config-name react --task sgdet --save-best \
  output_dir=./checkpoints/spatial/react_human
echo "ARM HUMAN DONE"
```

## Cell 7 — arm B: automatic labels (~2-3 h; test stays human gold)
```bash
%%bash
cd /kaggle/working/SGG-Benchmark
for s in train val; do
  cp datasets/spatial_sgg/$s/_annotations.auto.coco.json datasets/spatial_sgg/$s/_annotations.coco.json
done
cp datasets/spatial_sgg/test/_annotations.human.coco.json datasets/spatial_sgg/test/_annotations.coco.json
rm -rf checkpoints/spatial/react_auto
python tools/relation_train_net_hydra.py --config-path ../configs/hydra/Spatial \
  --config-name react --task sgdet --save-best \
  output_dir=./checkpoints/spatial/react_auto
echo "ARM AUTO DONE"
```

## Cell 8 — bundle logs for download
```bash
%%bash
cd /kaggle/working/SGG-Benchmark
zip -rq /kaggle/working/results.zip checkpoints/spatial -x "*.pth" -x "*.pt"
echo "RESULTS -> /kaggle/working/results.zip"; ls -la /kaggle/working/results.zip
```

## Post-run note: recovering the test-set evaluation

The framework's final/`--eval-only` evaluation is silently a no-op with this
config: `eval_only()` iterates `zip(hydra_cfg.datasets.test, loaders)` and,
unlike the train/val path, applies no name fallback — with `datasets.test`
undeclared the list is empty and "Evaluation completed!" prints instantly
(same-millisecond timestamps). Training and per-epoch validation are
unaffected. Workaround: pass `"++datasets.test=[SpatialRobot_test]"` on the
eval command line. The committed run's version output preserves the trained
`best_model_epoch_*.pth` and the detector, so the eval-only pass needs no
retraining: attach the previous notebook's output as an input, stage the
checkpoints, and run the two eval cells with the override (~5 min each).
