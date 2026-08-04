"""Generate the re-evaluation notebook (correct zero-shot set + per-group slices).

The seed-replication run staged the test annotations before evaluating but
left the TRAIN split as whatever the last training run had written, so the
zero-shot "seen triplet" set was taken from the auto arm's 213 types instead
of the human arm's 94. With 213 types seen, almost nothing in the test set
counts as unseen and every zR collapsed to zero. The pooled R/mR/F1 numbers
are unaffected, because they never consult training statistics.

This notebook re-evaluates the checkpoints that already exist (no retraining)
with the train split staged to HUMAN for every arm, which is the fixed
reference the original run used and the only way the two arms' zero-shot
numbers are comparable. It also scores each annotator group separately, so
the group-7 margin gets the same seed treatment as the pooled numbers.

    python scripts/kaggle/make_reeval_notebook.py

Kaggle: import the notebook, attach the dataset plus BOTH training notebooks'
outputs, GPU T4 x2, Save & Run All.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).with_name("reeval_seeds_and_groups.ipynb")

MD_INTRO = """\
# Re-evaluation: correct zero-shot reference + per-group slices

Nothing is retrained here. Six checkpoints already exist (two arms x three
seeds) and this notebook only scores them again, fixing one staging mistake
and adding the per-group breakdown.

**What went wrong the first time.** Zero-shot recall is measured against the
set of subject-predicate-object types *seen during training*, which the
framework reads from whatever is currently staged as the train split. The
replication notebook staged the test annotations before evaluating but left
the train split holding the auto arm's labels, so all four evaluations used
its 213 seen types rather than the human arm's 94. With 213 types seen,
almost no test triplet qualifies as unseen and every zR came back 0.000.
R@100, mR@100 and F1@100 never consult training statistics, so those numbers
were and remain valid.

**The fix.** Stage the human train split once, before any evaluation, and
leave it there. Both arms are then scored against the same fixed reference:
"triplet types the human annotation never contained". That is the comparison
the original run made (its logs show 94 seen triplets for both arms) and the
only one under which the two arms' zero-shot numbers mean the same thing.

**Also added.** Each annotator group in the test split is scored separately.
Group 7 is the only test annotator with no measured convention defect, so its
margin is the one claim in chapter 6 that most needs a spread rather than a
single run.

## Before running

1. Accelerator: **GPU T4 x2**.
2. Add Input: dataset **shah9212/spatial-sgg**.
3. Add Input: committed output of the **original** training notebook
   (`notebook-ssg`) - supplies the seed-42 checkpoints.
4. Add Input: committed output of the **seed replication** notebook -
   supplies the seed-43 and seed-44 checkpoints.
5. Add Input: committed output of the **vision-language arm** notebook -
   supplies react_vlm_s42/43/44. Omit it and the run still succeeds, simply
   without that arm; the assertion below only requires four runs.
5. **Save Version > Save & Run All (Commit)**.

Runtime ~40 minutes for two arms, ~60 with the vision-language arm
(9 runs x 4 slices = 36 evaluations, no training).
"""

CELLS: list[tuple[str, str]] = [
    ("markdown", MD_INTRO),

    ("markdown", "## Cell 1 - install"),
    ("code", """\
%%bash
cd /kaggle/working && rm -rf SGG-Benchmark
git clone -q https://github.com/Maelic/SGG-Benchmark.git
cd SGG-Benchmark && pip install -e . -q
pip install -q ultralytics hydra-core omegaconf
echo "INSTALL DONE"
"""),

    ("markdown", "## Cell 2 - copy dataset and config"),
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
echo "DATA COPIED"
"""),

    ("markdown", "## Cell 3 - background-index patch (idempotent)"),
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
print("PATCH OK")
"""),

    ("markdown",
     "## Cell 4 - gather the six checkpoints and the detector\\n\\n"
     "Two arms x three seeds. Seed 42 comes from the original training "
     "notebook, seeds 43 and 44 from the replication; both must be attached "
     "as inputs."),
    ("code", """\
import glob, os, shutil
os.chdir("/kaggle/working/SGG-Benchmark")

det = glob.glob("/kaggle/input/**/yolov8m_spatial.pt", recursive=True)
assert det, "detector not found - attach the original training notebook output"
os.makedirs("checkpoints/BACKBONES", exist_ok=True)
shutil.copy(det[0], "checkpoints/BACKBONES/yolov8m_spatial.pt")
print("detector:", det[0])

# seed 42 folders are named react_human / react_auto; 43 and 44 carry _sNN
WANT = {"react_human": ("human", 42), "react_auto": ("auto", 42),
        "react_human_s43": ("human", 43), "react_auto_s43": ("auto", 43),
        "react_human_s44": ("human", 44), "react_auto_s44": ("auto", 44),
        # the vision-language arm, trained in its own notebook; every seed
        # carries an _sNN suffix because it never had a seed-42-only run
        "react_vlm_s42": ("vlm", 42), "react_vlm_s43": ("vlm", 43),
        "react_vlm_s44": ("vlm", 44)}

RUNS = {}
for name, (arm, seed) in WANT.items():
    cfgs = [p for p in glob.glob(f"/kaggle/input/**/{name}/hydra_config.yaml", recursive=True)
            if os.path.basename(os.path.dirname(p)) == name]
    ckpts = [p for p in glob.glob(f"/kaggle/input/**/{name}/best_model_epoch_*.pth", recursive=True)
             if os.path.basename(os.path.dirname(p)) == name]
    if not (cfgs and ckpts):
        print(f"  MISSING {name}: cfg={len(cfgs)} ckpt={len(ckpts)}")
        continue
    RUNS[name] = {"arm": arm, "seed": seed,
                  "cfg": sorted(cfgs)[-1], "ckpt": sorted(ckpts)[-1]}
    print(f"  found {name}: {os.path.basename(RUNS[name]['ckpt'])}")

print(f"\\n{len(RUNS)} of 6 runs available")
assert len(RUNS) >= 4, "attach both training notebooks' outputs as inputs"
"""),

    ("markdown",
     "## Cell 5 - stage the fixed zero-shot reference, and build per-group test sets\\n\\n"
     "The train split is staged to HUMAN and left alone for the rest of the "
     "notebook: that is what defines the zero-shot set for every arm. The "
     "per-group test files filter images by the `group_N_` filename prefix, "
     "carrying their annotations and relations with them."),
    ("code", """\
import json, os, shutil
os.chdir("/kaggle/working/SGG-Benchmark")

# fixed zero-shot reference for BOTH arms: the human training annotation
for split in ["train", "val"]:
    shutil.copy(f"datasets/spatial_sgg/{split}/_annotations.human.coco.json",
                f"datasets/spatial_sgg/{split}/_annotations.coco.json")
print("train/val staged to HUMAN (defines the seen-triplet set; expect 94)")

TEST = "datasets/spatial_sgg/test"
full = json.load(open(f"{TEST}/_annotations.human.coco.json"))
shutil.copy(f"{TEST}/_annotations.human.coco.json", f"{TEST}/_annotations.full.coco.json")

def subset(group):
    keep = {im["id"] for im in full["images"] if im["file_name"].startswith(group + "_")}
    ann  = [a for a in full["annotations"] if a["image_id"] in keep]
    akeep = {a["id"] for a in ann}
    rel  = [r for r in full["rel_annotations"]
            if r["subject_id"] in akeep and r["object_id"] in akeep]
    d = dict(full)
    d["images"]          = [im for im in full["images"] if im["id"] in keep]
    d["annotations"]     = ann
    d["rel_annotations"] = rel
    path = f"{TEST}/_annotations.{group}.coco.json"
    json.dump(d, open(path, "w"))
    print(f"  {group}: {len(d['images'])} images, {len(rel)} relations -> {path}")
    return path

SLICES = {"full": f"{TEST}/_annotations.full.coco.json"}
for g in ["group_6", "group_7", "group_8"]:
    SLICES[g] = subset(g)
"""),

    ("markdown",
     "## Cell 6 - evaluate every run on every slice\\n\\n"
     "`--eval-only` is a no-op with this config, so `inference()` is called "
     "directly. Results are collected into one dictionary and printed as "
     "JSON at the end for easy copy-out."),
    ("code", """\
import os, json, glob, shutil, torch, logging, statistics as st
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

TEST = "datasets/spatial_sgg/test"
RESULTS = {}

for name, info in RUNS.items():
    for slice_name, slice_path in SLICES.items():
        shutil.copy(slice_path, f"{TEST}/_annotations.coco.json")
        out = f"./checkpoints/spatial/re_{name}_{slice_name}"
        os.makedirs(out, exist_ok=True)
        cfg = OmegaConf.load(info["cfg"])
        cfg.output_dir = out

        model = build_detection_model(cfg).to(cfg.model.device)
        DetectronCheckpointer(cfg, model).load(info["ckpt"])
        model.eval()
        loader = make_data_loader(cfg, mode="test")[0]
        print("=" * 78, flush=True)
        print(f"EVAL {name} on {slice_name} ({len(loader.dataset)} images)", flush=True)
        print("=" * 78, flush=True)
        with torch.no_grad():
            inference(cfg, model, loader, dataset_name="SpatialRobot_test",
                      iou_types=("bbox", "relations"), box_only=False,
                      device=cfg.model.device, expected_results=[],
                      expected_results_sigma_tol=4, output_folder=out, logger=logger)

        f = f"{out}/eval_results_top_100.json"
        if os.path.exists(f):
            d = json.load(open(f))
            zs = d.get("sgdet_zeroshot_recall", {}).get("100", [])
            RESULTS[f"{name}|{slice_name}"] = {
                "arm": info["arm"], "seed": info["seed"], "slice": slice_name,
                "R@100":  st.mean(d["sgdet_recall"]["100"]) if d.get("sgdet_recall") else None,
                "mR@100": d.get("sgdet_mean_recall", {}).get("100"),
                "F1@100": d.get("sgdet_f1_score", {}).get("100"),
                "zR@100": (st.mean(zs) if zs else 0.0),
                "n_zeroshot": len(zs),
            }

json.dump(RESULTS, open("/kaggle/working/reeval_results.json", "w"), indent=1)
print("\\n\\n===== RESULTS =====")
print(json.dumps(RESULTS, indent=1))
"""),

    ("markdown", "## Cell 7 - bundle for download"),
    ("code", """\
%%bash
cd /kaggle/working/SGG-Benchmark
zip -rq /kaggle/working/reeval_results.zip checkpoints/spatial -x "*.pth" -x "*.pt"
cp /kaggle/working/reeval_results.json /kaggle/working/reeval_results_copy.json
echo "-> /kaggle/working/reeval_results.zip  and  reeval_results.json"
ls -la /kaggle/working/reeval_results.*
"""),

    ("markdown",
     "## After the run\\n\\n"
     "Download `reeval_results.json` (and the zip for the logs) from the "
     "committed version's Output tab, drop the json into "
     "`outputs/sgg_benchmark/`, and run `python eval/seed_stats.py`.\\n\\n"
     "Sanity check while it runs: the log should say **94 seen triplets** for "
     "every evaluation. If it still says 213, the train split was not staged "
     "and the zero-shot numbers are again not comparable."),
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
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python", "version": "3.10"},
                       "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {OUT}  ({len(cells)} cells, "
          f"{sum(1 for k, _ in CELLS if k == 'code')} code)")


if __name__ == "__main__":
    main()
