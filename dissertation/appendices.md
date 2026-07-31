# Appendices

## Appendix A: Ethical Approval

This project performs secondary analysis of an existing, published research
dataset (Wang et al., 2025; released under CC-BY 4.0), collected by the
project supervisor's research group using a Boston Dynamics Spot robot. No new
personal data was collected for the annotation study. Scene images in the
dataset contain identifiable people; all figures reproduced in this
dissertation blur faces, and the dataset is used strictly as released.

A second body of data is used in §4.14 and §9.3: the complete 2,650-frame
capture from which the released images were cut, supplied directly by the
supervising group. Frames 000000–000883 are the released dataset itself,
verified by exact pixel match; the remaining 1,766 frames carry no
annotation and are **not** covered by the CC-BY release. They are used here
only to measure the pipeline's behaviour on unlabelled input, they are not
redistributed with this work, and no figure reproduces a frame from them
without the supervisor's specific agreement. They show the same laboratory
and the same people as the released portion, and the face-anonymisation rule
above applies to them unchanged.

The independent validation of the automatic labels (Chapter 4) collects
anonymous true/false judgements from adult volunteers through a purpose-built
web quiz. No names, email addresses, IP addresses or any other personal data
are recorded; each browser receives a random identifier used only to spread
item coverage and remove duplicate answers. Participation is voluntary,
takes about three minutes, and can be abandoned at any point; an information
panel on the first screen states the purpose, the data collected, and a
contact address. Faces in all quiz images are pixelated before publication,
and items where anonymisation would obscure the object under judgement were
removed rather than shown. The collection is covered by the University's
ethics self-assessment process.

<!-- TODO(user): insert the completed self-assessment form / approval
confirmation here once submitted. -->

**Demonstration footage.** The two video clips in §4.12 are royalty-free
stock footage from Pexels, used under the Pexels licence (free use, no
attribution required): clip 1 (desk scene, moving camera)
https://www.pexels.com/video/6558513/ and clip 2 (overhead desk, moving
hands) https://www.pexels.com/video/a-person-working-with-pictures-and-photos-taken-using-a-modern-camera-3250234/.

## Appendix B: Reproducibility

Environment: Windows 11, Python 3.11.9 (virtual environment), PyTorch 2.5.1
(CUDA 12.1), single NVIDIA RTX 2060 (6 GB). All thresholds, seeds and model
identifiers are pinned in `configs/default.yaml`.

Key commands:

- `python scripts/run_annotator.py`: annotate the full dataset (GPU, ~5 min)
- `python scripts/reannotate_from_cache.py`: re-evaluate rule changes offline
  from the geometry cache (~20 s, no GPU)
- `python eval/fit_near.py`: fit and report the `near` threshold
- `python eval/fidelity.py`: the RQ1 battery → `outputs/`
- `python eval/uncertainty.py --iters 2000`: cluster-bootstrap intervals for
  every headline recall → `outputs/tables/uncertainty.md`
- `python eval/annotator_agreement.py`: annotator heterogeneity and the
  estimated human-human agreement bounds → `outputs/tables/annotator_agreement.md`
- `python eval/ablations.py`: ablations A1–A6 → `outputs/tables/ablations.md`
- `python eval/depth_ablation.py`: the A8 depth-model comparison
- `python eval/seed_stats.py`: aggregates the benchmark arms across seeds →
  `outputs/tables/seed_replication.md`
- `python eval/keyframe_propagation.py --sweep 5,10,20,30`: content-adaptive
  frame selection and the viewpoint-stability measurement of §4.14 →
  `outputs/keyframe_propagation.json`
- `python eval/video_stability.py`: the persistence and Jaccard figures of
  §4.12, derived from the recorded per-frame files →
  `outputs/video_stability.json`
- `python eval/extension_scale.py`: throughput, density and predicate
  distribution on the unannotated capture (§4.15) →
  `outputs/extension_scale.json`
- `python eval/downstream.py --seeds 42,43,44`: the RQ2 experiment, all three
  label sources (human, self-trained, automatic)
- `python analysis/score_votes.py votes.csv`: scores the independent
  validation study (crowd precision, author-bias kappa, rater reliability);
  lives with the study's own repository
- `python scripts/make_figures.py`: regenerate all figures from cached results
- `pytest -q`: unit and invariant tests

The Chapter 6 experiment is reproducible from `scripts/kaggle/`: the dataset
converters (`export_sgg_benchmark.py`, `export_yolo_det.py`), the adapted
REACT++ configuration, the run recipes (`README.md`, `notebook_cells.md`, and
`seed_replication.ipynb` for the seed replication), and
the executed evaluation notebook with its outputs (`eval_notebook.ipynb`),
which is the recorded provenance of every number in Chapter 6's tables.
Training logs and parsed results: `outputs/sgg_benchmark/`.

### Full reproduction walk-through

**1. Environment.** Either build the container (`docker build -t
spatial-annotator .`; the `Dockerfile` at the repository root pins Python
3.11, torch 2.5.1 + cu121 and installs SAM2 from GitHub, and runs the test
suite at build time), or create a Python 3.11/3.12 venv and follow the three
numbered notes at the top of `requirements.txt`. The known pitfall is
documented there: installing SAM2 can silently replace CUDA torch with a
CPU wheel, fixed by reinstalling torch with `--no-deps --force-reinstall`
from the cu121 index. Verify with `python scripts/smoke_test.py --image
assets/sample.jpg`, which loads SAM2 and Depth Anything and reports CUDA
availability and peak memory (~0.65 GB).

**2. Data.** Clone the released dataset (CC-BY 4.0) and point
`dataset.root` in `configs/default.yaml` at it. The loader expects the
release's own layout (`img_data/group_N/*.jpg` plus the annotation JSONs)
and corrects the images' 180° EXIF orientation itself; nothing is
preprocessed on disk.

**3. One GPU pass, then everything is offline.**
`python scripts/run_annotator.py` (~5 min on the RTX 2060) writes the
annotations in all three native formats plus the geometry, contact and
depth caches under `outputs/`, and `outputs/pairs.csv`. Every experiment
below runs from those caches on CPU:

| command | produces | time |
|---|---|---|
| `pytest -q` | 25 unit + invariant tests | <1 min |
| `python scripts/reannotate_from_cache.py` | re-runs the rules after any threshold change | ~20 s |
| `python eval/fit_near.py` | the near-threshold protocol, `near_fit.json` | ~1 min |
| `python eval/fidelity.py` | the RQ1 battery, `fidelity_report.json` | ~2 min |
| `python eval/uncertainty.py --iters 2000` | cluster-bootstrap CIs | ~2 min |
| `python eval/annotator_agreement.py` | heterogeneity + Fréchet bounds | <1 min |
| `python eval/keyframe_propagation.py --sweep 5,10,20,30` | §4.14 segmentation, stability, propagation cost | ~3 min |
| `python eval/ablations.py` | A1–A6 sweeps | ~10 min |
| `python eval/depth_ablation.py` | A8 (needs the `outputs_base` pass) | <1 min |
| `python eval/downstream.py --seeds 42,43,44` | RQ2, three arms | ~4 h CPU |
| `python scripts/make_figures.py` | every figure | ~1 min |

The two GPU extras: `python scripts/run_sgdet.py --threshold 0.25` for the
deployment-mode pass (~47 min) and `python scripts/run_annotator.py
--config configs/depth_base.yaml --out outputs_base` for A8's Base-model
pass (~7 min).

**4. The benchmark (Chapter 6)** runs on Kaggle rather than locally
(REACT++ training needs more than 6 GB): upload
`datasets/spatial_sgg_upload.zip` (built by
`scripts/export_sgg_benchmark.py` and `scripts/export_yolo_det.py`), then
commit the notebooks in `scripts/kaggle/` in order (the training run in
`notebook_cells.md`, the seed replication in `seed_replication.ipynb`, and
the re-evaluation in `reeval_seeds_and_groups.ipynb`) on a T4 x2
accelerator. Download each committed version's outputs into
`outputs/sgg_benchmark/` and run `python eval/seed_stats.py`. The two
framework traps are documented in the notebooks themselves: class index 0
is reserved (patched idempotently in a cell), and `--eval-only` is silently
a no-op with this config, so evaluation calls `inference()` directly.

**5. The validation study** lives in its own public repository
(`robot-factcheck`): `tools/build_validation_set.py` regenerates the claim
set and site images from this repository's caches, and
`analysis/score_votes.py` scores the exported votes sheet. The private
answer key never enters the public repository.

Every reported number traces to one of the JSON/markdown artefacts these
commands write; no figure or table in this dissertation is produced by hand.

## Appendix C: Predicate specification

The complete geometric specification of the seven predicates (definitions,
thresholds, worked examples, and the correction and flagging rules) is
maintained in the repository at `docs/predicate_spec.md` and is reproduced in
the submitted version of this document.
