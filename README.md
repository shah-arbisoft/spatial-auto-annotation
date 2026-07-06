# Spatial-Auto-Annotation

A fully-automatic spatial-relationship annotator for robot-acquired indoor scenes.

Given a raw RGB image, the pipeline detects objects, segments them, estimates
monocular depth, lifts each object to a 3D position, and **computes** the seven
spatial predicates between every ordered pair of objects from geometry — no human
decides any label. It exports scene graphs in the dataset's own formats
(Visual Genome JSON, YOLO txt, h5).

This is the practical artefact of an MSc dissertation (University of Surrey,
MSc Data Science). The dissertation validates the labels two ways: **fidelity**
against ~900 human labels (RQ1), and **downstream utility** by training a
controlled relation classifier on human vs. auto labels (RQ2).

> **Key idea.** SGG models like REACT++ *predict* relationships from learned
> visual patterns, and only exist because humans labelled their training data.
> This pipeline *computes* relationships from measured geometry and runs *before*
> any learned relation model. It is the supplier of labelled data, not a
> competitor. This is valid precisely because the seven predicates are spatial
> and therefore computable from geometry. The perception models (detector,
> segmentation, depth) only *measure* where things are; a deterministic rule
> decides the relationship.

## The seven predicates

`behind`, `in front of`, `on`, `under`, `left of`, `right of`, `near`.

Their exact geometric definitions are the graded core of the project and live in
[docs/predicate_spec.md](docs/predicate_spec.md), implemented in
[src/predicates.py](src/predicates.py). The `near` threshold `T` is **fitted to
the human labels**, directly addressing the inconsistency the source paper flagged.

## Pipeline

```
RGB image
  ├─ detect.py    YOLOv10m (trained on the 900) or GroundingDINO  → boxes + classes
  ├─ segment.py   SAM2, box-prompted                              → per-object masks
  ├─ depth.py     Depth Anything v2 Small                         → relative depth map
  ├─ geometry.py  lift mask + depth                               → 3D position per object
  ├─ predicates.py  rules + near-threshold + correction + flags   → 7-predicate triplets
  └─ writers.py   VG JSON / YOLO txt / h5                         → scene graph
```

`pipeline.py` orchestrates these end to end.

## Repository layout

```
configs/      seeds, hyperparameters, thresholds (default.yaml)
data/         dataset + human labels (gitignored — obtain from supervisor)
src/          detect, segment, depth, geometry, predicates, writers, pipeline
eval/         fidelity (RQ1), ablations, classifier + downstream (RQ2)
docs/         predicate_spec.md — the seven-predicate definition
dissertation/ chapter drafts (write from week one)
scripts/      smoke_test.py and helpers
outputs/      tables and figures for the dissertation
```

## Environment

- **Hardware target:** one RTX 2060, 6 GB. Use Depth Anything v2 **Small** and a
  small SAM2 variant to fit memory. Offload bulk runs / training to Kaggle or Colab.
- **Python:** 3.10–3.12. **Note:** PyTorch and SAM2 do not yet ship wheels for
  Python 3.14, which is what is currently installed on this machine. Create a
  3.11 (or 3.12) virtual environment before installing `requirements.txt`.

```bash
# example, adjust the python launcher to a 3.11/3.12 interpreter
py -3.11 -m venv .venv
.venv\Scripts\activate          # Windows
pip install --upgrade pip
pip install -r requirements.txt # see comments inside for the CUDA build
python scripts/smoke_test.py --image assets/sample.jpg   # confirms depth + SAM2 run
```

## Reproducibility

Everything is seeded (see `configs/default.yaml`) and version-pinned. A Dockerfile
and reproducibility appendix are part of the deliverable. The dataset is CC-BY 4.0.

Reproducing the results end to end:

```bash
python scripts/run_annotator.py          # full dataset (GPU, ~5 min)
python eval/fit_near.py                  # fit + report the near threshold
python eval/fidelity.py                  # RQ1 tables -> outputs/tables/
python scripts/make_audit_pack.py        # manual-audit sample -> outputs/audit/
python scripts/reannotate_from_cache.py  # re-run rules offline after any change
pytest -q                                # rule-layer unit + invariant tests
```

## Licences

YOLO (Ultralytics) AGPL-3.0 · Depth Anything v2 **Small** Apache 2.0 (avoid
Base/Large — non-commercial) · SAM2 Apache 2.0 · dataset CC-BY 4.0.

## Status

Scope settled. **Dataset received** from the supervisor (Dr Peng Wang) and
verified: 838 annotated images, 8,928 target triplets across 9 annotator groups
— see [docs/DATASET_NOTES.md](docs/DATASET_NOTES.md). The loader
([src/dataset.py](src/dataset.py)) and writers ([src/writers.py](src/writers.py))
are validated byte-compatible with the SGDET-Annotate format (0-pixel box
round-trip). Next: wire the perception models end-to-end (Week 2). See
`dissertation/` for the in-progress write-up.
