# Appendices

## Appendix A — Ethics

This project performs secondary analysis of an existing, published research
dataset (Wang et al., 2025; released under CC-BY 4.0), collected by the
project supervisor's research group using a Boston Dynamics Spot robot. No new
data involving human participants was collected for this dissertation. Scene
images in the dataset contain identifiable people; all figures reproduced in
this dissertation blur faces, and the dataset is used strictly as released.

<!-- TODO(user): confirm with the supervisor / programme office whether a
self-assessment ethics form is required for secondary use of the published
dataset, and insert the completed form or approval confirmation here. -->

## Appendix B — Reproducibility

Environment: Windows 11, Python 3.11.9 (virtual environment), PyTorch 2.5.1
(CUDA 12.1), single NVIDIA RTX 2060 (6 GB). All thresholds, seeds and model
identifiers are pinned in `configs/default.yaml`.

Key commands:

- `python scripts/run_annotator.py` — annotate the full dataset (GPU, ~5 min)
- `python scripts/reannotate_from_cache.py` — re-evaluate rule changes offline
  from the geometry cache (~20 s, no GPU)
- `python eval/fit_near.py` — fit and report the `near` threshold
- `python eval/fidelity.py` — the RQ1 battery → `outputs/`
- `python eval/ablations.py` — ablations A1–A6 → `outputs/tables/ablations.md`
- `python eval/downstream.py --seeds 42,43,44` — the RQ2 experiment
- `python scripts/make_figures.py` — regenerate all figures from cached results
- `pytest -q` — unit and invariant tests

(The full walk-through — installation, dataset layout, expected outputs and
runtimes per script — is completed in the final assembly, together with the
Dockerfile.)

## Appendix C — Predicate specification

The complete geometric specification of the seven predicates — definitions,
thresholds, worked examples, and the correction and flagging rules — is
maintained in the repository at `docs/predicate_spec.md` and is reproduced in
the submitted version of this document.
