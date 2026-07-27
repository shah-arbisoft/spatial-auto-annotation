# Appendices

## Appendix A: Ethics

This project performs secondary analysis of an existing, published research
dataset (Wang et al., 2025; released under CC-BY 4.0), collected by the
project supervisor's research group using a Boston Dynamics Spot robot. No new
personal data was collected for the annotation study. Scene images in the
dataset contain identifiable people; all figures reproduced in this
dissertation blur faces, and the dataset is used strictly as released.

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
- `python eval/ablations.py`: ablations A1–A6 → `outputs/tables/ablations.md`
- `python eval/downstream.py --seeds 42,43,44`: the RQ2 experiment
- `python scripts/make_figures.py`: regenerate all figures from cached results
- `pytest -q`: unit and invariant tests

The Chapter 6 experiment is reproducible from `scripts/kaggle/`: the dataset
converters (`export_sgg_benchmark.py`, `export_yolo_det.py`), the adapted
REACT++ configuration, the run recipe (`README.md`, `notebook_cells.md`), and
the executed evaluation notebook with its outputs (`eval_notebook.ipynb`),
which is the recorded provenance of every number in Chapter 6's tables.
Training logs and parsed results: `outputs/sgg_benchmark/`.

(The full walk-through, covering installation, dataset layout, expected
outputs and runtimes per script, is completed in the final assembly, together
with the Dockerfile.)

## Appendix C: Predicate specification

The complete geometric specification of the seven predicates (definitions,
thresholds, worked examples, and the correction and flagging rules) is
maintained in the repository at `docs/predicate_spec.md` and is reproduced in
the submitted version of this document.
