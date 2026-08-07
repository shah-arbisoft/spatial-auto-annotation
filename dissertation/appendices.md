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
ethics self-assessment process, completed and submitted on 19 July 2026.
Because the study collects no personal data, involves no vulnerable groups
and carries no foreseeable risk beyond that of everyday life, it falls
within the self-assessment route rather than requiring full committee
review.

*The completed self-assessment form is attached at the end of this
appendix.*

**Demonstration footage.** The two video clips in §4.12 are royalty-free
stock footage from Pexels, used under the Pexels licence (free use, no
attribution required): clip 1 (desk scene, moving camera)
https://www.pexels.com/video/6558513/ and clip 2 (overhead desk, moving
hands)
https://www.pexels.com/video/a-person-working-with-pictures-and-photos-taken-using-a-modern-camera-3250234/.

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
- `python eval/ablations.py`: ablations A1–A7 → `outputs/tables/ablations.md`
- `python eval/depth_ablation.py`: the A8 depth-model comparison
- `python eval/parallax_ablation.py --method triangulate --gap 10`: the
  A9 multi-frame depth comparison; needs the raw capture (Appendix D.6) →
  `outputs/parallax_ablation.json`
- `python eval/seed_stats.py`: aggregates the benchmark arms across seeds →
  `outputs/tables/seed_replication.md`
- `python eval/keyframe_propagation.py --sweep 5,10,20,30,45,60`:
  content-adaptive frame selection and the viewpoint-stability measurement of
  §4.14 → `outputs/keyframe_propagation.json`. The sweep must include the
  coarse settings: the 89× compression figure §4.14 and §7.2 rely on comes
  from τ = 45, and because the script rewrites the whole file, a narrower
  sweep silently removes it
- `python eval/video_stability.py`: the persistence and Jaccard figures of
  §4.12, derived from the recorded per-frame files →
  `outputs/video_stability.json`
- `python eval/extension_scale.py`: throughput, density and predicate
  distribution on the unannotated capture (§4.15) →
  `outputs/extension_scale.json`
- `python scripts/run_vlm_pilot.py --make` then `python scripts/run_vlm_pilot.py`
  then `python eval/score_vlm_pilot.py`: the §4.16 vision-language
  comparison (30 numbered-box images, RQ1 battery, self-consistency
  diagnostics) → `outputs/vlm_pilot/`. The second model is the same three
  commands with `--model` and its own `--replies` file, then
  `python eval/compare_vlm_models.py` → `outputs/tables/vlm_models.md`.
  Reasoning models need `--max-output-tokens` above the 8192 default,
  because they spend most of the budget deliberating and a truncated reply
  is indistinguishable from a malformed one. `python
  scripts/vlm_manual_check.py` then writes a pack that lets the comparison
  be redone by hand in a browser: the image the model saw, the prompt
  verbatim, and the human, pipeline and model answers side by side, so the
  scoring script can be checked without running it.
- `python scripts/run_planner_llm.py` then `python eval/score_planner.py`:
  the §5.7 planner experiment (75 prompts over 25 scenes in three
  conditions) and its rule-based blind scoring →
  `outputs/planner/`, `outputs/planner_scores.json`. Both runner and scorer
  take `--replies`, which is how the second planner was run into
  `replies_pro.jsonl` and scored into `planner_scores_pro.json` without
  touching the first. Add `--sample N` to
  print plans with their verdicts for manual checking.
- `python eval/downstream.py --seeds 42,43,44`: the RQ2 experiment, all three
  label sources (human, self-trained, automatic) →
  `outputs/rq2_report.json`, `outputs/tables/rq2.md`
- `python analysis/score_votes.py votes.csv`: scores the independent
  validation study (crowd precision, author-bias kappa, rater reliability);
  lives with the study's own repository
- `python scripts/make_figures.py`: regenerate all figures from cached results
- `pytest -q`: unit and invariant tests

**The vision-language arms.** A fourth label source appears in §5.2, a fifth
condition in §5.7 and a third arm in §6.3.2, all derived from one model's
labels rather than from the pipeline's. They are reproduced separately here
because each begins with an API pass that costs money and cannot be repeated
byte-for-byte: the model behind `gemini-flash-latest` moved during the
project (§4.16), so a rerun answers as whatever that alias resolves to on the
day. The stored replies are therefore committed, and every scoring step below
reads them rather than re-querying, so all reported numbers reproduce offline
from this repository alone. Only step (a) needs a key.

- **(a) Label the training images.** `python scripts/run_vlm_pilot.py --make
  --n 600 --prompts outputs/vlm_pilot/prompts_train.jsonl` builds the
  prompts, then the same script without `--make` and with `--replies
  outputs/vlm_pilot/replies_train_f35.jsonl --model <model>` runs the pass.
  The committed replies are the ones every step below consumes.
- **(b) The RQ2 fourth arm (§5.2).** `python eval/downstream.py --seeds
  42,43,44 --vlm-replies outputs/vlm_pilot/replies_train_f35.jsonl --out
  outputs/rq2_report_vlm.json --table outputs/tables/rq2_vlm.md`. Note that
  the flag changes the *whole* experiment, not just the new column: every arm
  is restricted to the pairs the model covered, which is what makes the four
  columns comparable. That the human, self-trained and automatic figures come
  out identical to the three-arm run is the check that the restriction is
  fair, and it is why both JSONs are kept.
- **(c) The planner's conditions D and E (§5.7).** `python
  scripts/planner_experiment.py --vlm-replies
  outputs/vlm_pilot/replies_planner_pro.jsonl` rebuilds the prompt set with
  condition D (the model's relations through the identical filter) and
  condition E (the union of C and D); then `python scripts/run_planner_llm.py
  --model <model> --replies outputs/planner/replies_pro.jsonl` and `python
  eval/score_planner.py --replies outputs/planner/replies_pro.jsonl --out
  outputs/planner_scores_abcde.json`.
- **(d) The benchmark's third arm (§6.3.2).** `python
  scripts/export_sgg_benchmark.py --vlm-replies
  outputs/vlm_pilot/replies_train_f35.jsonl` emits a third annotation
  variant, `_annotations.vlm.coco.json`, alongside the human and automatic
  ones; `scripts/kaggle/notebook_cells_vlm.md` is the run recipe for training
  and evaluating that arm at three seeds, and `python eval/seed_stats.py`
  aggregates all three arms into `outputs/tables/seed_replication.md`.

The Chapter 6 experiment is reproducible from `scripts/kaggle/`: the dataset
converters (`export_sgg_benchmark.py`, `export_yolo_det.py`), the adapted
REACT++ configuration, the run recipes (`README.md`, `notebook_cells.md`,
`seed_replication.ipynb` for the seed replication and `notebook_cells_vlm.md`
for the vision-language arm), and
the executed evaluation notebook with its outputs (`eval_notebook.ipynb`),
which is the recorded provenance of every number in Chapter 6's tables.
Training logs and parsed results: `outputs/sgg_benchmark/`.

### Full reproduction walk-through

**1. Environment.** Either build the container (`docker build -t
spatial-annotator .`) or create a Python 3.11/3.12 venv and follow the three
numbered notes at the top of `requirements.txt`. The known pitfall is
documented in those notes: installing SAM2 can silently replace CUDA torch
with a CPU wheel, fixed by reinstalling torch with `--no-deps
--force-reinstall` from the cu121 index. Either way, verify with `python
scripts/smoke_test.py --image assets/sample.jpg`, which loads SAM2 and Depth
Anything and reports CUDA availability and peak memory (~0.65 GB).

The `Dockerfile` at the repository root pins Python 3.11, torch 2.5.1 +
cu121 and installs SAM2 from GitHub, applying that same fix in the build
itself, and a `.dockerignore` holds the build context to the 94 source files
that belong in an image, keeping the caches, the dataset and the credentials
file out of it. Both moving parts are pinned rather than tracked: the base
image by digest, since a tag can be re-pointed upstream, and SAM2, which has
no PyPI release, to commit `2b90b9f5`, since installing from a bare
repository URL fetches whatever `main` happens to be on the day of the
build.

The build has been run rather than merely written, and its log is kept at
`outputs/docker/build_summary.txt`: the base image in 193 s, system packages
in 31 s, and the Python dependencies, including SAM2 and the forced CUDA
torch reinstall, in 328 s. The final stage runs the project's test suite
*inside the container* and reports **66 passed** followed by a successful
import of the rule layer, so an environment that cannot import the rule
layer cannot produce an image at all.

The image is checked as an artefact rather than trusted on the strength of
its log, and the reason is worth recording as method. Two builds of this
image have failed from a full disk: one during the export stage, one during
unpacking, and the second wrote the tag before it died, so `docker images`
listed a plausible entry for an image that was never finished. A build log
and a tag are therefore not evidence; only the finished image is.

Inside the current image the test suite reports **66 passed**, torch resolves
to `2.5.1+cu121` rather than the CPU wheel the pitfall above produces, and
sam2, cv2, numpy, scipy, scikit-learn, Pillow and transformers all import.
`/app` is 19.3 MB, most of it the geometry cache, and the specification the
rules implement is the current one, checked on three strings that changed in
the last revision. The dataset and the credentials file are absent,
`.env.example` being the only environment file shipped, and so are the bulk
outputs the image has no use for: rendered annotations, the failure gallery,
the figures and the video frames.

The reproduction is the point. Mounting the dataset read-only, deleting
`outputs/pairs.csv` and running `scripts/reannotate_from_cache.py` inside the
container regenerates it at 84,881 rows with the SHA-256 of the file
committed here, and repeating it gives the same digest. Deleting first is
deliberate rather than tidy: the file ships inside the image, so a run that
failed silently would leave the committed copy in place and a naive check
would call that a reproduction. Twelve of the fifteen offline commands
complete in the container; the three that do not need roughly 200 MB of
GPU-produced intermediates the image deliberately omits, each of which
already has its committed JSON summary. `outputs/docker/verification.md`
records the full check, including the file-by-file integrity scan that
catches the half-written image a full disk produces.

Given a `--gpus all` flag the container also sees the card: CUDA reports
available, the RTX 2060 is enumerated, and a matrix product executes on the
device, so the GPU commands in the `Dockerfile` header are reachable rather
than aspirational.

The end-to-end check is the one that matters for reproducibility. Mounting
the dataset and the geometry cache read-only into the container and running
`scripts/reannotate_from_cache.py` re-annotated all 836 images and produced
a `pairs.csv` byte-identical to the one in this repository, 84,881 rows
sharing its SHA-256, together with the same 2,508 annotation files. The
container therefore does not merely install; it reproduces the annotations
this dissertation reports, exactly.

**2. Data.** Clone the released dataset (CC-BY 4.0) and point
`dataset.root` in `configs/default.yaml` at it. The loader expects the
release's own layout (`img_data/group_N/*.jpg` plus the annotation JSONs)
and corrects the images' 180° EXIF orientation itself; nothing is
preprocessed on disk.

**3. Everything below is offline, and the cache ships.**
`python scripts/run_annotator.py` (~5 min on the RTX 2060) writes the
annotations in all three native formats plus the geometry, contact and depth
caches under `outputs/`, and `outputs/pairs.csv`. That pass is the only step
needing a GPU, and it does not have to be repeated: the geometry cache and
`pairs.csv` are committed to the repository (10 MB, 1,672 files), so every
experiment below runs on a CPU with no perception models installed and no
GPU present.

Four of those commands still read the dataset's annotation files, because
they iterate the released images rather than the cache index:
`reannotate_from_cache.py`, `eval/ablations.py`,
`eval/keyframe_propagation.py` and `eval/depth_ablation.py`. They need
`annotated_data/` (17 MB) but never open a JPEG, so the images are not
required for any of them. The rest, including the whole RQ1 battery, run
from the committed caches and JSON alone; this was checked by running them
in a container with nothing mounted.

The check that the perception stage reproduces is
`scripts/reannotate_from_cache.py`, which rebuilds `pairs.csv` from the
cache and should return the identical file: 84,881 rows, SHA-256
`60281435…e1bd`.

| command | produces | time |
|---|---|---|
| `pytest -q` | 66 unit and invariant tests | <1 min |
| `python scripts/reannotate_from_cache.py` | re-runs the rules after any threshold change | ~20 s |
| `python eval/fit_near.py` | the near-threshold protocol, `near_fit.json` | ~1 min |
| `python eval/fidelity.py` | the RQ1 battery, `fidelity_report.json` | ~2 min |
| `python eval/uncertainty.py --iters 2000` | cluster-bootstrap CIs | ~2 min |
| `python eval/annotator_agreement.py` | heterogeneity + Fréchet bounds | <1 min |
| `python eval/keyframe_propagation.py --sweep 5,10,20,30,45,60` | §4.14 segmentation, stability, propagation cost | ~3 min |
| `python eval/ablations.py` | A1–A7 sweeps | ~30 s |
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

This is the complete geometric specification of the seven predicates: the
per-object measurements every rule reads, the rule for each predicate with its
thresholds and the evidence behind them, the correction step, and the flagging
policy. It is the operational definition §4.7.1 establishes the dataset never
had, and it is the reference `src/predicates.py` implements. The same text is
maintained in the repository at `docs/predicate_spec.md`, which is what the
code and the tests are checked against; the two are kept in step deliberately,
because a specification that drifts from its implementation is worse than none.

The seven predicates are exactly those of the source dataset (Wang et al.,
2025). The dataset's exact predicate strings are `to the left of` and `to the
right of`; the shorthand "left of" and "right of" is used below only for
readability. The canonical names and their dataset IDs are fixed in
`src/predicates.PREDICATE_IDS`: in front of = 6, behind = 2, on = 10,
under = 17, to the left of = 15, to the right of = 16, near = 9.

### C.1 Notation and the per-object measurements

After detection, segmentation and depth estimation, each object carries:

- `box = (x1, y1, x2, y2)`, the axis-aligned bounding box in image pixels,
  from the detector or from the ground-truth annotation.
- `M`, the binary segmentation mask from SAM2.
- `(cx, cy)`, the mask centroid in image coordinates.
- `d`, the per-object relative depth, Depth Anything v2 sampled over `M` and
  reduced by median.
- `P = (X, Y, Z)`, the lifted 3D position: normalised image coordinates plus
  scaled depth, retained per object for the downstream classifier's features.

**Image coordinate convention.** `x` increases to the right and `y` increases
downward, the standard image convention. Left and right are expressed in the
**camera frame**, as the camera sees the scene, matching how the human
annotators clicked subject then object on screen. This is stated explicitly
because it is a design choice that has to be justified against the annotation
tool rather than assumed, and §2.9 records that it is a decision and not a
fact about the world.

**Depth convention.** Depth Anything v2 returns a relative, ordinal depth map.
Smaller `d` is treated as nearer the camera. Because depth is relative and
per-image, every depth comparison is ordinal and between objects in the same
image; no rule consumes an absolute distance, and none may be compared across
images.

**Normalisation.** Distances and gaps are normalised by image dimensions, `x`
by width and `y` by height, so one threshold transfers across image sizes. The
`near` rule uses a size-relative 2D box gap rather than the lifted position
(C.7).

All predicates are defined for an **ordered pair (A, B)**, A the subject and B
the object, matching the dataset's subject-predicate-object triplet format.

### C.2 `on(A, B)`: A rests on B

A is `on` B when A sits directly above B, the two nearly touch, and they
overlap horizontally. Four conditions, all required.

1. **Above.** A is the higher object: `cy_A < cy_B`, with the bottom edge of A
   near the top of B.
2. **Touching.** The normalised vertical gap between the bottom of A and the
   top of B is at most `on_vertical_gap` (0.05). A small but non-negative gap
   captures resting contact while tolerating mask noise.
3. **Horizontal overlap.** The horizontal extents overlap by at least
   `on_horizontal_overlap` (0.20) of the **narrower** box's x-extent. This is a
   containment fraction rather than an IoU, so a small object sitting fully
   above a large one scores 1.0, which is the behaviour support requires.
   Something resting on another object must share its footprint.
4. **Depth co-location.** `|d_A - d_B| <= on_depth_eps` (0.06, calibrated on
   the training annotator groups). On a floor plane "farther" projects as
   "higher in the image", so an object *behind* another mimics the 2D
   signature of one stacked *on* it; truly stacked objects share a camera
   distance. Measured effect: held-out support F1 0.58 to 0.71, removing about
   half the audited false support labels at a cost of two recalled triplets.

Encoding support as *above, touching and horizontally overlapping* is what
stops a cup floating in front of a shelf from reading as resting on it.

**Mask-contact evidence, primary where masks exist.** The box conditions above
are the no-mask fallback. With SAM2 masks the rule uses the physical support
signature instead: `contact_below(A, B)` is the fraction of A's mask-bottom
columns with B's mask within five pixels below (`src/contact.py`), and `on`
requires `contact >= on_contact_min` (0.60, calibrated on the training groups,
with a flat optimum from 0.60 to 0.80) together with the depth gate and the
centroid order. This recovers the containment case the box test misses, nested
boxes at shallow viewing angles, formerly 79 to 88% of support misses, and
rejects cluster neighbours whose boxes touch but whose masks do not. Measured:
held-out support F1 0.71 to 0.87, and re-audited extra-label precision 0.07 to
0.73 for `on` and 0.20 to 0.80 for `under`.

**Class guard.** Classes listed in `no_support_classes` (shipped value:
`human`) are excluded from `on` and `under` in either role. The justification
is measured rather than assumed: the annotators never recorded a person as
supporting or being supported, on 0 of 2,466 gold support triplets, and mask
contact alone cannot distinguish a person *holding* an object from a surface
*supporting* one. The guard is a configuration entry rather than a hard-coded
name, so a deployment with different classes revises it in one line. The
residual failure mode it does not cover is one object held by another
non-guarded object, which remains a documented refinement.

### C.3 `under(A, B)`: A is below and supports B

`under` is the strict inverse of `on`, computed by evaluating the `on`
conditions with the arguments swapped:

```
under(A, B)  ==  on(B, A)
```

This guarantees consistency by construction: a pair can never be both
`on(A,B)` and `under(A,B)`.

### C.4 `left of(A, B)`: A's centre is left of B's

```
left_of(A, B)  ==  (cx_A < cx_B)    in camera-frame image coordinates
```

The magnitude `|cx_A - cx_B|`, normalised by width, gives the confidence. Below
`lateral_center_eps` (0.02) the centres nearly coincide, and the pair is
flagged ambiguous rather than silently labelled.

### C.5 `right of(A, B)`: A's centre is right of B's

```
right_of(A, B)  ==  (cx_A > cx_B)
```

The strict mirror of `left of`. Exactly one of the two holds outside the
ambiguity band; inside it neither is emitted and the pair is flagged.
`right_of(A,B) == left_of(B,A)`.

### C.6 `in front of(A, B)` and `behind(A, B)`

A two-stage cascade. Stage one is depth ordering:

```
in_front_of(A, B)  ==  (d_A < d_B)      (smaller depth = nearer)
behind(A, B)       ==  (d_A > d_B)
```

with `|d_A - d_B|` below `depth_eps` (0.03) meaning the depths are too close
to separate, which defers to stage two.

Stage two is the **ground-plane fallback**, which fires only where stage one
abstained. Two objects standing on the same floor are depth-ordered by
projection: the nearer object's box bottom sits lower in the image.

```
plane(A, B)  ==  (y2_A - y2_B) >  plane_band   =>  in front of
             ==  (y2_A - y2_B) < -plane_band   =>  behind
```

with `plane_band` = 0.005 of normalised image height, calibrated on the
training groups (ablation A7). Two guards apply. The fallback fires only when
neither object is elevated, meaning neither has mask contact at or above
`on_contact_min` with any partner, because an elevated object's box bottom
locates its support rather than itself; and only when mask evidence exists at
all, so it is off in box-only mode. Pairs both stages abstain on are flagged,
never guessed.

`behind` is the inverse throughout: `behind(A,B) == in_front_of(B,A)`.

Worked example, encoded in `tests/test_predicates.py`: a bottle with box
bottom 0.80 against a book with box bottom 0.60, depths 0.50 and 0.51 and so
inside `depth_eps`, both floor-standing, gives bottle `in front of` book. The
same pair with either object elevated is flagged instead.

### C.7 `near(A, B)`: A and B are close, relative to their size

```
near(A, B)  ==  box_gap_rel(A, B) <= near_T   AND   not (on(A,B) or under(A,B))
```

`box_gap_rel` is the edge-to-edge gap between the two normalised boxes divided
by the mean object size, the square root of box area. Proximity therefore
scales with the objects: a small gap between two books reads as near, while
the same absolute gap between a person and a cube may not. The gap is zero
when boxes touch or overlap, and `near` is symmetric,
`near(A,B) == near(B,A)`.

**Intended semantics.** Confirmed with the supervising group: `near` meant
"next to", the annotation team having merged an earlier separate "next to"
label into it. An adjacency reading supports both the gap metric and the
contact behaviour below.

**Contact exclusion.** Measured on the human labels, `near` co-occurs with
`on` or `under` on 0 of 469 pairs, and 74% of near pairs carry only `near`:
the annotators used it to mean close but with no contact relation. The rule
encodes that, so a pair already labelled `on` or `under` is never additionally
`near`.

**Why not 3D centroid distance.** Monocular depth is normalised per image, so
centroid distances are not comparable across scenes. In a metric comparison
every 3D-centroid variant transferred to held-out annotators at F1 at or below
0.024, while the size-relative gap transfers with recall 1.0.

**Fitting protocol, annotator-aware.** Only three of the nine annotator groups
ever used `near` (group_0: 244 labels, group_4: 129, group_8: 93; the rest
between 0 and 3). `near_T` is therefore fitted on the near-using groups inside
the training split (groups 0 and 4), on human-annotated non-contact pairs
only, and evaluated on the held-out near-using annotator (group_8). The fitted
value is **T = 1.372**, with held-out recall **1.000**: every pair that unseen
annotator called near falls within the threshold. Per-annotator precision at
the same T ranges from 0.16 to 0.63, which reflects how exhaustively each
annotator applied the label rather than any geometric disagreement. Cases
within `flag_near_band` (0.15 gap units) of `near_T` are flagged for optional
review.

### C.8 The correction step

Geometric consistency is enforced at two levels, and it is worth being precise
about which is which.

**By construction.** With the shipped rules the three mutually exclusive
families can never co-occur on one ordered pair: `on(A,B)` and `under(A,B)`
require strictly opposite centroid orderings, and left/right and front/behind
use strict comparisons separated by an ambiguity band. Inverses mirror exactly
across the two orderings. These invariants are pinned by a randomised test
over two thousand synthetic scenes (`tests/test_invariants.py`), so the
runtime conflict check is an assertion guarding future rule variants rather
than a filter that fires in practice.

**Active corrections**, adapted from Open3D-VQA's error-correction flow
(Zhang et al., 2025): `near` is suppressed on contact pairs (C.7), and
ambiguous cases are abstained and flagged rather than guessed (C.9).

### C.9 Confidence flags, and the honest cost

A pair is flagged, not dropped, when any of the following holds: the `near`
gap is within `flag_near_band` of `near_T`; the left/right centres are within
`lateral_center_eps`; or the front/behind depths are within `depth_eps` *and*
the ground-plane fallback cannot decide, because an object is elevated,
because no mask evidence exists, or because the bottom edges tie within
`plane_band`.

Flags are written alongside the triplets. Measured on the full dataset with
the shipped rules, 31.5% of ordered pairs carry some flag: `depth_ambiguous`
19.3%, `lateral_ambiguous` 10.0% and `near_threshold_edge` 8.5%, where a pair
may carry more than one. The depth share was 29.5% before the ground-plane
fallback shipped, which resolved about a third of the depth abstentions.

The flag types serve different purposes and are reported separately, which is
what makes the human-in-the-loop claim costable. Depth and lateral flags mark
*abstentions*: no wrong label was emitted, so there is nothing to review, and
the human annotators typically did not label those pairs either.
`near_threshold_edge` is the genuine *review queue*. Section 4.7 costs the
claim on the review queue alone rather than on the flag total, because
counting abstentions as human work would overstate the cost by a factor of
nearly four.

### C.10 Summary

| Predicate | Core test on the ordered pair (A, B) | Thresholds, shipped values | Symmetry |
|---|---|---|---|
| `on` | mask contact below, depth co-location, centroid order; box test is the no-mask fallback | `on_contact_min` 0.60, `on_depth_eps` 0.06, `on_vertical_gap` 0.05, `on_horizontal_overlap` 0.20 | `on(A,B) = under(B,A)` |
| `under` | inverse of `on` | as `on` | `under(A,B) = on(B,A)` |
| `left of` | `cx_A < cx_B` | `lateral_center_eps` 0.02 | `left(A,B) = right(B,A)` |
| `right of` | `cx_A > cx_B` | `lateral_center_eps` 0.02 | `right(A,B) = left(B,A)` |
| `in front of` | `d_A < d_B`, then the guarded ground-plane fallback | `depth_eps` 0.03, `plane_band` 0.005 | `front(A,B) = behind(B,A)` |
| `behind` | `d_A > d_B`, then the guarded ground-plane fallback | `depth_eps` 0.03, `plane_band` 0.005 | `behind(A,B) = front(B,A)` |
| `near` | `box_gap_rel <= near_T`, never on a contact pair | `near_T` 1.372 (fitted), `flag_near_band` 0.15 | symmetric |

Every threshold above is declared in `configs/default.yaml`, so no constant
that affects a label is buried in a function. `near_T` is fitted by
`eval/fit_near.py` under the protocol of C.7 and frozen there.

## Appendix D: Ablation derivations

Chapter 4 (§4.9) summarises the nine ablations and their verdicts. This
appendix gives the derivations: how each shipped parameter was calibrated on
the training annotator groups, what the held-out groups then reported, and
what the audits of the affected predictions found. The two declined
refinements are included in full, because a negative result is only useful
if the reader can see what was actually tried.

### D.1 The support depth co-location gate (ablation A1)

The audit's support-precision failure has a geometric cause: on a floor
plane, "farther" projects as "higher in the image", so a behind-pair
produces the same 2D box signature as a stacked pair. The repair is a depth
co-location condition on `on`/`under` (truly stacked objects share a camera
distance), an instance of the reject-the-geometrically-impossible correction
principle adapted from Open3D-VQA (Zhang et al., 2025; §2.5), calibrated on
the train groups (`on_depth_eps` = 0.06) and validated held-out (ablation
A1): support F1 on never-seen annotators rises 0.58 → 0.71. Downstream
effects on the headline table: support recall −2/−3 points, `on` restricted
precision 0.57 → 0.73, 44% fewer support emissions (of the 26 audited false
positives, the gate removes 12 while keeping all 4 true positives), and,
because fewer false contacts suppress fewer proximity labels, `near` recall
rises 0.87 → 0.95 (held-out 1.00). Mean recall is unchanged at 0.79 with a
substantially more trustworthy label set. The residual false fires are
same-depth cluster neighbours, which depth cannot separate by construction;
they are addressed next.

### D.2 The mask-contact rule (ablation A5)

The support signature that boxes cannot see, masks can: A rests on B iff the
pixels directly below A's mask-bottom boundary belong to B
(`src/contact.py`), which captures both stacking and the containment case,
and rejects side-by-side neighbours. Calibrated on the train groups
(`on_contact_min` = 0.60; the train-F1 plateau is flat from 0.60–0.80, so
the choice is uncritical), ablation A5: support F1 on held-out annotators
rises again, 0.71 → **0.87**, with `on` recall 0.82 → 0.88 and restricted
precision 0.73 → 0.88 simultaneously, the rare change that improves both
error directions at once, exactly as the failure gallery and audit
predicted. Knock-on: `near` recall reaches **0.997 pooled (715/717; the two
residual misses are contact-boundary cases, §4.10) and 1.00 held-out** as
the last contact-boundary suppressions disappear; headline mean recall 0.79
→ **0.81**. The A4 sweep confirms the fitted threshold sits exactly at the
recall plateau's knee: recall is flat from T = 1.372 upward while emissions
keep growing, so the fitted value is the least-permissive point achieving
maximal agreement. A 30-sample re-audit of the new support extras confirms
the precision claim independently: extras correct rise from 1/15 and 3/15
(box rule) to **11/15 and 12/15**, an estimated true support precision of
~0.27 → ~0.9. The seven remaining wrong/uncertain extras have structure: a
person *holding* a remote fires contact (holding ≠ resting), one occluded
bottle-behind-bottle pair, and three distant clusters too small to verdict
confidently. The person-holding mode is closed by a **class-aware guard**:
annotators never label person-support (0 of 2,466 gold support triplets
involve a person on either side), so `on`/`under` are simply not evaluated
for that class, removing ~130 false emissions at no recall cost, since the
person side carried no gold to recover.

### D.3 The ground-plane fallback (ablation A7)

The depth abstention band was the
single largest miss cause, and most of it is resolvable without depth at all:
two objects standing on the same floor are depth-ordered by pure projection.
The nearer object's box bottom sits lower in the image, a pixel-precise cue
exactly where relative depth is noisiest. The shipped rule fires only where
the depth rule abstained, only when *neither* object rests on another object
by the tool's own contact evidence (an elevated object's box bottom locates
its support, not itself), and only outside a small bottom-edge band
(`plane_band` = 0.005, calibrated on the train groups; ablation A7). On the
train groups the fallback adds 386 committed directions at 0.91 agreement; on
held-out group 7 it adds 54 and **every one agrees with the annotator**.
Effect on the headline: front/behind recall 0.52/0.55 → **0.64/0.66** (aligned
overall 0.67 → **0.84**), mean recall 0.81 → **0.85**, and the depth-ambiguous
flag rate falls 29.5% → 19.3%. A seeded 15-sample audit of the fallback's
*extra* predictions (pairs no human labelled) estimates true precision
conservatively at **11/15 ≈ 0.73**, and the four wrong/uncertain cases share
one structure: an object resting on something the detector has no box for (a
book on an unannotated case; a stacked-book pair whose contact fell below
threshold), i.e. elevation the guard cannot see. That residual mode is
documented, bounded, and exactly the undetected-support refinement the support
audit already motivates.

### D.4 Follow-up refinements: one shipped, two declined

The
class-aware guard *shipped*: annotators never label person-support (0 of
2,466 gold triplets), so support is no longer evaluated when either object is
a person, removing ~130 held-object emissions at zero recall cost and
eliminating the person-holding audit mode by construction. Two further
candidates were built, measured, and rejected on the evidence. (i)
*Guard-only surface detection* (zero-shot prompts for tables, cases and trays
feeding the elevation guard; `scripts/run_surface_guard.py`) suppressed 6% of
the fallback's extra commits at a cost of 0.01 behind recall, but blocked
none of the four audited failures, whose supports are either annotated-class
objects with below-threshold contact (flat stacked books) or surfaces the
detector missed; the machinery and its cache are retained as an ablation
(`outputs/surface_guard_ablation/`). (ii) *A larger depth model* (Depth
Anything v2 Base, 4× the parameters) moved front/behind recall by
+0.001/+0.002: the remaining depth ambiguity lives in the scenes, not in
model capacity, and the Small variant's Apache licence is kept. Both null
results bound where further engineering can and cannot help.

### D.5 Why a geometric cue, not a bigger depth model (ablation A8)

It is worth asking whether the depth pair would improve simply by using a
stronger depth network. It does not: swapping Depth Anything v2 Small for
the 4× larger Base variant and re-running the whole dataset moves
front/behind recall by +0.001 and +0.002, from 0.640/0.654 to 0.641/0.656,
while mean recall *falls* from 0.848 to 0.847. (This ablation is its own
end-to-end run, so its baseline sits about 0.0008 below the headline figures
of §4.2, which is why `behind` appears here as 0.654 and there as 0.66. The
offset is smaller than any quantity being compared and applies to both arms
equally, so it cannot affect the verdict.) The depth-predicate limit is
*monocular ambiguity* (two objects at a similar camera distance are
inseparable by any monocular model, regardless of its fidelity), not the
network's quality. This is precisely why the fallback that worked is a
geometric projection cue rather than a heavier perception model, and it
justifies shipping the Small variant: identical accuracy, an Apache-2.0
licence, and half the VRAM.


### D.6 Why not multi-frame depth either (ablation A9)

Ablation A8 rules out a bigger *monocular* model, but it leaves the sharper
objection open. The images are consecutive frames of a robot walk (§4.14), so
neighbouring frames exist that view the same scene from a different camera
position. Two views constrain depth geometrically, without any learned prior,
and that is the classical remedy for exactly the ambiguity A8 blames. If the
front/behind shortfall is a depth problem, this is what should fix it.

Two multi-frame estimators were built and scored against the same gold, on
the seven annotator groups that use the standard direction convention, so
that a geometrically correct answer is not penalised by the inversion of §4.5
(`eval/parallax_ablation.py`).

The first reads *residual displacement*: under camera translation a nearer
object sweeps further across the image, so after subtracting the median
motion, which is the part rotation contributes equally to every object,
the larger residual should be the nearer object. It performs at chance,
0.445 to 0.512 across baselines from 5 to 60 frames. The reason is that the
assumption is wrong for this motion: a walking robot translates forward as
well as sideways, and forward motion produces flow radiating from the focus
of expansion, where displacement depends on an object's distance from that
point as much as on its depth. An object near the focus of expansion barely
moves however close it is.

The second recovers the camera pose properly, estimating the essential matrix
from whole-image correspondences, decomposing it into rotation and
translation, and triangulating the points inside each object's box. Focal
length is assumed rather than calibrated, which is tolerable because only the
depth *ordering* is read and that is insensitive to moderate focal error.
This is the correct construction and it does much better than the first, but
not well enough:

| Depth cue | Ordering accuracy | Pairs answered |
|---|---|---|
| Two-view triangulation (10-frame baseline) | 0.706 | 337 of 3,597 |
| Monocular cascade, same pairs | 0.875 | 337 of 3,597 |

Both numbers fall on the same pairs, so the comparison is like for like. The
multi-frame estimate is 0.17 worse where it applies, and the two disagree on
27% of pairs, with the monocular cascade right more often in that
disagreement. Coverage is the harsher problem: triangulation returns an
answer for 9% of the depth-labelled pairs, because it needs several trackable
corners inside an object's box and the dataset's objects are small,
low-texture cubes and boxes photographed at 640×480 in greyscale. Widening
the baseline trades coverage for geometry without helping accuracy: at 40
frames, accuracy falls to 0.610.

Splitting those pairs by how far apart the two triangulated depths actually
are says more than the aggregate does, and it is the check that separates a
weak estimator from a broken one:

| Relative depth separation | Pairs | Ordering accuracy |
|---|---|---|
| 0.000–0.014 | 86 | 0.453 |
| 0.014–0.045 | 85 | 0.776 |
| 0.045–0.124 | 86 | 0.895 |
| 0.124 and above | 86 | 0.721 |

The estimator is real. Given objects at moderately different depths it
reaches 0.895, which is the monocular cascade's own accuracy on this slice,
so the implementation is not the thing holding it back. What it cannot do is
the part that matters: on the quartile of pairs whose depths are nearly
equal it performs at chance, 0.453. Those are precisely the pairs the
monocular cascade abstains on, and the reason is the same for both methods.
Two objects at genuinely similar camera distance are not separated by
measuring depth more carefully, because the quantity being measured is
almost the same for each of them. Multi-view geometry inherits that limit
rather than removing it. The fall to 0.721 in the top quartile is the
opposite failure and a real weakness of two-view triangulation: a handful of
badly conditioned points produce depths that are wrong *and* far apart, so
the largest separations include the worst outliers.

The honest scope of this result is that it rules out the cheap version of the
idea rather than the idea itself. A careful multi-view reconstruction over
many frames, with bundle adjustment and real intrinsics, would estimate depth
better than two views and an assumed focal length. What the ablation
establishes is that geometric depth is not free here, that the obvious
implementations lose to the monocular cascade on both accuracy and coverage,
and, read with §4.14, that the return on any of them is bounded: a predicate
that already reproduces its own verdict 0.955 of the time across viewpoints
does not have much room to gain from measuring depth more precisely. There is
also a design cost that no accuracy figure captures. Every multi-frame method
requires neighbouring frames at inference time, which the single-image
annotator this project set out to build does not have.

## Appendix E: Extended validation studies

Five studies report their headline result in the chapter that owns them and
their supporting detail here: the vision-language baseline of §4.16, whose
diagnostics are what make its failure interpretable rather than merely worse
(E.1); the viewpoint-stability measurement of §4.14, whose segmentation
evidence and coverage limits qualify how far the stability figures reach
(E.2); the independent precision study of §4.13, whose instrument can be
judged before its results exist (E.3); the video processing of §4.12, whose
settings and open-vocabulary failures qualify the only out-of-domain
evidence in the dissertation (E.4); and the planner experiment of §5.7,
whose relation filter and blind scorer are what the comparison rests on
(E.5).

### E.1 The vision-language baseline: diagnostics and limits

The recall, precision and F1 tables this section refers to are in §4.16,
which reports the headline comparison over two models: they recover 0.40 and
0.45 of the human triplets against the pipeline's 0.83, and are the more
precise labellers on the judged pairs, 0.42 and 0.39 against 0.35, while
losing F1 on every predicate. The diagnostics below are the smaller model's
unless stated, since it is the one the manual-check pack was built for.

One asymmetry must be stated or the precision comparison will be read as
stronger than it is. The pipeline's apparent false positives were audited
(§4.4) and found to be largely *correct but unrecorded*, relations that hold
geometrically which the annotators did not write down, giving audited true
precision near 1.0 on the lateral and proximity predicates. Its 0.35 here is
therefore a floor set by sparse gold rather than a measure of error. No
equivalent audit exists for the model's assertions, so its 0.42 is not known
to be a floor in the same way. The two are comparable as agreement with the
human record, and are not comparable as truthfulness.

**How it fails is the interesting part, and it is not the way a
badly-calibrated model fails.** Three diagnostics separate wrong answers
from absent ones.

It does not contradict itself: across 1,187 emitted relations there is not
one case of a pair given both a predicate and its opposite. Nor is its
front/behind convention inverted in the manner of annotator groups 6 and 8;
only 8 of 145 depth-pair misses name the opposite direction, against a rate
approaching 1.0 for a genuine convention flip.

What it does instead is **fall silent**. Of the human triplets it misses,
the majority are pairs on which it said nothing at all: 27 of 49 for *to the
left of*, 31 of 49 for *to the right of*, 45 of 80 for *in front of*. It
emits about 40 relations per image where the pipeline emits several hundred.

And its silence is uneven in a familiar way. It asserts *to the left of* 374
times but supplies the matching *to the right of* on the swapped pair in
only 65% of those cases, leaving 131 assertions whose inverse it never
states, while support relations carry their inverse 100% of the time. That
is the same defect §4.5 measures in the human annotation, where one group
recorded 188 instances of *on* and no instances of *under*.

Section 4.16 draws the conclusion these diagnostics support. One point
belongs here rather than there, because it is a property of the pipeline
rather than of the model: what the pipeline has over it is not fluency and
not per-assertion agreement, but exhaustiveness and the guaranteed
anti-symmetry of §3.6.

Three limits on this result, one of them now partly settled. The pilot is
thirty images at one prompt, and model capacity was the obvious confound: a
larger model might simply have closed the gap. Running the identical battery
on a reasoning model an order of magnitude larger tests that directly, and
the answer is that capacity moves the numbers without moving the verdict.
Mean recall rises from 0.400 to 0.445 against the pipeline's 0.834, pooled
F1 from 0.397 to 0.405 against 0.488, and the larger model buys its extra
recall by asserting more, which costs it precision, 0.389 against 0.419. It
is better in the way a more willing annotator is better, not in the way a
more accurate instrument is. What remains untested on this axis is prompting
and fine-tuning: chain-of-thought scaffolding or supervised adaptation could
still move the numbers, and neither was attempted.

The other two limits stand unchanged. The prompt asks for every pair that
stands in a relationship, so part of the silence may be the model's own
judgement about what is worth recording, which is itself the annotator
behaviour under discussion rather than an artefact. And the recall column
rewards density, which is why the precision comparison is reported beside
it; neither column alone is a verdict.

One diagnostic does improve with scale, and it is the one that matters for
the argument §4.16 makes. The smaller model supplied *to the left of*
without its inverse on 0.35 of its assertions; the larger one on 0.16.
Deliberation buys internal consistency. It does not buy enough: 0.16 is
still a sixth of a symmetric relation asserted in one direction only, a
defect the geometric rules cannot exhibit at all because §3.6 enforces the
inverse by construction.

### E.2 Viewpoint stability: segmentation evidence and coverage

*Segmentation.* At τ = 10 the full 2,650-frame sequence collapses to 892
segments, a 3.0× reduction; over the 884 released frames it gives 331, 2.7×.
Scored on the released portion, where the eight layout changes are known,
every one is recovered within five frames (boundary recall 1.00). Precision
against those eight is low by construction and not a defect: viewpoint
changes within a block are genuine content changes, merely finer-grained.
The standard alternative has no usable operating point at all on this
material, for the reason §3.10 gives: thresholding consecutive-frame
differences at 15 grey levels fires 13 times and finds none of the eight,
and at 5 it finds all eight among 604 firings, a precision of 0.013.

**Limits.** Two, both restricting scope rather than direction. Stability is
computed only on pairs that could be matched between keyframe and frame, and
those are the pairs whose objects moved least in the image, plausibly the
easier population, so the figures are an upper bound. And matched coverage
falls as segments grow, from 88.7% of pairs at τ = 5 to 58.5% at τ = 10 and
26.5% at τ = 20, so aggressive compression leaves most pairs uncovered
rather than mislabelled. The cause is box drift under camera motion rather
than missing annotation: at τ = 20, 65% of skipped frames carry an identical
object count to their keyframe, and relaxing the overlap criterion from 0.5
to 0.1 cuts unmatched objects from 5.6 to 2.2 per frame. Carrying object
identity with a tracker, as `scripts/run_video.py` already does for video,
rather than with per-frame overlap, is the straightforward extension that
would recover it.

### E.3 The independent validation study: design and scoring

§4.13 states the weakness this study addresses and its status. What follows
is the design, recorded here so that a reader can judge the instrument before
its results exist.

**Sampling.** A stratified sample of 2,002 automatic labels, 286 per
predicate, drawn from the tool's *extra* predictions: ordered pairs the human
annotators never labelled, which is exactly the population with no ground
truth to score against. Each claim is rendered as the source photograph with
the subject outlined in red and the object in blue, presented with a single
sentence ("the book is on the box"), and answered TRUE or WRONG / can't tell
by volunteers recruited through an open link. The instructions restate
Chapter 3's operational definitions: camera-frame laterality, "in front of"
as nearer the camera, support as physically resting rather than held, and an
explicit instruction to answer WRONG when unsure, which reproduces the
conservative rule used in the author's own audits. Each browser receives a
random identifier that prevents repeat judgements without identifying
anybody, and faces are anonymised in every image (Chapter 8).

**Coverage.** Stratified by what each analysis requires. An aggregate
precision estimate needs only one judgement per claim, since the sample is
random either way, so ordinary claims target a single rater. The 147 claims
that also carry an author verdict target three raters each, because the
crowd-versus-author comparison and the inter-rater reliability figure both
need several independent judgements on the *same* item; those claims are
served first. The design therefore needs about 2,300 judgements rather than
the 6,000 a uniform three-rater target would demand, and it fails gracefully:
the author-comparison subset completes after roughly 30 participants, so that
analysis survives a thin turnout while any further response widens the
precision estimate.

**Scoring**, fully specified in advance (`analysis/score_votes.py`): ties
resolve to WRONG, matching the audit protocol; reflex-speed responses and
raters who disagree systematically with everyone else can be excluded by
pre-declared filters. Crowd precision is reported per predicate with binomial
intervals, author agreement as percentage and Cohen's kappa (Cohen, 1960),
and crowd-internal reliability as Krippendorff's alpha.

### E.4 Video processing: settings and the open-vocabulary failures

Section 4.12 reports what the two clips show. This section records how they
were processed and what went wrong, both of which qualify the reading.

**Processing.** Each frame is annotated independently by the deployment-mode
stack of §4.11 with open-vocabulary prompts; object identities are carried
between frames by greedy IoU tracking, and each pair's predicates are
smoothed by a plus-or-minus-two-frame temporal majority vote
(`scripts/run_video.py`; overlays and per-frame records in
`outputs/video/`). The vote is the only component with no counterpart in the
still-image pipeline, and its effect is measured in §4.12 rather than
assumed: a small gain on the static scene and a larger one where detection
churns, which is the behaviour a majority filter should show.

**Persistence and agreement.** Persistence is measured over pairs co-visible
in at least 20 frames, where co-visible means both track identities appear
in that frame; 81% and 89% of pair-predicates are present in at least 90% of
their co-visible frames. The frame-to-frame Jaccard dips in clip 2 align
with the hands picking objects up, which is the change the smoothing is
required not to erase.

**Open-vocabulary failures.** Two are plainly visible and worth recording,
because both are detection failures rather than relation failures and so
support the same attribution §4.11 makes for the dataset itself. Content
displayed *on the laptop screen* is detected as real objects standing in
real relations to the objects around it. And items outside the prompt list
snap to the nearest prompted class: an earbuds case is labelled a `cup`.
Neither is corrected, and both are present in the released overlays.

### E.5 The planner experiment: prompt construction and scoring rules

Section 5.7 reports the design in outline and the result. Two components
need recording in full, because the comparison is only as good as they are.

**The relation filter.** Both relation conditions are passed through one
filter before their relations are written into the prompt
(`scripts/planner_experiment.py`): it keeps the support relations among the
mentioned objects plus the target's one-hop neighbourhood, canonicalised and
deduplicated. Without that step the comparison would confound label quality
with prompt length. It does not equalise prompt length, and cannot: after
identical filtering the automatic condition still carries 22.6 relations per
prompt against the human condition's 3.1, because the automatic labels are
twenty times denser to begin with, and discarding true relations to match a
sparser source would be a different experiment. What the filter guarantees
is the weaker but sufficient property that neither source is given a
relation the other would have been denied. The task sentence, the object
list and the instruction to give a minimal numbered plan are identical
across conditions, and all conditions of a scene are answered by one model
in one sitting, so no scene can be split across models by a quota
interruption (`scripts/run_planner_llm.py`).

**The scoring defect that manual checking caught.** Models frequently open
with a preamble restating the task ("To pick up box0 safely, follow these
steps:"), and counting that preamble as a step made every such plan appear
to grasp the target before clearing anything. Before the fix condition C
scored 0.64; after it, 0.88. The figure below the fix is the correct one.
The episode is recorded because it is the only evidence that the blind
scorer measures what it claims to: a rule-based judge inherits whatever its
author failed to anticipate, and the hand-read sample is what exposed this
one.
