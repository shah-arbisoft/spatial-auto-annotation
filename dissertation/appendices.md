# Appendices

## Appendix A: Ethical Approval

The full ethics record is submitted as a **separate document, `ethics.pdf`**,
with the two signed forms attached to it: the University ethics
self-assessment, completed and submitted on 19 July 2026, and the Secondary
Data Checklist, countersigned by the supervisor before any analysis began.
This appendix records what that document establishes, so the dissertation can
be read without it.

The work is a secondary analysis of an existing, published dataset (Wang et
al., 2025; CC-BY 4.0) collected by the supervisor's research group; no new
personal data was collected for the annotation study. Scene images contain
identifiable people, so every figure reproduced here blurs faces and the
dataset is used strictly as released. The unreleased 1,766-frame remainder
of the capture, used in §4.12 and §9.3, is **not** covered by that licence:
it is used only to measure behaviour on unlabelled input, is not
redistributed, and no figure reproduces a frame from it. The independent
validation of Chapter 4 collects **pseudonymous** true/false judgements from
adult volunteers, with no names, email addresses or IP addresses, only a
per-browser random identifier used to spread coverage and drop duplicates,
which UK GDPR treats as personal data, which is why the study is described
as pseudonymous rather than anonymous here and in §1.3 and §8.3. Because no
directly identifying data is collected, no vulnerable groups are involved
and there is no foreseeable risk beyond that of everyday life, the study
falls within the self-assessment route rather than requiring full committee
review.

**Demonstration footage.** The two video clips of E.4 are royalty-free stock
footage from Pexels, used under the Pexels licence (free use, no attribution
required): clip 1 (desk scene, moving camera)
https://www.pexels.com/video/6558513/ and clip 2 (overhead desk, moving
hands)
https://www.pexels.com/video/a-person-working-with-pictures-and-photos-taken-using-a-modern-camera-3250234/.

## Appendix B: Reproducibility

Environment: Windows 11, Python 3.11.9 (virtual environment), PyTorch 2.5.1
(CUDA 12.1), single NVIDIA RTX 2060 (6 GB). All thresholds, seeds and model
identifiers are pinned in `configs/default.yaml`.

**The vision-language arms.** A fourth label source appears in §5.2, a fifth
condition in §5.7 and a third arm in §6.3.2, all derived from one model's
labels and not from the pipeline's. They are reproduced separately here
because each begins with an API pass that costs money and cannot be repeated
byte-for-byte: the model behind `gemini-flash-latest` moved during the
project (§4.13), so a rerun answers as whatever that alias resolves to on the
day. The stored replies are therefore committed, and every scoring step below
reads them rather than re-querying, so all reported numbers reproduce offline
from this repository alone. Only step (a) needs a key.

Four passes produce them, and the repository README carries the exact
invocations. Only the first, labelling 600 training images, needs a key; the
rest read the committed replies. Two of the four carry a caveat that belongs
here rather than in a README. In the RQ2 arm (§5.2) the `--vlm-replies` flag
changes the *whole* experiment and not just the new column, restricting every
arm to the pairs the model covered, which is what makes the four columns
comparable; that the human, self-trained and automatic figures come out
identical to the three-arm run is the check that the restriction is fair, and
it is why both JSONs are kept. In the planner (§5.7) condition D is the
model's relations through the identical filter and condition E the union of C
and D, so neither condition sees a filter the others did not.

The Chapter 6 experiment is reproducible from `scripts/kaggle/`: the dataset
converters (`export_sgg_benchmark.py`, `export_yolo_det.py`), the adapted
REACT++ configuration, the run recipes (`README.md`, `notebook_cells.md`,
`template_seed_replication.ipynb` for the seed replication and `notebook_cells_vlm.md`
for the vision-language arm), and
the executed evaluation notebook with its outputs (`executed_1b_eval_seed42.ipynb`),
which is the recorded provenance of every number in Chapter 6's tables.
Training logs and parsed results: `outputs/sgg_benchmark/`.

### How the repository makes the results re-runnable

Section 3.11 states the principle; the mechanics are here, because three of
the four requirements in §3.2 are unverifiable without them.

**Configuration and caching.** Every threshold, seed and model identifier
lives in `configs/default.yaml`, so no constant is buried in a function, and
the runner caches each object's lifted geometry after the single GPU pass.
That separates the expensive stage from the cheap one: any rule or threshold
change re-evaluates the whole dataset offline in about 20 seconds with no
GPU (`scripts/reannotate_from_cache.py`) against roughly five minutes for a
full perception run, which is what made the audit-driven rule repairs of
Chapter 4 affordable and let the ablation battery run as a sweep.

**Test strategy.** The suite is 66 tests running in about a second, which is
deliberate: a suite slow enough to skip constrains nothing. Worked examples
from the predicate specification are encoded as unit tests over the rule
layer so the two cannot drift apart silently; a randomised invariant test
fuzzes two thousand synthetic scenes against the structural guarantees §3.6
promises; and the rest cover the format writers, the detector adapters of
§3.9, frame selection and the reply parsers.

**The format writers, verified.** Section 3.7 gives the reason
byte-compatibility is a requirement of the research design; these are the
fields it comes down to. The writers reproduce the
SGDET-Annotate structure exactly: centre-form `boxes_1024`/`boxes_512` in
the resized frames, index-aligned `labels` and `attribute` arrays,
`relationships` as subject–object index pairs with a parallel `predicates`
ID array, and the same six-dataset h5 layout with int64 attributes. A
load→write round trip reproduces `boxes_1024` and `labels` with zero error,
and the h5 matches a real export key-for-key and dtype-for-dtype
*(measured)*, both checks sitting in the suite so a later change to a writer
cannot pass unnoticed.

**The detector contract.** The rule layer (`src/predicates.py`) imports
nothing but `numpy` and never receives an image, and the entry point takes
boxes as an argument, so no detector is wired into the pipeline. The
contract is explicit (`src/detectors.py`): one method returning pixel boxes,
class names and scores, with three implementations: open-vocabulary
prompting where no trained model exists, an adapter for any ultralytics
checkpoint including the source paper's YOLOv10m weights, and a reader for
externally computed detections. Twelve unit tests pin it, one driving the
rule layer from a detector written against the documentation alone. Two
coupling points are documented: the support guard keys on the literal class
name `human`, and the fitted thresholds assume boxes of roughly the
tightness the annotators drew, so a detector with systematically different
boxes should re-run §3.8's calibration (twenty seconds offline from the
cache). A worked example is in `docs/CUSTOM_DETECTOR.md`.

**Environment.** Python 3.11 with CUDA torch 2.5.1, pinned, and the one
genuinely awkward step documented instead of left to be rediscovered:
installing SAM2 can silently replace the CUDA build of torch with a CPU
wheel, so the pipeline still runs, produces identical labels and takes an
order of magnitude longer, which is the worst class of failure because
nothing reports it. A smoke test verifies on first setup that both models
load, that CUDA is in use, and that peak memory sits inside the 6 GB budget.

### Where each chapter's numbers come from

Every quantitative claim names the script that produced it and the artefact
it is reproducible from.

**Chapter 1.** Chapter summary: the manual-annotation bottleneck, the compute-not-predict idea, the research questions and objectives, and the shape of the argument.

**Chapter 2.** Facts attributed to the source paper are verified against its arXiv full text (2506.12525); all other cited works are verified against their published versions. Full bibliography entries: [references.md](references.md).

**Chapter 3.** Written alongside the build, so every design claim below is implemented and, where stated, measured. Numbers marked *(measured)* come from runs over the released dataset (836 annotated images); the fitting protocol and per-annotator analysis are reproducible from `eval/fit_near.py` and `outputs/near_fit.json`.

**Chapter 4.** All numbers are generated by `eval/fidelity.py` from the full 836-image run with the shipped rule set (including the support depth gate, §4.9) and are reproducible from `outputs/fidelity_report.json`. The audit (§4.4) predates the gate; §4.9 reports the gate's effect on the audited samples.

**Chapter 5.** Numbers generated by `eval/downstream.py` and, for §5.7, by `eval/score_planner.py`. The three-arm figures are reproducible from `outputs/rq2_report.json` and the four-arm table of §5.2 from `outputs/rq2_report_vlm.json`; the planner scores from `outputs/planner_scores.json`, with conditions D and E in `outputs/planner_scores_abcde.json`.

**Chapter 6.** Numbers generated on Kaggle (T4 GPU) with SGG-Benchmark; training logs and the exact per-epoch series in `outputs/sgg_benchmark/`; conversion by `scripts/export_sgg_benchmark.py`; run recipe in `scripts/kaggle/`.

**Chapter 7.** Every quantitative claim below is established in Chapters 4–6 and reproducible from the repository; this chapter interprets, connects and stress-tests them.

### Full reproduction walk-through

**1. Environment.** The container or a Python 3.11/3.12 virtual
environment; the README gives both recipes and `requirements.txt` the three
notes they depend on. One pitfall is worth naming because it fails silently:
installing SAM2 can replace CUDA torch with a CPU wheel, so the smoke test
reports peak memory (~0.65 GB) and CUDA availability instead of merely
importing.

The `Dockerfile` at the repository root pins Python 3.11, torch 2.5.1 +
cu121 and installs SAM2 from GitHub, applying that same fix in the build
itself, and a `.dockerignore` holds the build context to the 94 source files
that belong in an image, keeping the caches, the dataset and the credentials
file out of it. Both moving parts are pinned rather than tracked: the base
image by digest, since a tag can be re-pointed upstream, and SAM2, which has
no PyPI release, to commit `2b90b9f5`, since installing from a bare
repository URL fetches whatever `main` happens to be on the day of the
build.

The build has been run rather than merely written, and its log, with the
per-stage timings, is committed at `outputs/docker/build_summary.txt`. The
final stage runs the project's test suite *inside the container* and reports
**66 passed** followed by a successful import of the rule layer, so an
environment that cannot import the rule layer cannot produce an image at
all.

The image is checked as an artefact and not trusted on the strength of
its log, and the reason is worth recording as method. Two builds of this
image have failed from a full disk: one during the export stage, one during
unpacking, and the second wrote the tag before it died, so `docker images`
listed a plausible entry for an image that was never finished. A build log
and a tag are therefore not evidence; only the finished image is.

Inside the current image the test suite reports **66 passed**, torch
resolves to `2.5.1+cu121` rather than the CPU wheel the pitfall above
produces, every expected dependency imports, and the specification the rules
implement is the current one, checked on three strings that changed in the
last revision. What the image does *not* carry matters as much: no dataset,
no credentials file, `.env.example` alone, and none of the bulk outputs it
has no use for. The full inventory is in the README.

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

**2. Data.** The released dataset (CC-BY 4.0), read from its own layout
with `dataset.root` pointed at it. Nothing is preprocessed on disk: the
loader corrects the images' 180° EXIF orientation in memory, so no derived
copy exists that could drift from the release.

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
| `python scripts/reannotate_from_cache.py` | re-runs the rules after any threshold change; rebuilds `outputs/pairs.csv` | ~20 s |
| `python eval/fit_near.py` | the `near` threshold protocol, `near_fit.json` | ~1 min |
| `python eval/fidelity.py` | the RQ1 battery, `fidelity_report.json`, `tables/rq1_tables.md` | ~2 min |
| `python eval/uncertainty.py --iters 2000` | cluster-bootstrap intervals for every headline recall, `tables/uncertainty.md` | ~2 min |
| `python eval/annotator_agreement.py` | annotator heterogeneity and the Fréchet bounds, `tables/annotator_agreement.md` | <1 min |
| `python eval/keyframe_propagation.py --sweep 5,10,20,30,45,60` | §4.12 frame selection, stability and propagation cost, `keyframe_propagation.json` | ~3 min |
| `python eval/ablations.py` | ablations A1–A7, `tables/ablations.md` | ~30 s |
| `python eval/depth_ablation.py` | ablation A8; needs the `outputs_base` pass below | <1 min |
| `python eval/support_guard_ablation.py` | ablation A10, whether contact height can replace the class guard, `support_guard_ablation.json` | <1 min |
| `python eval/parallax_ablation.py --method triangulate --gap 10` | ablation A9; needs the raw capture (D.6), `parallax_ablation.json` | ~5 min |
| `python eval/parallax_ablation.py --method triangulate --gap 20 --focal-sweep 0.5,0.7,0.9,1.2,1.6,2.0` | D.6's focal-length sensitivity check, `parallax_focal_sweep.json` | ~25 min |
| `python eval/video_stability.py --dir outputs/video_085` | the E.4 persistence and Jaccard figures, `video_stability.json`; the `--dir` is required, since the default points at the pre-refit pass | <1 min |
| `python eval/extension_scale.py` | E.6 throughput, density and predicate distribution | <1 min |
| `python eval/seed_stats.py` | the benchmark arms aggregated across seeds, `tables/seed_replication.md` | <1 min |
| `python eval/downstream.py --seeds 42,43,44` | RQ2, all arms, `rq2_report.json`, `tables/rq2.md` | ~20 min |
| `python eval/score_vlm_pilot.py`, then `python eval/compare_vlm_models.py` | the §4.13 comparison and `tables/vlm_models.md`, from the stored replies | <1 min |
| `python eval/crowd_validation.py` | the §4.15 volunteer comparison, `crowd_validation.json`; needs the study's `report.json`, which is not in this repository | <1 min |
| `python eval/score_planner.py` | the §5.7 blind scoring, `planner_scores.json` | <1 min |
| `python eval/planner_paired_tests.py` | §5.7's exact McNemar tests over the paired scenes, `planner_paired_tests.json` | <1 min |
| `python scripts/make_figures.py` | every figure | ~1 min |

Four commands have a trap or an option worth knowing, and they are the ones
a reader is most likely to run wrongly.

`keyframe_propagation.py` rewrites its whole output file, so the sweep must
include the coarse settings: the 89× compression figure §4.12 and §7.2 rely
on comes from τ = 45, and a narrower sweep removes it silently.
`score_planner.py` and `run_planner_llm.py` both take `--replies`, which is
how the second planner was run and scored without touching the first, and
`--sample N` prints plans beside their verdicts for manual checking.
`run_vlm_pilot.py` needs `--max-output-tokens` above the 8192 default for a
reasoning model, which spends most of its budget deliberating, and a
truncated reply is indistinguishable from a malformed one.
`python scripts/vlm_manual_check.py` writes a pack that lets §4.13 be
redone by
hand in a browser, showing the image the model saw, the prompt verbatim and
the human, pipeline and model answers side by side, so the scoring script
can be checked rather than trusted.

The independent validation study is scored by
`python analysis/score_votes.py votes.csv`, which lives with that study's
own repository rather than this one; E.3 records the instrument and §4.15
the result.

The two GPU extras: `python scripts/run_sgdet.py --threshold 0.25` for the
deployment-mode pass (~47 min) and `python scripts/run_annotator.py
--config configs/depth_base.yaml --out outputs_base` for A8's Base-model
pass (~7 min).

**4. The benchmark (Chapter 6)** runs on Kaggle and not locally,
since REACT++ training needs more than 6 GB; the README gives the upload and
notebook order. Two framework traps cost real runs and are documented in the
notebooks themselves, because either one silently produces a plausible wrong
number: class index 0 is reserved, patched idempotently in a cell, and
`--eval-only` is a no-op with this configuration, so evaluation calls
`inference()` directly rather than trusting the flag.

**5. The validation study** lives in its own public repository
(`audit-game`), which regenerates its claim set from this repository's
caches and scores the exported votes. The private answer key never enters
either repository, so the study can stay open while its answers stay
closed.

Every reported number traces to one of the JSON/markdown artefacts these
commands write; no figure or table in this dissertation is produced by hand.

### The four constraints that shaped the design

Section 1.3 names them; this is what each ruled out. All perception runs on
a **single 6 GB consumer GPU**, which excludes the largest segmentation and
depth checkpoints and makes Chapter 3's small-model choices obligatory
rather than preferences; ablation A8 asks what that costs and finds almost
nothing on the predicate it was expected to hurt. There was **no budget for
paid annotation**, so the independent re-estimate of precision is an unpaid
volunteer study (E.3); it closed at 20 raters with a control arm of
human-written claims, so §4.15 can report what the tool scores against what
the annotators score on the same instrument, while the audits around it
remain the author's own, with the circularity §2.9 states as an objection
before any result is reported. The project uses **one dataset**, the one
whose bottleneck the work exists to address, and the price is that
generalisation rests on argument and not on a second domain. And the
benchmark runs use **free hosted GPU sessions**, which caps the affordable
seeds and rules out a hyper-parameter search: Chapter 6's three-seed
replication is what that budget allows, and its width is reported rather
than smoothed over.

### What each limitation would take to settle

Section 9.3 states each limitation with the experiment that would settle it;
these are the specifications.

**Execution, not just planning.** The planner experiment closes one of the
two remaining links but no robot moved. The same five conditions on a
physical Spot, or in a simulator with contact physics, would close the last
one, and it is the only link left between labels and behaviour.

**An independent verdict on the shipped support rule.** The volunteer study
closed at 1,415 usable judgements from 20 raters and agrees with the
author's blind audit on support to 0.009, with a control arm establishing
that the same raters score human annotation at 0.940 on the same instrument.
Its claims were drawn on 17 July, a month before `on_contact_min` was
re-fitted from 0.60 to 0.85, so every `on` and `under` claim in it comes
from the superseded rule. Re-running that arm on post-refit labels is the
cheapest outstanding item in the list: the instrument exists, the rendering
pipeline exists, and only a fresh draw and a fresh round of volunteers is
needed.

**Ten seeds per arm, and the notebook that would run them.** Three seeds
bound the paired benchmark difference to [-0.070, +0.069], so Chapter 6
reports parity without establishing it; ten per arm would tighten that to
about ±0.020. `scripts/kaggle/unrun_seed_power_10x.ipynb` implements it,
pinning the framework commit §7.4 names as uncontrolled and ordering runs
seed-major so an interrupted session leaves the arms balanced. It needs
about 45 GPU-hours against a 30-hour weekly allowance, and ships unrun.

**A labelled cross-domain sample.** What transfers is the calibration
procedure, not the fitted constants. A few dozen labelled images from a
second domain would turn E.4's qualitative evidence into a measurement, and
it is cheap enough that a replication should simply include one.

**Stereo, or a calibrated RGB-D capture.** The supplied folder is named
`rightimg`, implying a left counterpart held by the supervising group. True
stereo would attack the front/behind bound directly, supplying disparity at
every frame from a known baseline, which is what the multi-frame estimators
of A9 lack: those must recover the camera's motion first, and on small
low-texture objects they answer for only 9% of pairs and are 0.20 less
accurate where they do (§4.9, Appendix D.6). A calibrated pair removes both problems,
and it keeps the method's premise intact, since stereo is available at
capture time whereas depth recovered from a robot walking twenty frames is
not available to a single-image annotator.

**Ground truth for the unlabelled portion.** The pipeline was run over the
1,766 frames nobody has annotated (E.6). Capacity and stability on
unfamiliar input are therefore measured; correctness on that portion is not,
and cannot be without labels. A few hundred labelled triplets from those
frames, an afternoon of annotation, would close it.

## Appendix C: Predicate specification

This is the complete geometric specification of the seven predicates: the
per-object measurements every rule reads, the rule for each predicate with its
thresholds and the evidence behind them, the correction step, and the flagging
policy. It is the operational definition §4.7 establishes the dataset never
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
   containment fraction instead of an IoU, so a small object sitting fully
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
requires `contact >= on_contact_min` (0.85; first calibrated at 0.60 on the
training groups and re-fitted after the blind audit of §4.14 measured what
the lower value cost in precision) together with the depth gate and the
centroid order. This recovers the containment case the box test misses, nested
boxes at shallow viewing angles, formerly 79 to 88% of support misses, and
rejects cluster neighbours whose boxes touch but whose masks do not. Measured:
held-out support F1 0.71 to 0.87, and re-audited extra-label precision 0.07 to
0.73 for `on` and 0.20 to 0.80 for `under`.

**Class guard.** Classes listed in `no_support_classes` (shipped value:
`human`) are excluded from `on` and `under` in either role. The justification
is measured and not assumed: the annotators never recorded a person as
supporting or being supported, on 0 of 2,466 gold support triplets, and mask
contact alone cannot distinguish a person *holding* an object from a surface
*supporting* one. The guard is a configuration entry rather than a hard-coded name, so a
deployment with different classes revises it in one line, and it does not
cover one object held by another unguarded object: a manipulator, a trolley
or an animal would defeat it. That is a fair objection to a class list
standing in for geometry, and ablation A10 (Appendix D.8) tests the obvious
geometric replacement rather than conceding the point in the abstract. It
does not work, and the measurement says why.

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
left_of(A, B)  ==  (cx_B - cx_A) > lateral_center_eps
                   in camera-frame image coordinates, cx normalised by width
```

As with depth, the band is inside the rule: `lateral_center_eps` is 0.02, so a
pair whose centres are closer than that is not labelled either way. The
magnitude `|cx_A - cx_B|` gives the confidence, and pairs below the band are
flagged ambiguous and not silently labelled.

### C.5 `right of(A, B)`: A's centre is right of B's

```
right_of(A, B)  ==  (cx_A - cx_B) > lateral_center_eps
```

The strict mirror of `left of`. Exactly one of the two holds outside the
ambiguity band; inside it neither is emitted and the pair is flagged.
`right_of(A,B) == left_of(B,A)`.

### C.6 `in front of(A, B)` and `behind(A, B)`

A two-stage cascade. Stage one is depth ordering:

```
in_front_of(A, B)  ==  (d_B - d_A) >  depth_eps     (smaller depth = nearer)
behind(A, B)       ==  (d_A - d_B) >  depth_eps
otherwise            stage one abstains and defers to stage two
```

The band is part of the test and not a separate check: `depth_eps` is 0.03,
so a pair whose depths differ by less than that is not ordered by stage one
at all. This is `src/predicates.py` verbatim, where the two rules are
`(b.depth - a.depth) > depth_eps` and its mirror.

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
near(A, B)  ==  box_gap_rel(A, B) <= near_T
                AND not (on(A, B) or under(A, B))
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
ever used `near` (group_0: 461 labels, group_4: 160, group_8: 93, summing to
714 of the 717 in Table 4.1; group_2 supplies the remaining 3 and the other
five groups none). `near_T` is therefore fitted on the near-using groups inside
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

### C.9 Confidence flags, and what they cost

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
| `on` | mask contact below, depth co-location, centroid order; box test is the no-mask fallback | `on_contact_min` 0.85, `on_depth_eps` 0.06, `on_vertical_gap` 0.05, `on_horizontal_overlap` 0.20 | `on(A,B) = under(B,A)` |
| `under` | inverse of `on` | as `on` | `under(A,B) = on(B,A)` |
| `left of` | `cx_A < cx_B` | `lateral_center_eps` 0.02 | `left(A,B) = right(B,A)` |
| `right of` | `cx_A > cx_B` | `lateral_center_eps` 0.02 | `right(A,B) = left(B,A)` |
| `in front of` | `d_A < d_B`, then the guarded ground-plane fallback | `depth_eps` 0.03, `plane_band` 0.005 | `front(A,B) = behind(B,A)` |
| `behind` | `d_A > d_B`, then the guarded ground-plane fallback | `depth_eps` 0.03, `plane_band` 0.005 | `behind(A,B) = front(B,A)` |
| `near` | `box_gap_rel <= near_T`, never on a contact pair | `near_T` 1.372 (fitted), `flag_near_band` 0.15 | symmetric |

Every threshold above is declared in `configs/default.yaml`, so no constant
that affects a label is buried in a function. `near_T` is fitted by
`eval/fit_near.py` under the protocol of C.7 and frozen there.

### C.11 Design detail behind Chapter 3

Section 3.4 gives the pipeline table and §3.6 the correction policy; these
are the parts that did not need to sit in the chapter.

**Normalisation and orientation.** Coordinates are normalised by image size
so thresholds transfer across resolutions, and depth is inverted to "smaller
is nearer" and min-max normalised per image. The sign was fixed after
front/behind agreement rose from ~26% to ~74%, which is the kind of error
that looks like a modelling failure and is not. An EXIF-aware loader makes
the 180-degree-rotated captures upright before any box, mask or depth is
read; the dataset stores that rotation behind a flag, and reading it wrongly
produced correct-looking unit tests over an upside-down image (§9.4).

**Why support is demoted rather than resolved.** `on` and `under` are
independent tests over *different* contact evidence, the mask-contact
fraction measured each way round, so noise in either can make both fire on
one pair; that case becomes an `on_under_conflict` flag and neither label is
emitted. Picking the stronger of two contradictory signals would produce a
label the evidence does not support while looking exactly like one it does,
and an annotator that fabricates under uncertainty cannot be audited. The
alternative of emitting everything and letting the consumer sort it out was
rejected because RQ2's consumer is a model, which has no way to sort it out.

**The class-aware guard.** Support is not evaluated when either object is a
person: the annotators never recorded one, on **0 of 2,466 gold support
triplets**, and mask contact cannot distinguish an object *resting on*
someone from one being *held* by them. A rule that cannot represent the
distinction its evidence turns on should decline the pair and not guess,
and the guard is a configuration entry (`no_support_classes`) rather than a
special case buried in code. It is still a class list standing in for
geometry and would not cover a manipulator or an animal holding something;
ablation A10 tests whether contact height can replace it and finds it
cannot, at a cost of half the support recall (D.8).

**The `near` fitting protocol.** The naive protocol, fitting one threshold to
all `near` labels and testing on held-out images, fails informatively: fitting on
groups 0–5 and testing on 6–8 yields held-out F1 = 0.009, because the label
was applied by only three of nine groups with very different exhaustiveness.
The protocol therefore fits on human-*annotated*, non-contact pairs only,
since unannotated pairs are not reliable negatives under sparse annotation
and contact pairs are never `near` by the measured convention; it uses only
the training-split groups that used the label at all (0 and 4); and it
reports agreement on the held-out near-user, group 8, which contributed
nothing to the fit. Per-annotator precision at the fitted T is 0.41 / 0.63 /
0.16, so what varies across annotators by about fourfold is how exhaustively
each applied the label, not where they placed it.

## Appendix D: Ablation derivations

Chapter 4 (§4.9) summarises the ten ablations and their verdicts. This
appendix gives the derivations: how each shipped parameter was calibrated on
the training annotator groups, what the held-out groups then reported, and
what the audits of the affected predictions found. The two declined
refinements are included in full, because a negative result is only useful
if the reader can see what was actually tried.

### D.0 The ten ablations at a glance

Section 4.9 gives the verdicts in prose; this is the register they
summarise. Every parameter was selected on the training annotator groups
alone, with the held-out column reported and never optimised against.

| # | What it tests | Setting | Verdict |
|---|---|---|---|
| A1 | support depth co-location gate | `on_depth_eps` 0.06 | **shipped**; held-out support F1 0.58 → 0.71 |
| A2 | front/behind abstention band | `depth_eps` 0.03 | **shipped**; bounds the trade (recall 0.71 at ε=0 for 0.26–0.36 precision) |
| A3 | lateral abstention band | `lateral_center_eps` 0.02 | **shipped**; recall flat to 0.02 while precision rises |
| A4 | proximity threshold | `near_T` 1.372 | **shipped**; the knee of the recall plateau, held-out recall 1.00 |
| A5 | mask-contact support rule | `on_contact_min` 0.60, re-fitted to 0.85 (§4.14) | **shipped**; held-out support F1 0.71 → 0.87, both error directions at once |
| A6 | `near` contact exclusion | on | **shipped**; costs 2 recalled triplets, prevents 4,084 labels contradicting the measured convention |
| A7 | ground-plane depth fallback | `plane_band` 0.005 | **shipped**; front/behind 0.52/0.55 → 0.70/0.71, mean recall 0.79 → 0.85 |
| A8 | larger depth model (Base, 4× parameters) | n/a | **declined**; +0.001/−0.002 front/behind, mean recall marginally lower |
| A9 | multi-frame depth (two-view triangulation) | n/a | **declined**; 0.706 against the monocular cascade's 0.902, on 9% of pairs |
| A10 | geometric drop fraction in place of the class guard | n/a | **declined**; the resting and held populations overlap, no threshold separates them (D.8) |

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
(`on_contact_min` = 0.60 here; the train-F1 plateau is flat from 0.60–0.80,
so the choice looked uncritical, and §4.14 later shows why that plateau was
the wrong thing to read), ablation A5: support F1 on held-out annotators
rises again, 0.71 → **0.87**, with `on` recall 0.82 → 0.88 and restricted
precision 0.73 → 0.88 simultaneously, the rare change that improves both
error directions at once, exactly as the failure gallery and audit
predicted. Knock-on: `near` recall reaches **0.997 pooled (715/717; the two
residual misses are contact-boundary cases, §4.10) and 1.00 held-out** as
the last contact-boundary suppressions disappear; headline mean recall 0.79
→ **0.81**. The A4 sweep confirms that `near_T`, the proximity threshold and not
the contact one under discussion here, sits exactly at its recall
plateau's knee: recall is flat from T = 1.372 upward while emissions
keep growing, so the fitted value is the least-permissive point achieving
maximal agreement. A 30-sample re-audit of the new support extras appeared
to confirm the precision claim: extras correct rise from 1/15 and 3/15 (box
rule) to **11/15 and 12/15**, pooled 0.13 → 0.77. That re-audit was
unblinded, and §4.14 shows the figure does not survive blinding; the
threshold below was fitted on train F1 against gold covering a tenth of
ordered pairs, so a false positive outside the gold cost the fit nothing,
and the plateau called uncritical here is flat for that reason. A second
independent signal for any future refit is the supporting object's size:
`on(A, B)` requires B to be able to hold A up, and a 20-pixel cube is not a
surface. The seven
remaining wrong/uncertain extras have structure: a person *holding* a remote
fires contact (holding ≠ resting), one occluded bottle-behind-bottle pair,
and three distant clusters too small to verdict confidently. The
person-holding mode is closed by a **class-aware guard**: annotators never
label person-support (0 of 2,466 gold support triplets involve a person on
either side), so `on`/`under` are simply not evaluated for that class,
removing ~130 false emissions at no recall cost, since the person side
carried no gold to recover.

### D.3 The ground-plane fallback (ablation A7)

The depth abstention band was the single largest miss cause, and most of it
turned out to be resolvable without depth at all. Appendix C.6 specifies the
rule and its two guards; what follows is the evidence that selected it. On the
train groups the fallback adds 386 committed directions at 0.91 agreement; on
held-out group 7 it adds 54 and **every one agrees with the annotator**.
Effect on the headline: front/behind recall 0.52/0.55 → **0.70/0.71** (aligned
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
+0.001/−0.002: the remaining depth ambiguity lives in the scenes, not in
model capacity, and the Small variant's Apache licence is kept. Both null
results bound where further engineering can and cannot help.

### D.5 Why a geometric cue, not a bigger depth model (ablation A8)

It is worth asking whether the depth pair would improve simply by using a
stronger depth network. It does not: swapping Depth Anything v2 Small for
the 4× larger Base variant and re-running the whole dataset moves
front/behind recall by +0.001 and −0.002, from 0.696/0.711 to 0.697/0.709,
while mean recall *falls* from 0.845 to 0.843, and the front/behind emit
rate moves by +0.001. Deltas are differences between the rounded figures
printed here, so the columns add up as read. Both arms are end-to-end runs
on the shipped rule set, so the comparison isolates the depth model alone.
An earlier version of this appendix reported the same ablation with both
arms at the pre-refit support threshold and predicted that re-running it on
the shipped labels would move the two arms together without changing a
difference that small; it did. The depth-predicate limit is *monocular
ambiguity* (two objects at a similar camera distance are inseparable by any
monocular model, regardless of its fidelity), not the network's quality.
This is precisely why the fallback that worked is a geometric projection cue
rather than a heavier perception model, and it justifies shipping the Small
variant: identical accuracy, an Apache-2.0 licence, and half the VRAM.


### D.6 Why not multi-frame depth either (ablation A9)

Ablation A8 rules out a bigger *monocular* model, but it leaves the sharper
objection open. The images are consecutive frames of a robot walk (§4.12), so
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
length is **assumed and not calibrated**, and that assumption is a real
limitation of this ablation rather than a tolerable simplification. A wrong
intrinsic matrix does not merely scale the reconstruction: the essential
matrix decomposed from it is wrong too, so the triangulated points sit in a
projectively distorted space in which depth order is not guaranteed to
survive, and the distortion is worst away from the image centre and under the
forward translation a walking robot supplies. The result below therefore
bounds *uncalibrated* two-view triangulation on this capture, and should not
be read as a measurement of what multi-view geometry can do here.
Whether that limitation decides the result is testable rather than arguable,
and `--focal-sweep` tests it. Re-running the 20-frame baseline across a
fourfold range of assumed focal lengths, 0.5 to 2.0 image widths, moves
ordering accuracy between 0.678 and 0.739 while the monocular cascade on the
same pairs holds at 0.888 to 0.900. Two things follow. The shipped
assumption of 0.9 widths is the worst of the six tested, so the objection
that the intrinsic was picked instead of measured is correct, and moving to
the best of them buys 0.061. And that is not enough to matter: at its
narrowest the gap to the monocular cascade is 0.148, and no focal assumption
in the range closes it. The projective distortion is real, and it is not
what decides this comparison.

This is the correct construction and it does much better than the first, but
not well enough:

| Depth cue | Ordering accuracy | Pairs answered |
|---|---|---|
| Two-view triangulation (10-frame baseline) | 0.706 | 337 of 3,597 |
| Monocular cascade, same pairs | 0.902 | 337 of 3,597 |

Both numbers fall on the same pairs, so the comparison is like for like. The
multi-frame estimate is 0.20 worse where it applies, and the two disagree on
27% of pairs, with the monocular cascade right more often in that
disagreement. Coverage is the harsher problem: triangulation returns an
answer for 9% of the depth-labelled pairs, because it needs several trackable
corners inside an object's box and the dataset's objects are small,
low-texture cubes and boxes photographed at 640×480 in greyscale. Widening
the baseline trades coverage for geometry and makes accuracy worse, not
better: at 40 frames it falls to 0.562 on 258 pairs.

Splitting those pairs by how far apart the two triangulated depths actually
are says more than the aggregate does, and it is the check that separates a
weak estimator from a broken one. The split is computed by the ablation
script itself, so the four rows sum to the 337 pairs above:

| Relative depth separation | Pairs | Ordering accuracy |
|---|---|---|
| 0.000–0.013 | 84 | 0.440 |
| 0.014–0.044 | 84 | 0.774 |
| 0.044–0.117 | 84 | 0.893 |
| 0.117 and above | 85 | 0.718 |

The estimator is real. Given objects at moderately different depths it
reaches 0.893, within a point of the monocular cascade's 0.902 over all 337
pairs, so the implementation is not the thing holding it back. What it cannot
do is the part that matters: on the quartile of pairs whose depths are nearly
equal it performs at chance, 0.440. Those are precisely the pairs the
monocular cascade abstains on, and the reason is the same for both methods.
Two objects at genuinely similar camera distance are not separated by
measuring depth more carefully, because the quantity being measured is
almost the same for each of them. Multi-view geometry inherits that limit
rather than removing it. The fall to 0.718 in the top quartile is the
opposite failure and a real weakness of two-view triangulation: a handful of
badly conditioned points produce depths that are wrong *and* far apart, so
the largest separations include the worst outliers.

The scope of this result is that it rules out the cheap version of the
idea rather than the idea itself. A careful multi-view reconstruction over
many frames, with bundle adjustment and real intrinsics, would estimate depth
better than two views and an assumed focal length. What the ablation
establishes is that geometric depth is not free here, that the obvious
implementations lose to the monocular cascade on both accuracy and coverage,
and, read with §4.12, that the return on any of them is bounded: a predicate
that already reproduces its own verdict 0.958 of the time across viewpoints
does not have much room to gain from measuring depth more precisely. There is
also a design cost that no accuracy figure captures. Every multi-frame method
requires neighbouring frames at inference time, which the single-image
annotator this project set out to build does not have.

### D.7 Failure diagnosis by predicate

Section 4.3 and §4.10 state what the diagnosis establishes. Two tables reach
it from opposite sides. The first records, for each human triplet the tool
failed to recover, the labels it *did* emit on that pair, which identifies
the failure mode directly:

| Gold predicate | Missed | Most frequent co-emissions on missed pairs |
|---|---|---|
| on | 275/1465 | near (275), behind (248): support demoted to proximity/depth |
| under | 253/1001 | in front of (234), near (232) |
| to the left of | 34/972 | near (33), to the right of (24): flips at the centre band |
| to the right of | 18/1174 | to the left of (16) |
| in front of | 611/2013 | near (548), behind (460), to the left of (256) |
| behind | 457/1584 | near (419), in front of (368) |
| near | 2/717 | on (2): the contact-exclusion boundary |

Two structural signatures stand out. Missed front/behind pairs carrying the
*opposite* direction (460 + 368) are almost entirely the convention-inverted
groups of §4.5, and their count *rose* with the ground-plane fallback,
because pairs the tool used to abstain on are now committed in the direction
those groups invert. And the two remaining missed `near` pairs carry
`on`/`under`: the two rules disputing the contact boundary, the same
support-rule frontier the audit (§4.4) identifies from the precision side.

The second re-checks each rule's individual conditions against the cached
geometry and mask-contact maps, attributing every miss to a cause.

| Predicate | Dominant causes (share of that predicate's misses) |
|---|---|
| in front of | abstained in ambiguity band 54% · convention-inverted annotators 45% · **genuine depth error 1%** |
| behind | convention-inverted 51% · abstained 43% · **genuine depth error 6%** |
| on | mask contact below threshold 80% · depth-gate suppressed 18% · occlusion or centroid error 2% |
| under | contact below threshold 70% · depth-gate suppressed 18% · no contact measured (occlusion) 12% |
| near | 2 remaining misses (contact boundary) |
| to the left/right of | centre flip 71–89% · abstained 11–29% (52 cases total) |

### D.8 Can geometry replace the class guard? (ablation A10)

A class list is a semantic patch on a geometric rule, so the natural
question is whether the distinction it encodes has a geometric signature. It
should: an object *resting* on something meets it at the supporter's top
edge, while an object *held* meets it partway down the holder's body. That
is measurable without any class name. For every pair passing the contact and
depth gates, `eval/support_guard_ablation.py` computes the drop fraction,
the height at which the subject's bottom edge falls inside the object's
vertical extent, where 0 means the subject sits on the object's top surface.

The two populations overlap too much to separate. Across 836 images, the
1,190 gold-confirmed resting pairs above the contact threshold have a median
drop of 0.19 and a 10th-to-90th percentile range of -0.04 to 0.50; the 51
pairs the guard blocks have a median of 0.38 and a range of 0.33 to 0.41,
which sits inside the resting distribution and not beside it.

| drop threshold | gold resting pairs kept | person pairs blocked |
|---|---|---|
| ≤ 0.10 | 31.5% | 100% |
| ≤ 0.20 | 52.8% | 100% |
| ≤ 0.30 | 72.5% | 92.2% |
| ≤ 0.40 | 83.4% | 13.7% |
| ≤ 0.50 | 89.9% | 2.0% |

There is no threshold that does both jobs. Excluding every blocked pair
costs about half the genuine support recall; keeping most of the support
recall lets almost every held object back in. That is a negative result, and
a useful one, because the objection assumes a geometric solution exists and
is merely being skipped, and on this data it does not exist at this level of
geometry. Contact height cannot tell a hand from a shelf because a hand at
waist height and a shelf at waist height are the same measurement.
Separating them needs something the pipeline does not have, either surface
normals from real 3D or an affordance notion of what can support, and both
are the future work of §9.3 rather than a threshold.

What the ablation does settle is the guard's blast radius. Fifty-one pairs
in 836 images reach the contact threshold with a person on either side, so
the class list changes 51 decisions out of the 42,440 the tool makes. It is
a narrow patch over a real gap, and the gap is a limit of monocular geometry
rather than of the rule set.

## Appendix E: Extended validation studies

Six studies sit here. Three support a result reported in a chapter and
carry the detail behind it: the vision-language baseline of §4.13, whose
diagnostics make its failure interpretable rather than merely worse (E.1);
the viewpoint-stability measurement of §4.12, whose segmentation evidence and
coverage limits qualify how far the stability figures reach (E.2); and the
planner experiment of §5.7, whose relation filter and blind scorer are what
the comparison rests on (E.5).

The other three are reported here in full, because their instruments need
more room than a chapter allows. The independent precision study needs its
design read independently of what it found (E.3); the video processing is
qualitative, over two clips with no ground truth (E.4); and the scale run
has no labels to be correct against (E.6). Each is work done and is recorded as such, with what it does and does
not establish stated in its own section.

### E.1 The vision-language baseline: diagnostics and limits

The larger model redistributes the picture rather than resolving it. It
asserts more (414 judged-pair assertions against 344), lifting recall and
costing precision, so its F1 lands within 0.008 of the smaller model's, and
where the smaller model looked most interesting, `behind`, the larger is
markedly worse at 0.281 precision against 0.478. It is closest to the
pipeline on support, 0.74 against 0.91, and `behind` recall actually falls
with scale, 0.169 to 0.138.

Section 4.13 reports the headline comparison over two models: they recover
0.40 and 0.45 of the human triplets against the pipeline's 0.83, and are the
more precise labellers on the judged pairs, 0.42 and 0.39 against 0.35,
while losing F1 on every predicate.

**The setting.** Thirty images, stratified across all nine annotator groups,
go to the model with the ground-truth boxes drawn on and numbered, and it
answers by index (`scripts/run_vlm_pilot.py`). That is the PredCls setting
the pipeline is evaluated in, so neither is scored on detection, and the
prompt carries Chapter 3's definitions verbatim, without which the run would
measure §2.5's reference-frame ambiguity rather than accuracy. Two models
were run, because the objection to one is that a larger model would close
the gap: `gemini-flash-latest` is small and non-reasoning,
`gemini-3.1-pro-preview` a reasoning model an order of magnitude larger.
Scaling moves mean recall from 0.400 to 0.445 against the pipeline's 0.834,
and on the depth pair 0.24 against 0.65 puts the model below the geometric
method's known weak point: scaling the model does not scale the ability
being measured.

**Why recall alone would be unfair.** It rewards whoever asserts more, and
the pipeline makes 885 assertions on the 374 judged pairs against 344 and
414. Restricted to those pairs, where precision is defined, the column
reverses and both models are the more precise, 0.419 and 0.389 against
0.347; they buy it with silence, at a price steep enough that both lose F1
on every predicate, 0.397 and 0.405 against 0.485 pooled. On the pairs it
did judge the smaller model's recall is **0.686** and not 0.378, and on
the predicates where the tool's advantage looks largest the gap closes
entirely: `to the left of` 0.909 against 0.918, and `on` 0.864 against
0.860, where the model is marginally the better of the two. A headline
recall of 0.40 therefore measures two things at once, how often the model is
wrong and how often it declines, and only the first is a claim about spatial
competence.

The per-predicate recall underneath that verdict is this:

| Predicate | Gold | Flash | Pro | Pipeline |
|---|---|---|---|---|
| on | 57 | 0.667 | 0.737 | 0.860 |
| under | 47 | 0.660 | 0.766 | 0.809 |
| to the left of | 49 | 0.408 | 0.469 | 0.918 |
| to the right of | 49 | 0.327 | 0.388 | 0.939 |
| in front of | 80 | 0.188 | 0.237 | 0.650 |
| behind | 65 | 0.169 | 0.138 | 0.662 |
| near | 34 | 0.382 | 0.382 | 1.000 |
| **Mean** | **381** | **0.400** | **0.445** | **0.834** |

The per-predicate form of the precision comparison is below, and the
diagnostics after it are the smaller model's unless stated, since it is the
one the manual-check pack was built for.

| Predicate | Flash P | Pro P | Pipeline P | Flash F1 | Pro F1 | Pipeline F1 |
|---|---|---|---|---|---|---|
| on | 0.950 | 0.840 | **0.961** | 0.784 | 0.785 | **0.907** |
| under | 0.886 | 0.837 | **0.974** | 0.756 | 0.800 | **0.884** |
| to the left of | 0.408 | **0.489** | 0.385 | 0.408 | 0.479 | **0.542** |
| to the right of | 0.381 | **0.396** | 0.387 | 0.352 | 0.392 | **0.548** |
| in front of | **0.484** | 0.475 | 0.364 | 0.270 | 0.317 | **0.466** |
| behind | **0.478** | 0.281 | 0.309 | 0.250 | 0.186 | **0.422** |
| near | 0.105 | 0.084 | **0.123** | 0.165 | 0.138 | **0.219** |
| **micro** | **0.419** | 0.389 | 0.347 | 0.397 | 0.405 | **0.485** |


The pipeline makes 885 assertions on the 374 judged pairs against the models'
344 and 414, which is why recall alone would be an unfair verdict and why
§4.13 reports the restricted comparison beside it. The models' precision
advantage is pooled rather than uniform: it comes from the four predicates
where the pipeline asserts most freely, and on `on`, `under` and `near` the
pipeline is the more precise of the three. Its F1 is nonetheless the highest
in every row, which is the shape of the trade the whole comparison is about.

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

Section 4.13 draws the conclusion these diagnostics support. One point
belongs here and not there, because it is a property of the pipeline
rather than of the model: what the pipeline has over it is not fluency and
not per-assertion agreement, but exhaustiveness and the guaranteed
anti-symmetry of §3.6.

Three limits on this result, one of them now partly settled. The pilot is
thirty images at one prompt, and model capacity was the obvious confound: a
larger model might simply have closed the gap. Running the identical battery
on a reasoning model an order of magnitude larger tests that directly, and
the answer is that capacity moves the numbers without moving the verdict.
Mean recall rises from 0.400 to 0.445 against the pipeline's 0.834, pooled
F1 from 0.397 to 0.405 against 0.485, and the larger model buys its extra
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
the argument §4.13 makes. The smaller model supplied *to the left of*
without its inverse on 0.35 of its assertions; the larger one on 0.16.
Deliberation buys internal consistency. It does not buy enough: 0.16 is
still a sixth of a symmetric relation asserted in one direction only, a
defect the geometric rules cannot exhibit at all because §3.6 enforces the
inverse by construction.

### E.2 Viewpoint stability: segmentation evidence and coverage

Object indices are not stable across frames, the annotators having recorded
different subsets of the scene from frame to frame: `group_0` alone contains
43 distinct object orderings, which is why matching is by class and box
overlap instead of by index. Compression over the 802 pair-bearing frames
(§4.1) is lower than the 2.7× below because those frames are a subset, so
consecutive members sit further apart in the original capture.

*What the released images are.* Pixel-matching them against the 2,650-frame
raw capture the supervising group later supplied identifies them exactly:
they are frames 000000–000883 of one continuous walk, and each annotator
group is a contiguous 100-frame block (`group_0` = frames 0–99, and so on).
That has one consequence for §4.1's split, which §4.12 states: a group is
simultaneously an annotator identity *and* a temporal block holding one
arrangement, so the split is held out by scene as well as by annotator. The
annotator reading survives, since an inverted front/behind convention (§4.5)
and a `near` label used by three groups in nine (§3.2) are labelling
behaviours no arrangement of furniture can produce, and the confound runs
favourably: 0.74 on held-out groups is generalisation to an unseen annotator
*and* an unseen arrangement.

*The per-predicate result.* Consecutive frames show a scene from different
viewpoints, so the pipeline's verdicts can be checked against themselves
with no human labels: a relation fixed by geometry should survive the camera
moving, and one decided by a coin toss at a threshold should not. Frames
were segmented by content drift (§3.10) and each segment's predicates
propagated from its keyframe to the rest
(`eval/keyframe_propagation.py`). At τ = 10 over the 802
pair-bearing frames (568 keyframes, 234 propagated frames, 11,352 comparable
object pairs):

| Predicate | Stability | Recall (propagated) | Recall (per frame) |
|---|---|---|---|
| on | 0.878 | 0.798 | 0.798 |
| under | 0.878 | 0.760 | 0.740 |
| to the left of | 0.989 | 0.959 | 0.966 |
| to the right of | 0.989 | 0.989 | 0.989 |
| in front of | 0.958 | 0.682 | 0.648 |
| behind | 0.958 | 0.636 | 0.659 |
| near | 0.972 | 1.000 | 1.000 |
| **Mean** | **0.946** | **0.832** | **0.829** |

Propagated recall's small advantage over per-frame (0.832 against 0.829) is
likely an artefact of the selection rule, the segment representative being
the frame nearest the segment mean.

*Segmentation.* Distances are mean absolute differences over 64×48
mean-subtracted greyscale thumbnails, the subtraction discarding the
exposure shifts of an auto-exposing camera, which would otherwise fire
boundaries of their own (§3.10). At τ = 10 the full 2,650-frame sequence
collapses to 892 segments, a 3.0× reduction; over the 884 released frames it
gives 331, 2.7×.
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
and not with per-frame overlap, is the straightforward extension that
would recover it.

### E.3 The independent validation study: design and scoring

The true-precision estimates of §4.4 and §4.9 carry one weakness no amount
of sampling fixes: they were verdicted by the author of the tool being
evaluated. Conservative rules and published evidence mitigate that without
removing it, and the accurate description is "author-verdicted". This study
re-estimates precision with disinterested judges, and pairs every estimate
with the same judges' verdict on the human annotations so that it has
something to be read against. It is reported in §4.15 and scored below; the
design is given first, so that a reader can judge the instrument
independently of what it found.

It measures four things the author-verdicted audits cannot: crowd precision
per predicate on 412 tool claims against §4.4's fifteen per predicate, some
twenty-seven times the sample; the same raters' precision on claims the *annotators* wrote, which
is what turns the first number from a bare score into a comparison; an
author-bias check against the author's own verdict on the 147 items carrying
both; and whether disputed claims are wrong or merely ambiguous, through
inter-rater reliability, which is the distinction the disagreement
literature of §2.3 insists on.

**Sampling.** Two arms. The *treatment* arm is drawn from the tool's *extra* predictions: ordered pairs the human annotators never labelled, which is exactly the population with no ground truth to score against. The *control* arm is drawn from the human annotations themselves, 84 per predicate, rendered through the identical pipeline and interleaved by the same shuffle, so nothing on the page distinguishes them and a rater cannot tell which arm a claim belongs to; the arm is recorded only in the private key. It began as 2,002 treatment claims, 286 per predicate, and was resized on 24 August to 412 treatment and 588 control: spreading raters across 2,002 items meant most would never be judged, and the 412 retained were exactly those already carrying a vote, so no judgement collected was discarded and no claim already voted on changed its identity.

That resize needs stating precisely, because the retained 412 are not a fresh
random draw from the 2,002: they are the claims the collection process
happened to reach, and items were served to raters in sequential batches, so
retention is correlated with serving order rather than independent of it. Two
properties bound what that can do. The batches are predicate-mixed rather than
predicate-blocked, since every early batch contains all seven predicates, and
retention is even across predicates to within sampling error: 412 of 2,002
overall, 20.6%, with per-predicate rates from 17.1% to 23.1%, which a
chi-square test cannot distinguish from uniform (χ² = 5.20 on 6 df, p = 0.52).
So the mechanism demonstrably does not skew the sample by predicate, which is
the axis the per-predicate estimates rest on. It cannot be shown to be
ignorable for anything else that correlates with serving order, and the
accurate description is a sample selected by response rather than a
randomised one. Each claim is rendered as the source photograph with
the subject outlined in red and the object in blue, presented with a single
sentence ("the book is on the box"), and answered TRUE or WRONG / can't tell
by volunteers recruited through an open link. The instructions restate
Chapter 3's operational definitions: camera-frame laterality, "in front of"
as nearer the camera, support as physically resting rather than held, and an
explicit instruction to answer WRONG when unsure, which reproduces the
conservative rule used in the author's own audits. Each browser receives a
random identifier that prevents repeat judgements without identifying
anybody, and faces are anonymised in every image (Chapter 8).

**Coverage.** Stratified by what each analysis requires. An aggregate precision estimate needs only one judgement per claim, the estimate being over claims rather than over raters, so ordinary claims target a single rater. The 147 claims that also carry an author verdict target three raters each, because the crowd-versus-author comparison and the inter-rater reliability figure both need several independent judgements on the *same* item; those claims are served first. Against the resized pool that is about 1,300 judgements rather than the 3,000 a uniform three-rater target would demand. The priority ordering was the design's hedge against a thin turnout, and it worked in the direction intended: all 147 priority claims reached three raters, so the author comparison and the reliability figure rest on the coverage they were specified for and not on whatever the response happened to allow.

**Scoring**, fully specified in advance (`analysis/score_votes.py`): ties
resolve to WRONG, matching the audit protocol; reflex-speed responses and
raters who disagree systematically with everyone else can be excluded by
pre-declared filters. Crowd precision is reported per predicate with binomial
intervals, author agreement as percentage and Cohen's kappa (Cohen, 1960),
and crowd-internal reliability as Krippendorff's alpha.

**What it returned.** Collection closed at 1,415 usable judgements from 20 raters, covering 832 of the 1,000 claims, 83.2%, with no rater supplying more than 15% of the total. All 412 treatment claims and 420 of the 588 control claims carry at least one verdict, and 147 carry three or more, which is the subset the author comparison and the reliability figure need. Of the pre-declared filters, the duplicate rule fired on 12 submissions, all of them from before the 11 August server fix; no response fell below the 800 ms floor, and no rater was excluded as a systematic outlier. Section 4.15 reports the figures; `eval/crowd_validation.py` reproduces them from the scorer's report, separating the arms on the claim id, since the scorer's own per-predicate totals pool both.

**What the arm establishes, in full.** Section 4.15 gives the verdicts; the
reasoning behind each is here.

*The control arm is what makes the treatment arm readable.* Half the pool is
drawn from the human annotations rather than the tool, rendered through the
identical pipeline and interleaved by the same shuffle, so nothing on the
page distinguishes them and a rater cannot tell which arm a claim belongs
to. Without it a low score on the tool would be uninterpretable, since the
same raters under the same conservative instruction might score any claim
low. They do not: on the human-written claims they answer TRUE 0.940 of the
time (395/420) against 0.726 on the tool's (299/412), so the raters are not
uniformly severe and what the tool scores is about the tool.

*Why the denominators differ.* The judges saw different draws from the same
pre-refit generation rather than a common sheet: the volunteers were shown
126 support claims from the study pool and the two blind judges 94 from the
audit pack of §4.14. The three columns are therefore three estimates of one
quantity, not three verdicts on one sheet, which is what makes their
agreement worth reporting at all.

*The ranking agreement.* Across all seven predicates the volunteers rank the
tool almost exactly as the author does, Spearman 0.96 against the model's
0.34, landing 0.074 away on average. `near` shows the same shape from the
other side as support does: 0.923 and 1.000 from the two human judges
against the model's 0.625.

*Against the human baseline.* On the five predicates the support threshold
does not touch, the tool scores 0.864 against the annotators' 0.926, a gap
of 0.063, and on `in front of` it is marginally ahead. On support it scores
0.413 against 0.975 (118/121), a gap of 0.563 whose intervals do not come close to
overlapping, the same weakness §4.14 found and the refit responded to, now
measured against what human annotation scores on the identical instrument.

**Reading the three-judge comparison.** One asymmetry bounds how far §4.15's
comparison can be pushed: the tool's claims are its *extra* predictions, on
pairs the annotators passed over, while the control claims are pairs they
chose to record. Annotators record what is clear, so some of every gap
between the two arms is the difficulty of the claim and not the quality
of the label. For the 0.063 gap on the five untouched predicates that
reservation may account for most of it; for the 0.563 support gap it cannot.
The agreement figures also read as real signal rather than noise: both the
crowd–author agreement (0.871, κ 0.683) and crowd-internal reliability
(α 0.703) roughly doubled as the sample grew, which is what signal does and
noise does not, and neither approaches 1.0, which is §2.3's account of spatial
language predicts that residue better than rater carelessness does, some
disagreement being over what the words mean rather than what the photograph
shows. The comparison also cannot fully separate label quality from claim
difficulty, because the two arms are drawn from populations the annotators
themselves divided.

### E.4 Video processing: settings and the open-vocabulary failures

This records how the clips were processed and what went wrong, both of which
qualify the reading above.

**The clips and the thresholds.** Two royalty-free stock clips, sourced in
Appendix A, share nothing with the robot dataset: different scenes, a
different camera, and objects almost entirely outside the six annotated
classes (monitor, keyboard, mouse, mug, spectacles, plants, lamp, notepad,
laptop, wallet, earbuds case). Nothing was retuned for them, so `near_T`,
the depth band, the contact fraction and the plane band keep the values
fitted on groups 0-5. The two are complementary regimes: a moving camera
over a static desk (clip 1, 99 frames), where any variation is measurement
noise, and a static overhead camera with moving hands (clip 2, 79 frames),
where relations genuinely change and smoothing must not erase them.

**Processing.** Each frame is annotated independently by the deployment-mode
stack of §4.11 with open-vocabulary prompts; object identities are carried
between frames by greedy IoU tracking, and each pair's predicates are
smoothed by a plus-or-minus-two-frame temporal majority vote
(`scripts/run_video.py`; overlays and per-frame records in
`outputs/video/`). The vote is the only component with no counterpart in the
still-image pipeline, and its effect is measured instead of assumed:
{{fig:video-stability}} plots frame-to-frame agreement before and after it,
lifting persistence from 0.763 to 0.774 on the static clip and from 0.928 to
0.964 where the hands make detection churn, which is the behaviour a
majority filter should show.

**What the clips show.** Three things hold. Relations are stable wherever
identity is: for pairs co-visible in at least 20 frames the predicate
persists at 0.77/0.96 mean (`eval/video_stability.py`). The rules transfer
to objects never calibrated on, a pen on a notepad and a
wallet-and-photograph stack labelled by the same mask-contact evidence
fitted on six classes. And the camera-frame semantics behave as designed: in
the bird's-eye clip front/behind re-maps to distance from the viewer's edge
of the desk, which is the reference-frame dependence §2.5 cites from
RoboSpatial, demonstrated rather than asserted.

These figures are lower on clip 1 and higher on clip 2 than the version of
this appendix written before the support threshold was refitted (§4.14),
which reported 0.90 and 0.94. The refit removes support emissions on weak
contact evidence, and on a desk viewed from a moving camera those are
exactly the borderline pairs that persisted across frames by inertia; on the
overhead clip, where the hands make support genuinely intermittent, removing
them raises persistence instead. Both clips were re-processed end to end on
the shipped configuration, so the numbers here describe the tool being
submitted.

**Persistence and agreement.** Persistence is measured over pairs co-visible
in at least 20 frames, where co-visible means both track identities appear
in that frame; 81% and 89% of pair-predicates are present in at least 90% of
their co-visible frames. Frame-to-frame triplet agreement (Jaccard 0.84 and
0.59) is dominated by zero-shot detection churn and, in clip 2, by genuine
hand motion; the dips there align with the hands picking objects up, which
is the change the smoothing is required not to erase.

**Open-vocabulary failures.** Two are plainly visible and worth recording.
Content displayed *on the laptop screen* is detected as real objects
standing in real relations to the objects around it, and items outside the
prompt list snap to the nearest prompted class: an earbuds case is labelled
a `cup`. Neither is corrected, and both are present in the released
overlays.

**What this bounds.** There is no video ground truth, so these are
qualitative judgements over two clips and a labelled cross-domain sample
remains future work (§7.6). What they settle is that transfer is not blocked
by the object vocabulary, because every failure visible in them is a
detection failure rather than a relation failure, which is the attribution
§4.11 makes for the dataset itself.

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

**The failure with no relations at all.** Condition A never clears the
occluder, in twenty-five scenes out of twenty-five and under both planners.
In seven of those plans the planner names the occluding object, but only ever
as something to steer around: "maintaining a clear path above cube0, book2,
book3", or "lift box0 to a safe height to clear book1".

**The six scenes condition C fails on.** On the shipped labels the support
relation the task depends on is present in 18 of the 25 prompts, absent from
scenes 1, 4, 7, 16, 19, 24 and 25 (on the pre-refit labels it was absent
from 4, 16 and 24 alone); scene 7's plan is nonetheless recovered from the
surrounding description, so absence of the relation is nearly but not quite
sufficient for failure. The six failing scenes are 1, 4, 16, 19, 24 and 25
(§5.7), and the failure is the same one six times. In each, the prompt does name the occluder and the target together (scene 4 offered "box6 is in front of book7" and "box6 is near book7", and scene 24 added "to the left of"), but in none of the six does it say the occluder is resting on the target. No plan therefore moves it: the scored `clear_step` is empty in all six, while every plan grasps the target and none invents an object. Scene 16 shows the mechanism sharply. The prompt lists four objects in front of box7, and the plan grasps box7 and lifts it clear of everything nearby, cube3 named among them, without ever taking cube3 off it, because nothing said cube3 was on it. Each of the six reasoned correctly from incomplete input.

**How the three findings were established.** Section 5.7 states them; the
detail is here. *The two planners agree exactly.* Condition C fails on the
same six scenes under both `gemini-flash-latest` and the reasoning model
`gemini-3.1-pro-preview`, which is what makes the result a property of the
prompt and not of the engine reading it, and it disposes of the
objection that a more capable planner would infer support from the object
list: it does not, in twenty-five scenes of twenty-five, twice. *Every
failure is a missing support relation.* In all six C failures the relation
list lacked the load-bearing support relation, and in no scene was the
relation present and ignored; six misses in twenty-five is 24%, against the
support recall of 0.81/0.75 in §4.2, so the planner result is the fidelity
result one level higher in the chain. *The two automatic sources fail on
disjoint scenes.* D scores 20 of 25 and its five failures do not intersect
C's six, so a union supplying more support relations repairs exactly the
failing cases and cannot break the working ones -- E clears the occluder in
all 25, gaining six scenes over C and losing none, and the disjointness
survives the threshold refit that widened C's own gap.

**Five limits on the planner result.** The scenes were selected to contain
an occluder, so the result speaks to that situation and not to task planning
at large. The vision-language model's assertions were never audited as the
tool's were (§4.4), so the union's gain is measured on the planning task
alone. Condition B is handed the exact fact the task tests, so what is being
compared is the label sources and not the planners. Both planners are
Gemini, so the invariance holds across model size and reasoning mode but not
across vendors, and the source paper's own motivating example used a
different family. And no robot moved, so this measures plans, not
executions, and closes the gap between labels and robot behaviour by one
link rather than entirely.

**The paired tests, in full.** Twenty-five scenes is small, and the pairing
is what makes it enough: every condition is put to the same scenes, so the
evidence sits in the scenes where two conditions disagree, and an exact
McNemar test over those (`eval/planner_paired_tests.py`) says which
comparisons this sample can settle. Supplying relations at all separates
from supplying none, 25 discordant scenes for the human labels and 19 for
the tool's, all in one direction, p < 10^-5. The union's gain over the tool
alone is 6 scenes to 0, p = 0.031, as is the human arm's lead over the tool
alone, the comparison that runs against this project. What 25 scenes cannot
settle is named too: the tool against the vision-language source is 6
discordant scenes to 5, p = 1.00, so the two-scene margin means nothing and
§5.7 declines to read it, and the union's edge over that source alone, 5 to
0, reaches only p = 0.063. The paired tests are also sharp exactly where the
absolute rates are not: C's own rate is 19 of 25 with a 95% interval of
[0.55, 0.91]. The experiment therefore measures *which source is better on
these scenes* far more precisely than how often any of them would succeed in
general, and that generalisation is weak because of what the scenes are,
not how many.

**The scoring defect that manual checking caught.** Models frequently open
with a preamble restating the task ("To pick up box0 safely, follow these
steps:"), and counting that preamble as a step made every such plan appear
to grasp the target before clearing anything. Before the fix condition C
scored 0.64; after it, 0.88, and the post-fix number is the correct one. Both
were measured on the pre-refit labels, where C stood at 22 of 25; on the
shipped labels of §4.14 the same corrected scorer gives 19 of 25 (§5.7).
The episode is recorded because it is the only evidence that the blind
scorer measures what it claims to: a rule-based judge inherits whatever its
author failed to anticipate, and the hand-read sample is what exposed this
one.


### E.6 The scale run: timing and the distribution check

Throughput and density in Chapter 4 are measured on the 836 annotated images,
which leaves the claim that the method extends to new captures resting on an
extrapolation. The 1,766 unannotated frames of the raw sequence (§4.12,
Appendix A) remove that gap: robot output nobody has labelled, from the same
platform but later in the session, with arrangements the tool has never been
shown. Content-adaptive selection reduces them to 562 keyframes, 3.1×, at
6.15 s per frame on the RTX 2060, finishing in 58 minutes against just over
three hours for every frame (`outputs/extension_scale.json`). The run emits
185,242 triplets, 330 per frame, with no empty graph anywhere, against the
human process's 8,926 triplets across 836 images. The risk with unfamiliar
input is not loud failure but quiet drift, a pipeline still emitting labels
whose distribution has silently changed, and that does not happen: against the
same detector on the annotated images the predicate distribution shifts by a
total variation distance of 0.032, largest single move 0.015.

What this demonstrates is capacity and stability, not correctness. There is no
ground truth and no accuracy claim is made; establishing correctness on this
portion needs labels that do not exist, and §4.12's viewpoint-consistency
measurement is the closest available substitute, being a check on
self-agreement rather than on truth. Two figures above need accounting for.

*Yield and detections.* The run averages 330 triplets per frame from 11.7
detected objects, and the human process recorded about 11 triplets per image.

*Timing.* 6.15 s per frame is 586 frames per hour. It includes writing an inspection overlay per
frame to make the output checkable by eye. Annotation is roughly half of it,
so a JSON-only deployment run would land near 3.3 s per frame. The measured
figure leads above because it is the one the repository reproduces.

*The two distribution differences.* The largest predicate shift is *in front
of*, 0.181 to 0.195. The `on`/`under` share is low in both runs, 0.006
annotated against 0.003 here, which is a property of open-vocabulary
detection rather than of the new frames: more detections means more pairs,
and most pairs are not in contact. Density per frame is lower for the same
reason in reverse, 330 against 633 triplets, since the later arrangements
hold fewer objects (11.7 detections against 16.9) and pair count grows with
the square.

### E.7 The blind audit: sampling, guards and key handling

Section 4.14 reports the instrument and the result. The construction rules
are here, because a blind audit is only as good as the independence of the
items in it.

**Sampling independence.** At most one claim is drawn per (annotator group,
subject class, object class). Section 4.12 establishes that each group is one
continuous walk holding a single physical arrangement, so two claims sharing
a group and a class pair can otherwise be the same physical relation seen
from two viewpoints, which would make the sample narrower than its item count
suggests.

**Class guard.** The shipped class guard is applied before sampling, so
nothing is audited that the tool would no longer emit. Without it the audit
would measure a rule set the project does not distribute.

**The pre-refit (v3) pack in full.** Section 4.14 quotes this pack's support
rows and carries its author column beside the v4 re-audit; the complete
per-predicate verdicts, with Wilson 95% intervals, are these:

| Predicate | Author | Model |
|---|---|---|
| on | 16/43 0.372 [0.24, 0.52] | 25/43 0.581 [0.43, 0.72] |
| under | 22/51 0.431 [0.31, 0.57] | 35/51 0.686 [0.55, 0.80] |
| to the left of | 22/24 0.917 [0.74, 0.98] | 23/24 0.958 [0.80, 0.99] |
| to the right of | 23/24 0.958 [0.80, 0.99] | 22/24 0.917 [0.74, 0.98] |
| in front of | 23/24 0.958 [0.80, 0.99] | 22/24 0.917 [0.74, 0.98] |
| behind | 22/24 0.917 [0.74, 0.98] | 20/24 0.833 [0.64, 0.93] |
| near | 24/24 1.000 [0.86, 1.00] | 15/24 0.625 [0.43, 0.79] |
| **support pooled** | **38/94 0.404 [0.31, 0.51]** | **60/94 0.638 [0.54, 0.73]** |
| decoys rejected | 19/28 0.679 [0.49, 0.82] | 24/28 0.857 [0.69, 0.94] |

**Why a model may judge what §4.13 shows it cannot annotate.** One objection
arrives immediately: §4.13 spends a section establishing that a
vision-language model makes a poor annotator, and §4.14 then gives one a
vote. The two tasks differ in the half that failed. What §4.13 measures is
*coverage*, the model never having addressed 171 of 381 gold triplets, 44.9%,
so its headline recall is mostly that silence, while on the pairs it did judge
it was the *more precise* of the two, 0.419 against the pipeline's 0.347.
Judging a claim that is handed to it asks only for the half that measured
sound, since the item is supplied and nothing has to be enumerated. That
would still be only an argument if the audit did not test it, and the decoys
test it: the model rejected 24 of 28 relations the tool never emitted,
against the author's 19 of 28, so on this pack it is the stricter of the two
and not a judge that agrees with whatever it is shown.

**The threshold repair, in sequence.** The cause §4.14 identifies is a
threshold fitted where its error was invisible: sorted by the contact
fraction the rule fires on, audited claims below 0.85 are correct 1 time in
11 (4/44) and above it 2 times in 3 (34/50). The obvious response, raising
the threshold until precision recovers, cannot be evaluated on the sample
that suggested it: a cut-off chosen by inspecting those 94 verdicts and then
scored against them would be optimistic by an unknown amount, which is the
error that produced the 0.9 in the first place. The cut-off was therefore
fitted the way every threshold in Chapter 3 is fitted, on the 63 audited
claims from annotator groups 0–5, where precision rises steeply to 0.686 at
0.85 and flattens after; on the 31 claims from groups 6–8 that no part of
the fit saw, it predicted **0.367 → 0.667**. `on_contact_min` was then set
to 0.85 and every experiment in this dissertation re-run against the new
labels. That prediction is not what §4.14 reports, because a projection from
held-out items selected under the *old* labels is still an extrapolation: a
second pack was drawn from the new emissions instead, at 219 items, 191 claims
and 28 decoys, same construction, same blinding, same two judges, and
audited independently, which is the v4 column there.

**Where the two judges agree, and where they do not.** Over the v3 pack they
agree closely on the laterals and the depth pair, to within 0.042 and 0.083
respectively, which is what makes the `near` divergence of 0.375 that §4.14
reports stand out rather than read as general disagreement.

**The decoys as a measure of the judges themselves.** The pack's own control
rows say something about each judge that the claim rows cannot. Both judges
rejected **all eight** support decoys, so on the predicate that failed
neither is disposed to agree with the tool for the sake of it, which is what
rules out reading 0.404 as an auditor being harsh. Elsewhere the author is
the more generous of the two, and consistently so: three of four `behind`
decoys accepted against the model's one, and two of four for `in front of`
and for `near`. That is a measured author bias, confined to the family §4.5
shows the annotators themselves used inconsistently, and it is reported
rather than corrected because the same instruction governed both judges.

**What the direction of the restricted-versus-audited gap says about the
human record.** For five predicates restricted precision badly understates
audited precision, and for support it overstates it, and the direction is
diagnostic of what the gold *is*. A lateral relation holds for nearly every
ordered pair and the annotators wrote down a handful, so their labels are a
small sample of a large truth and the tool's extras are mostly further
instances of it. Support is rare and salient: a thing resting on a thing was
worth recording and was recorded, so the human labels are close to the
complete set of easy cases and what the tool adds beyond them is mostly not
there. Restricted precision therefore understates a predicate whose gold is
a sample and overstates one whose gold is nearly exhaustive, and no single
reading of §4.3 is correct for both, which is why the protocol pairs it
with an audit instead of reporting either alone.

**Judge drift between the two packs.** Nine of 28 decoys were accepted in v3
and one in v4, so the second sheet is the work of a stricter judge than the
first. The model's decoy rejection rose over the same interval, 0.857 to
0.929, which suggests the v4 pack is also the easier of the two to judge.
Neither movement accounts for the size of the support change, but both are
reasons to read the v3 and v4 columns of §4.14 as two measurements rather than
one series.

**Blinding.** Claims and decoys are shuffled together and the answer key is
written to a separate file that the judging interface never reads. The key is
not distributed with the repository, since publishing it would make any
later re-audit unblindable.

## Appendix F: Supplementary tables and figures

### F.1 Per-predicate and per-slice results at seed 42

Section 6.3 reports the headline benchmark result and Appendix F.1 the two
decompositions it draws on. Both come from the same pair of best checkpoints
scored on the same 210-image test set, at seed 42 only, and from the same
single training session as every other figure in Chapter 6; §6.3.1 replicates
the slice decomposition across three seeds, and where the two disagree the
three-seed figures are the ones any claim rests on. They do disagree here: at
this seed the automatic arm leads on group 6, where the three-seed mean puts
the human arm ahead, which is the disagreement the replication exists to
expose. Per-predicate figures are parsed from the evaluation log of the
all-arms run (`scripts/kaggle/executed_6_all_arms_085.ipynb`, collected in
`outputs/sgg_benchmark/per_predicate_085.json`).

| per-predicate mR@100 | human-trained | auto-trained |
|---|---|---|
| on | 0.571 | **0.648** |
| under | 0.454 | **0.738** |
| to the left of | **0.185** | 0.165 |
| to the right of | **0.350** | 0.232 |
| in front of | **0.078** | 0.060 |
| behind | **0.135** | 0.084 |
| near | **0.117** | 0.108 |

| test slice (mR@100) | human-trained | auto-trained |
|---|---|---|
| full test, as annotated | 0.270 | **0.291** |
| full test, conventions aligned* | 0.310 | **0.312** |
| group 6 alone (inverted convention) | 0.299 | **0.309** |
| group 7 alone (consistent annotator) | 0.254 | **0.280** |
| group 8 alone (inverted, dense `near` user) | **0.130** | 0.123 |

\* groups 6/8's front/behind gold flipped, one disclosed bit per group, as in
§4.5.

The seed ranges above absorb one further source of variation. Re-scoring the
*same* checkpoints a second time does not reproduce them exactly: an
independent re-evaluation pass moved the pooled human mR@100 by 0.001 and the
group-8 figures, drawn from the smallest slice at 37 images, by up to 0.008.
Inference is not bit-deterministic, so a margin of that order is not a result.
The figures reported are those of the re-evaluation committed with this
repository, and no claim in Chapter 6 turns on a difference smaller than the
per-seed spread.

### F.2 Downstream recall by label source

Section 5.2 carries the per-predicate figures and the seed spreads. The chart
below is the same experiment drawn instead of tabulated, which makes the
ordering easier to take in at once.


### F.3 Design decisions and the alternatives rejected

Section 3.12 states what the decisions have in common and names the four that
later evidence could have overturned. The full table is here.

| Decision | Alternative rejected | Why |
|---|---|---|
| Compute relations from geometry | learned relation model | supplies (not consumes) training labels; auditable; valid for spatial predicates |
| PredCls evaluation | detector-in-the-loop RQ1 | isolates the contribution; detection already established by the paper |
| SAM2 masks, multimask best-score | box-only geometry | robustness (empty-mask failure measured); box-only kept as ablation |
| Depth Anything v2 Small, relative | metric/stereo depth; larger variants | data is mono RGB; 6 GB budget; Apache-2.0 licence; **and measured: the Base model gives identical accuracy (A8), so the limit is monocular ambiguity, not model capacity** |
| Median masked depth | mean | robust to mask edge bleed |
| Abstention bands + flags | forced binary decisions | converts model uncertainty into measurable human cost |
| Ground-plane fallback for depth ties | metric depth models; multi-frame fusion | free 2D cue, pixel-precise in the depth band; guarded by own contact evidence; metric depth needs new capture, multi-frame breaks the single-image contract |
| `near` = relative box gap + contact exclusion | 3D centroid distance | measured: centroid metrics don't transfer (F1 ≤ 0.024); near never co-occurs with contact (0/469) |
| Annotator-aware `near` fit | fit/test across all groups | only 3/9 groups used the label; naive protocol conflates annotator habits with tool error |
| Byte-compatible writers | own format + converter | RQ1/RQ2 comparability; verified zero-error |
| Config + geometry cache | ad-hoc constants, full re-runs | reproducibility; 20 s offline re-evaluation |

### F.4 The vision-language training arm, by test slice

Section 6.3.2 reports what this arm establishes and the one result that does
not fit. Mean mR@100 over three seeds, per-seed range in brackets.

| slice | human | automatic | vision-language |
|---|---|---|---|
| full test | 0.293 (0.270–0.322) | 0.292 (0.290–0.296) | **0.329** (0.312–0.357) |
| group 6 (defect) | 0.327 (0.299–0.364) | 0.305 (0.296–0.312) | **0.337** (0.316–0.366) |
| group 7 (clean) | 0.278 (0.254–0.298) | 0.289 (0.280–0.300) | **0.362** (0.357–0.364) |
| group 8 (defect) | 0.147 (0.130–0.164) | 0.120 (0.108–0.131) | **0.164** (0.146–0.183) |
| aligned gold | 0.333 (0.310–0.369) | 0.316 (0.312–0.320) | **0.369** (0.354–0.390) |

Three readings of the group-7 result remain open and this experiment does not
separate them. The arm trains on 14,626 relations against the human arm's
5,421, so the gain may be density rather than the source. Group 7 is 73
images, small enough that a 0.07 margin over three seeds is suggestive rather
than settled. And the arm's assertions were never audited as the tool's were
(§4.4), so their correctness is assumed, not measured. The position this
supports is
that on clean gold a vision-language source is at least competitive with both
alternatives, and that establishing why would need the audit and a larger
clean slice, neither of which this project has.

### F.5 Downstream indicators beyond recall

Section 5.2.1 states the finding and why the columns cannot be read as an
error rate. The full table is below, macro-averaged over predicates except
where marked.

| arm | macro R | macro P | macro F1 | macro AP | micro F1 |
|---|---|---|---|---|---|
| human-trained | 0.297 | **0.252** | 0.267 | **0.230** | **0.262** |
| pseudo-labelled | 0.365 | 0.243 | **0.289** | 0.215 | 0.273 |
| vision-language | 0.380 | 0.197 | 0.253 | 0.219 | 0.221 |
| auto-trained | **0.758** | 0.136 | 0.194 | 0.164 | 0.066 |

Two things follow from it, the second a concession. The result strengthens
rather than weakens the reading Chapter 6 reaches independently, because it
shows a metric rewarding agreement with annotation practice favouring
whichever arm imitates it, here in a controlled experiment with a different
model, different features and a different metric family from the benchmark's.
But strictly these precision figures are uninterpretable rather than
favourable: the audit of §4.4 covered the rule layer's extra predictions, not
the classifier's, so auditing a stratified sample of the classifier's own
false positives is what would settle them, and that is not in this
dissertation.

### F.6 Restricted precision on the annotated pairs

Section 4.3 quotes these figures and states why they are a floor and not
an error rate. Recall and F1 are restricted to the same 8,790 annotated pairs.

| Predicate | P | R | F1 | support |
|---|---|---|---|---|
| on | 0.95 | 0.81 | 0.88 | 1465 |
| under | 0.92 | 0.75 | 0.83 | 1001 |
| to the left of | 0.35 | 0.96 | 0.51 | 972 |
| to the right of | 0.42 | 0.98 | 0.59 | 1174 |
| in front of | 0.43 | 0.70 | 0.53 | 2013 |
| behind | 0.35 | 0.71 | 0.47 | 1584 |
| near | 0.11 | 1.00 | 0.20 | 717 |

### F.7 Bounding annotator agreement without overlapping assignments

Section 4.6 reports this bound as a measurement this dataset cannot support
and gives the reason there. The derivation and the batch figures behind that
refusal are here.

Against any common reference, if annotators A and B agree with it on
fractions p_A and p_B, their mutual agreement obeys the Fréchet inequalities
max(0, p_A + p_B − 1) ≤ p_AB ≤ 1 − |p_A − p_B|. Averaged over all 21 pairs of
consistent annotators these place annotator-to-annotator agreement in
[0.78, 0.96], and the tool's own mean agreement of 0.892 lies inside it
(`eval/annotator_agreement.py`).

Two things limit what that is worth, and neither is buried. Fréchet bounds
are the loosest bounds available, so an 18-point interval rules little out:
had the tool scored 0.80 or 0.94 the same sentence could be written. And the
groups labelled disjoint batches, so the bounds presume each annotator's rate
would carry to another ~100-image batch.

That presumption is doubtful, and the dataset can say so. The batches differ
substantially in what they contain: mean objects per frame runs from **4.47 in
group 4 to 14.30 in group 1**, a coefficient of variation of 0.31 across the
nine, and pair count grows with the square of that. An annotator working a
sparse batch faces fewer and easier decisions than one working a dense batch,
so their agreement rates with a common reference are not measured under
comparable conditions and cannot be assumed transferable between them. The
Fréchet inequalities themselves hold for any joint distribution and are not
affected; what the batch heterogeneity undermines is the step *before* them,
which treats p_A and p_B as though they described the same task. The interval
is therefore an estimate resting on an assumption this dissertation can show
to be false and not merely unverified, and it should be read as
indicative only. The assumption-free
half of §4.6 is the heterogeneity spread, which needs no such presumption:
the tool is deterministic, so the 0.082 range in its agreement across the
seven consistent annotators (sd 0.028) is variation in the annotators and
nothing else, though a range that narrow carries correspondingly little
weight. It is also much narrower than it was: on the pre-refit labels the
same spread was 0.216, and the support refit of §4.14 closed most of it by
lifting group 3 from 0.72, which is why the case that the consistent
annotators differ rests on the two inverted groups and the measured defects
of §4.5 and §4.9 rather than on this band.

### F.8 Held-out cluster-bootstrap intervals

Section 4.6 quotes the two depth predicates and states that the other five sit
at 0.82 or better. The full set is below, from the same 2,000-resample
cluster bootstrap over images described in §4.2, computed on the held-out
annotator groups only.

One caveat belongs with every interval here, and it follows from a finding
made later than the procedure. Clustering on the image is the right unit
for the dependence *inside* a scene, which is what §4.2 argues, but it
resamples images as though they were independent draws. They are not:
§4.12 establishes that these 836 images are consecutive frames of one
robot trajectory, with the pipeline's own labels persisting across
neighbouring frames at 0.90–0.92. Adjacent frames therefore carry much
less than one image of independent information each, the effective sample
is smaller than 836, and these intervals are narrower than a
block-resampling scheme over contiguous runs of frames would give. They
should be read as a lower bound on the width, not as calibrated coverage;
the point estimates are unaffected, since they are the population value
for these images. A replication should resample trajectory segments
rather than frames, which the released per-image outputs support.

| Predicate | Recall (held-out) | 95% interval |
|---|---|---|
| on | 0.853 | 0.813–0.890 |
| under | 0.823 | 0.768–0.875 |
| to the left of | 0.953 | 0.937–0.970 |
| to the right of | 0.985 | 0.974–0.996 |
| in front of | 0.199 | 0.148–0.257 |
| behind | 0.369 | 0.300–0.443 |
| near | 1.000 | 1.000–1.000 (see note) |

The `near` row is a boundary artefact and not extraordinary precision. The
held-out groups carry 93 `near` gold triplets and the tool misses none, so
every resample returns the same value and the bootstrap has no spread to
report. The number to quote is the exact one-sided bound for zero failures
in 93 draws, **0.968**, and even that is a statement about recovering the
*recorded* `near` labels: §4.8 sets out why this predicate's recall says
least about the labels the tool adds beyond them.

### F.9 The test gold's convention contamination, and the second labelling rule

Section 6.3.1 states the finding; the arithmetic behind it is here, because
it bounds every absolute figure in that chapter.

**How much of the yardstick is affected.** Of the 2,818 relations in the
benchmark test gold, **1,189 (42%) are front/behind, and 859 of those (72%)
were written by the two annotators §4.5 convicts of inverting the
convention**, so **30% of the entire yardstick is a predicate labelled in
the opposite direction to the convention every training group used**. Both
arms train on groups 0–5, where no inversion is measured, so neither can
score those relations and the penalty falls on them equally: the inversion
sets a *ceiling* instead of a differential.

**How far it bites depends on how the metric aggregates.** R@100 counts
instances, so the cap applies to 30% of them directly. mR@100 averages over
the seven predicates, so front and behind carry two sevenths of it however
many instances they hold, and what the 72% figure sets is a ceiling near
0.28 on those two components rather than a 30% reduction in the mean. Both
metrics are depressed and every absolute figure in Chapter 6 is a lower
bound on both sides, but the 30% is a share of the gold and not a share of
mR@100.

**The two mechanisms that erase Chapter 5's advantage.** Section 6.4 names
four; two need their evidence set out. *Ranking dilution.* R@K is a
per-image ranking budget, so any prediction absent from the gold consumes
budget as a miss. The auto-trained model predicts densely, like its
supervision, and Chapter 4's audits established that such extras are
overwhelmingly true, so against gold annotating ~10% of pairs they outrank
the annotated ones and are scored as errors -- §4.3's restricted-precision
artefact reproduced at benchmark level. The human-trained arm learned the
annotators' labelling prior instead, which is what a ranking metric against
human-selected gold rewards. *Convention mismatch is a shared penalty.*
Re-scoring against aligned gold lifts both arms almost equally, +0.041 human
and +0.035 auto, the human arm's *in front of* recall jumping 0.124 to 0.386
and the auto arm's 0.101 to 0.248, so both pay the same tax and the gap
barely moves; the initial hypothesis that the denser arm is punished harder
for its confidence is refuted by that measurement and withdrawn. It
replicates at the shipped threshold, the retrained auto arm moving from
0.292 to 0.316 mR@100 and 0.255 to 0.292 R@100 against aligned gold, +0.025
against the earlier +0.035, so roughly a tenth of the absolute mR@100
anywhere in Chapter 6 is an artefact of that defect, on both sides.

**Where the difference between the arms actually sits.** Section 6.3.1
reports the ordering; the per-annotator detail is this. The human arm leads
on the two annotators §4.5 convicts of inverting the convention, by 0.022 on
group 6 and 0.027 on group 8, and *trails* on group 7, the one annotator
this dissertation convicts of nothing, by 0.011. None of the three
differences is separable across seeds, so the ordering is offered as a
consistent direction and not as three measured effects; what makes it worth
reporting is the sign change on the clean annotator, since whatever
advantage the human labels carry does not survive contact with an annotator
who followed the stated convention.

Group 6 also shows the fingerprint §6.4 attributes to annotator *selection*
rather than convention. Its *lateral* gold, geometrically unambiguous
relations that both models predict freely, is recalled at 0.48/0.68 by the
human arm against 0.31/0.24 by the auto arm, over three seeds with disjoint
ranges. Laterals have no convention to invert, so what differs there is
which pairs the annotator chose to record, and the human-trained model
ranks exactly those highly because it learned human selection habits rather
than more geometry, a smaller gap than the pooled group-6 lead this
section opens with, because a shared-convention predicate has nothing else
inflating it.

| Predicate | Human-trained recall | Auto-trained recall |
|---|---|---|
| to the left of | 0.48 (0.47–0.49) | 0.31 (0.24–0.37) |
| to the right of | 0.68 (0.61–0.72) | 0.24 (0.21–0.26)|

**Two labelling rules support the ordering, not one.** The figures in §6.3.1
are the shipped `on_contact_min` of 0.85 (§4.14); the same experiment at the
earlier 0.60 gave 0.278, 0.286, 0.307 and 0.109 across the same four slices.
Raising the threshold improved three slices and cost the fourth, and the
ordering by annotator defect held under both. A pattern that survives
changing the labelling rule is a property of the test annotation rather than
of one configuration of the tool.

### F.10 The front/behind decomposition by annotator group

Section 4.5 reports the finding and its figure plots the decomposition; the
per-group figures underneath it are these, for the shipped cascade (depth
ordering plus the ground-plane fallback of §4.9, which lifted the emit rates
of the abstention-heavy groups 2 and 3 by about half):


| Group | Gold | Emit rate | Agreement when committed | Convention | Raw recall | Aligned recall |
|---|---|---|---|---|---|---|
| group_0 | 724 | 0.94 | 0.95 | same | 0.89 | 0.89 |
| group_1 | 639 | 1.00 | 1.00 | same | 1.00 | 1.00 |
| group_2 | 351 | 0.85 | 1.00 | same | 0.85 | 0.85 |
| group_3 | 258 | 0.82 | 0.99 | same | 0.81 | 0.81 |
| group_4 | 65 | 1.00 | 0.57 | same | 0.57 | 0.57 |
| group_5 | 371 | 1.00 | 0.99 | same | 0.98 | 0.98 |
| group_6 | 415 | 0.99 | **0.05** | **inverted** | 0.05 | 0.94 |
| group_7 | 330 | 0.94 | 1.00 | same | 0.94 | 0.94 |
| group_8 | 444 | 0.84 | **0.02** | **inverted** | 0.02 | 0.82 |
| **overall** | 3597 | | | | **0.70** | **0.91** |

### F.11 Reading the headline recall table

Section 4.2 carries the table and states the result; these are the three
readings underneath it, and the qualification that belongs with `near`.

**The baselines, read two ways.** The tool recovers 81% of all human
triplets, 7,276 of 8,926, against 14% for random and 23% for majority on the
same triplet-weighted basis. The mean row of Table 4.1 instead puts majority
at 0.14, which is the right number for a per-predicate question and the
wrong one for this comparison: guessing `in front of` everywhere recovers
2,013 of 8,926 triplets, because that predicate is a quarter of the gold.
Both figures are reported so that neither reading can flatter the tool by
itself.

**What the box-only column does and does not show.** On recall alone
box-only geometry is level with the full pipeline on the lateral pair and
slightly ahead on support, 0.84 and 0.77 against 0.81 and 0.75, while
falling 0.13 behind on `near`. Recall is the wrong axis to read that on: the
mask rule was adopted because it lifts support *precision* and recall
together, held-out support F1 0.71 to 0.87 (A5, §4.9), and a looser box rule
buys its extra recall with the false fires §4.4 audits. What the column does
show is that masks earn their place on support quality and on `near` rather
than on lateral recall, and that the pipeline's unshared advantage is the
depth pair, 0.70/0.71 against 0.00. Even there the ground-plane fallback, a
pure box cue, needs masks to fire, because its elevation guard is
mask-contact evidence (§4.9).

**Why held-out beats pooled on some predicates and collapses on others.**
Held-out exceeds pooled on on/under/near and falls far below it on
front/behind, and both are annotator signatures: convention inversion for
the depth pair (§4.5), and direction-usage asymmetry for support, where
several groups label one direction only, group_2 recording 188 *on* and no
*under* and group_8 only *under*, while the held-out groups' support labels
happen to be canonical stackings the rules recover at 0.95–1.00.

**The `near` qualification.** `near` is recovered completely once its
inconsistent usage is accounted for, 0.997 pooled and 1.00 held-out, which
answers the predicate the source paper reports as failing for every model it
benchmarks (§2.2). That is a claim about recall. On the precision side
`near` is the weakest of the five audited predicates: 0.792 audited, the
widest disagreement between the two judges, and two of its four decoys
accepted by the author (§4.14). A rule that fires on sixty times more pairs
than the annotators labelled will recover their labels almost by
construction, so this is the predicate where the recall figure most
overstates what is known.

**Where the residual depth error sits.** Section 4.5 decomposes the depth
pair's shortfall into calibrated abstention and a convention the annotators
did not share, in measured proportions. Genuine depth error is the remainder
and not absent: Appendix D.7 measures it at 1% of `in front of` misses
and 6% of `behind`, so the dominant terms are not depth error, which is a
weaker claim than none of it being.

### F.12 The reasoning behind the answers

Chapters 4 to 6 and §9.1 state their verdicts; the weighing behind them is
collected here, so that each answer can be read as an answer.

**Why RQ1 divides by predicate rather than by mean (§4.8).** The two axes
divide the seven predicates differently. On recall, five reach
human-comparable levels and the exception is the depth pair; on precision, a
*different* five audit blind at 0.79–1.00 and the exception is support. So
no predicate is weak on both axes and none is strong on both except the
laterals and `near`: support recalls well and cannot be trusted where it
adds, and the depth pair is trustworthy where it commits and commits less
often than the annotators did. A mean over seven would conceal exactly that
structure, which is why §1.2.2 required the answer per predicate.

**Why the classifier result is scoped rather than headline (§5.8).** At this
dataset's scale of human annotation the automatic labels are not merely good
enough but substantially better training material, because density and
consistency dominate raw human authority when supervision is sparse; the
self-trained arm rules out the cheap alternative, recovering only 15% of the
gap because it propagates the annotators' silence rather than their
knowledge. That is the dissertation's core claim, that removing the bottleneck
can *grow* dataset utility rather than approximate it, demonstrated on the
dataset's own held-out annotators and against the obvious rival. Its scope
is set by what follows it: Chapter 6 returns parity, so the advantage is an
advantage on the controlled classifier where features are held identical,
and it does not carry to the benchmark's ranked metric.

**Both readings of the benchmark, and why neither is available alone
(§6.5).** The benchmark result is real: a consumer *evaluated against
human-annotated scene graphs* is better supervised by human labels, which
carry the annotation prior the evaluation shares. The interpretation is
equally real: the ranking metric inherits every defect measured in the gold,
and the advantage is concentrated exactly where the annotation is defective
and absent where it is not, which is what annotation-prior agreement would
look like. Section 1.2.2 set a non-inferiority criterion, so parity is the
shape a pass takes; but the numbers refuse a strong claim in either
direction, the paired mean difference being -0.0006 with a 95% interval of
[-0.070, +0.069], three seeds bounding the gap to about a quarter of the
metric's own value, and a margin of ±0.01 needing roughly forty runs per
arm. What the same three seeds *do* resolve is the part that is not a null:
zero-shot recall separates with disjoint ranges, 0.225–0.309 against
0.004–0.079, as does reproducibility, a 0.006 spread against 0.052, both
running the automatic arm's way. Set against the cost of obtaining them,
nine annotators against five minutes on one consumer GPU,
indistinguishability on the ranking metric is close to the result the
project set out to obtain, since RQ1 and RQ2 ask whether the human can be
removed and not whether the machine wins.

**Why the critical reading is credible rather than convenient (§6.5).**
Neural Motifs (Zellers et al., 2018) established that a frequency baseline
ignoring the image is hard to beat; Unbiased SGG (Tang et al., 2020)
formalised how thoroughly such models absorb the annotation distribution;
and Northcutt, Athalye and Mueller (2021) showed erroneous test labels
reorder rankings across ten benchmarks. What Chapter 6 adds is a case where
the confound is *isolated by construction*, since the arms differ only in label
source, share a frozen detector, and the per-annotator defects were measured
beforehand in Chapter 4, so the advantage is attributed to annotation
practice and not inferred. The one remaining instrument is a manual
audit of the auto arm's top-ranked false positives, the analogue of §4.4,
left as designed follow-up.

**What the two counting conventions do to RQ1's comparability (§9.1).** The
criterion named two references. The trivial baselines are beaten by a wide
margin. The second, how well two human annotators would agree with each
other, this dataset cannot supply: the nine groups labelled disjoint batches
differing threefold in object density, so §4.6 reports the Fréchet bound as
a measurement the data cannot support, and the assumption-free spread of
0.082 is too narrow to carry weight either. Comparability therefore rests on
the trivial baselines and the per-predicate audit, and the yardstick §1.2.2
hoped for is one this dataset could not supply, which is itself a finding
about the dataset's construction rather than a gap in the evaluation.
