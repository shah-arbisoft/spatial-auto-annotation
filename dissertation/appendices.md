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
- `python eval/keyframe_propagation.py --sweep 5,10,20,30`: content-adaptive
  frame selection and the viewpoint-stability measurement of §4.14 →
  `outputs/keyframe_propagation.json`
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

The resulting image, 7.02 GB over eleven layers, was then checked as an
artefact rather than trusted on the strength of its log, because the machine
that built it ran out of disk during the export stage and a partly written
layer is not always an obvious failure. Reading every one of the 84,972
files in the image returned no errors, so no layer is truncated. Inside the
finished image the test suite reports **66 passed** a second time, torch
resolves to `2.5.1+cu121` rather than the CPU wheel the pitfall above
produces, and `pip check` is clean apart from a platform-metadata note on
`ninja`, a build-time dependency of SAM2. Every one of the 60 Python and YAML
files under `src/`, `eval/`, `tests/`, `configs/` and `scripts/` was
hash-identical to the working tree as it stood at the commit the image was
built from (`a10378f`); analysis scripts added afterwards are in the
repository but not in that image, which is the ordinary consequence of an
image being a snapshot. `/app` is 2.0 MB, and the dataset, the caches and the
credentials file are absent, `.env.example` being the only environment file
shipped.

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

**3. One GPU pass, then everything is offline.**
`python scripts/run_annotator.py` (~5 min on the RTX 2060) writes the
annotations in all three native formats plus the geometry, contact and
depth caches under `outputs/`, and `outputs/pairs.csv`. Every experiment
below runs from those caches on CPU:

| command | produces | time |
|---|---|---|
| `pytest -q` | 66 unit and invariant tests | <1 min |
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

It is worth
asking whether the depth pair would improve simply by using a stronger depth
network. It does not: swapping Depth Anything v2 Small for the 4× larger Base
variant and re-running the whole dataset moves front/behind recall by +0.001
and +0.002 (0.640/0.654 → 0.641/0.656) and leaves mean recall marginally
*lower*, 0.848 against 0.847. The depth-predicate
limit is *monocular ambiguity* (two objects at a similar camera distance are
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

Two studies in Chapter 4 report their headline result in the chapter and
their supporting detail here: the vision-language baseline of §4.16, whose
diagnostics are what make its failure interpretable rather than merely
worse, and the viewpoint-stability measurement of §4.14, whose segmentation
evidence and coverage limits qualify how far the stability figures reach.

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
