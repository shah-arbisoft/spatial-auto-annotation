# Chapter 3: Research Methodology and Design

> Written alongside the build, so every design claim below is
> implemented and, where stated, measured. Numbers marked *(measured)* come from
> runs over the released dataset (836 annotated images); the fitting protocol and
> per-annotator analysis are reproducible from `eval/fit_near.py` and
> `outputs/near_fit.json`.

This chapter sets out how the project was conducted and why the annotator is
built the way it is: the methodology that structures the work (§3.1), the
problem and its constraints (§3.2), the design and the seven predicate rules
that follow from it (§3.3–§3.11), and every decision collected in one table
(§3.12).

## 3.1 Research methodology

Three process models dominate data-science practice. **KDD** (Fayyad,
Piatetsky-Shapiro and Smyth, 1996) frames the work as a nine-step pipeline
from selection through transformation to interpretation, oriented at
discovering patterns in existing databases. **SEMMA** (Sample, Explore,
Modify, Model, Assess), associated with the SAS toolchain, compresses this
into five tool-centred stages and says little about the surrounding business
problem. **CRISP-DM** (Wirth and Hipp, 2000) adds two things the others lack:
an explicit *Business Understanding* phase before any data is touched, and an
explicit cycle in which evaluation results feed back into earlier phases.
Azevedo and Santos (2008) compare the three and observe that CRISP-DM is
effectively a superset of SEMMA and an implementation of KDD with stronger
process guidance.

CRISP-DM was chosen for two concrete reasons. The motivating problem is not
pattern discovery but an engineering question posed by a dataset's own
authors, so a phase pinning the business problem (the annotation bottleneck,
Chapters 1–2) before modelling is structurally necessary. Decisively, the
project's actual course followed CRISP-DM's evaluation-to-modelling loop in a
way the linear models do not describe: the audit of support-rule precision
(Chapter 4) sent the work back to modelling twice, a depth gate then a
mask-contact test, each pass re-validated on held-out annotators. §1.3 maps
every phase to a concrete part of this dissertation, including the two
findings that came straight out of Data Understanding: the stored image
orientation and the three measured annotator behaviours. Ethical
considerations are summarised in §1.3 and detailed in Appendix A.

## 3.2 Problem analysis

The task is to produce, from a raw robot-acquired RGB image, the same
artefact a human annotator produces with SGDET-Annotate: a scene graph over
the dataset's six object classes with the seven spatial predicates (*on,
under, to the left of, to the right of, in front of, behind, near*) exported
in Visual Genome JSON, YOLO txt and h5. Four requirements shape the design:

1. **No human decides any label.** The point is to remove the per-image human
   bottleneck; a rule may abstain and flag, but it may not ask.
2. **Comparability with the human labels.** RQ1 requires the outputs to align
   with the human annotations pair-for-pair and byte-for-byte in format, or no
   fair comparison is possible.
3. **Modest hardware.** A single 6 GB RTX 2060; anything heavier must be
   optional. (In practice the pipeline peaks at 0.65 GB and annotates the full
   dataset in about five minutes *(measured)*.)
4. **Reproducibility.** Every threshold in one config, every run seeded, and
   rule changes re-evaluable without re-running perception.

Three properties of the dataset, established in Chapter 2 and verified
directly on the released files, drive specific design responses:

- **Monocular RGB only.** No metric depth exists, so depth must be estimated,
  and estimated depth is *relative and per-image* (Yang, L. et al., 2024). Design
  response: all depth comparisons are ordinal and within-image; no rule
  consumes absolute depth.
- **Sparse annotation.** Humans labelled ~10% of object pairs *(measured:
  6,458 annotated of 42,440 unordered pairs)*. Design response: the tool labels
  every pair (density is the value added), and the evaluation protocol treats
  human labels as a recall target rather than an exhaustive gold standard.
- **Inconsistent `near`.** The source paper flags this qualitatively; measured,
  only 3 of 9 annotator groups ever used `near` (244/129/93 labels; the rest
  0–3), and those three each labelled a different fraction of equally-close
  pairs. Design response: a fitted, annotator-aware threshold (§3.8) rather
  than agreement with a consensus that does not exist.

## 3.3 Design principle: compute, don't predict

Learned scene-graph generators (REACT++ and predecessors; Neau and Falomir,
2026) *predict* relations from visual patterns and therefore require labelled
training data. They sit downstream of annotation and cannot replace it. This
pipeline *computes* relations from measured geometry with deterministic rules,
so it can run before any learned model and supply the labels such models
consume. The approach is valid precisely because the seven predicates are
spatial: each is decidable from positions, extents and depth order. The
perception models used (SAM2, Depth Anything) only *measure* where things are;
no learned component decides a relationship.

**Evaluation setting.** The relation stage is evaluated with ground-truth
boxes and classes (the SGG literature's *PredCls* setting, §2.7). This
isolates the contribution, since the paper already establishes detection at
0.93 mAP@50 with YOLOv10m and re-deriving that number adds nothing, and it
makes object indices line up one-to-one with the human relationship records,
so no fragile box-matching stage sits inside the RQ1 comparison.
Detector-in-the-loop operation (YOLOv10m vs. Grounding DINO) is retained as
an ablation and as the deployment mode for genuinely new images.

## 3.4 Pipeline architecture

```
image -- boxes+classes --> SAM2 masks --> depth map --> per-object geometry
                                                            |
        writers (VG JSON / YOLO / h5) <-- flags <-- correction <-- 7 rules
```

| Stage | Choice | Rejected alternative | Justification |
|---|---|---|---|
| Boxes | ground truth (study); YOLOv10m / Grounding DINO (deployment) | detector inside RQ1 | isolate the relation stage (§3.3) |
| Masks | SAM2 (Ravi et al., 2024), box-prompted, small variant; multimask + best score | boxes only | mask centroids and masked depth are robust to box slack; single-mask mode returned empty masks on loose boxes *(measured)*; box-only kept as an ablation |
| Depth | Depth Anything v2 **Small** (Yang, L. et al., 2024), HF pipeline | Base/Large; stereo/metric methods | 6 GB budget; Apache-2.0 (Base/Large are non-commercial); no metric depth exists for this data |
| Lift | centroid (x, y) + **median** depth over the mask | mean depth; full 3D reconstruction | median resists edge bleed where masks overlap background; reconstruction is unnecessary for ordinal tests |
| Fallback | empty mask → box region | image centre / drop object | a failed segmentation must not move the object; regression-tested |
| Rules | explicit thresholds, one function per predicate | learned relation head | the graded contribution; auditable and fittable |
| Correction | reject impossible label sets, demote to flags | emit everything | geometric consistency is checkable for free; adapted from Open3D-VQA's correction flow (Zhang et al., 2025) |
| Confidence | flag ambiguity bands for optional review | silent guesses | human-in-the-loop accelerator claim needs an explicit abstention mechanism |
| Writers | byte-compatible VG JSON / YOLO / h5 | own schema + converter | drop-in comparability (requirement 2); verified against real exports |

All coordinates are normalised by image size so thresholds transfer across
resolutions; depth is inverted to "smaller is nearer" and min–max normalised
per image (the HF model emits larger = nearer; the sign was fixed after
front/behind agreement rose from ~26% to ~74%). Images are loaded through an
EXIF-aware helper so the 180°-rotated captures are made upright before any
box, mask or depth is read; the boxes are stored in the upright frame.

## 3.5 The seven rules

The complete specification is **Appendix C**: every rule with its thresholds,
the shipped values, the evidence behind each one, and the correction and
flagging policy. It is also maintained in the repository as
`docs/predicate_spec.md`, which is the copy the code and tests are checked
against. The design rationale in brief:

- **on / under** encode *support*: subject above object, near-touching
  (vertical gap within ±0.05), with horizontal extents overlapping (≥0.20 of
  the narrower). This keeps "floating in front of" from reading as "on".
  `under` is defined as the strict inverse, so the pair can never contradict.
  A known edge case, a small object whose box projects entirely inside its
  support's box at shallow viewing angles, is documented and assessed in the
  ablations rather than hand-tuned early.
- **to the left of / to the right of** compare horizontal centres in the
  camera frame, the frame the annotators themselves used on screen (RoboSpatial
  distinguishes ego/world/object frames (Song et al., 2025); this dataset's
  tool shows the camera view, so the ego frame is the faithful choice). An
  ambiguity band (0.02) abstains and flags when centres nearly coincide.
- **in front of / behind** is a two-stage cascade. Depth ordering decides
  first, with an abstention band (0.03): relative depth makes these the hardest
  predicates, because objects on the same surface often differ by less than the
  depth model can resolve. Where depth abstains, a **ground-plane fallback**
  decides from pure projection: two objects standing on the same floor are
  depth-ordered by which box bottom sits lower in the image, a pixel-precise
  cue exactly where depth is noisiest. The fallback is guarded by the tool's
  own support evidence. It never fires when either object rests on another
  object (mask contact ≥ 0.60 with any partner), because an elevated object's
  box bottom says where its *support* is, not where it is; and it has its own
  small band (0.005, calibrated on train groups; on held-out group 7 every
  commit the fallback added was correct). Pairs both stages abstain on are
  flagged rather than guessed (recall 0.64/0.66 *(measured)*, from 0.52/0.55
  depth-only; both bands are recall/precision levers swept in the ablations).
- **near** is a size-relative proximity test: edge-to-edge box gap divided by
  mean object size, below a fitted threshold, and **never on contact pairs**;
  measured, `near` co-occurs with on/under on 0 of 469 human pairs, so the
  annotators used it as "close but no contact relation". A 3D-centroid metric
  was rejected on evidence: per-image depth normalisation makes centroid
  distances incomparable across scenes, and every centroid variant transferred
  to held-out annotators at F1 ≤ 0.024, while the relative-gap metric transfers
  with recall 1.0 (§3.8).

## 3.6 Correction and confidence

After the rules run on an ordered pair, mutually exclusive sets (on/under,
left/right, front/behind) are enforced; a contradiction that survives the
thresholds (mask or depth noise) demotes the pair to a flag rather than
emitting an impossible triplet. Ambiguity flags (lateral tie, depth tie,
near-threshold edge, resolved contradiction) are written alongside the
triplets. The flags are the design's honesty mechanism: the tool is presented
as a human-in-the-loop accelerator whose residual human cost is *measurable*
(the flag rate is reported with the RQ1 results), not as an infallible oracle.

## 3.7 Output compatibility

The writers reproduce the SGDET-Annotate structure exactly: centre-form
`boxes_1024`/`boxes_512` in the resized frames, index-aligned `labels` and
`attribute` arrays, `relationships` as subject–object index pairs with a
parallel `predicates` ID array, and the same six-dataset h5 layout with
int64 attributes. Verified against the real files: a load→write round trip
reproduces `boxes_1024` and `labels` with zero error, and the h5 matches a
real export key-for-key, dtype-for-dtype *(measured)*. Auto-labels are
therefore drop-in replacements for human labels, which is the property RQ2
depends on.

## 3.8 Calibrating `near`: an annotator-aware protocol

The naive protocol (fit one threshold to all `near` labels, test on held-out
images) fails, and the failure is informative. Fitting on annotator groups 0–5
and testing on 6–8 yields held-out F1 = 0.009: the training annotators' habits
do not predict the test annotator's, because the label was applied by only
three of nine groups and with very different exhaustiveness.

The adopted protocol therefore (i) uses only human-*annotated*, non-contact
pairs as fit data, since unannotated pairs are not reliable negatives under
sparse annotation and contact pairs are never `near` by the measured
convention; (ii) fits only on annotator groups that used the label within the
training split (groups 0 and 4); and (iii) reports agreement on the held-out
near-using annotator (group 8), who contributed nothing to the fit.

Results *(measured)*: fitted **T = 1.372** (gap/mean-size units); held-out
recall **1.000**, meaning every pair the unseen annotator called near lies
within the threshold, with per-annotator precision 0.41 / 0.63 / 0.16 at the
same T. Since recall is 1.0 for all three annotators simultaneously, the
human labels are directionally consistent with a single threshold; what
varies (by ~4×) is how exhaustively each annotator applied the label. The
fitted threshold applies one definition uniformly, which is exactly the
"spatial thresholds for near" the source paper's future work requests.
Whether the tool's extra near pairs (the precision gap) are genuinely near
is checked by a manual audit in the evaluation chapter rather than assumed.

## 3.9 Modularity: the detector as the replaceable part

The claim that the rules are detector-agnostic (§4.11) is an architectural
property, not a description of intent. The rule layer (`src/predicates.py`)
imports nothing but `numpy` and never receives an image; the pipeline's entry
point takes boxes as an argument, `annotate_image(image_rgb, boxes_px,
labels, segmenter, depther, cfg)`. A detector is therefore not wired into the
pipeline at all, it is simply what supplies that argument.

That is what makes the conditional measurement in §4.11 meaningful. Holding
the boxes fixed, the relation layer scores the same whether they came from a
detector or from ground truth, so detector quality and relation quality are
separately attributable, and a better detector improves the system without a
line of rule code changing.

To make the property usable rather than merely true, the contract is explicit
(`src/detectors.py`): one method returning pixel boxes, class names and
scores, with three implementations, open-vocabulary prompting where no
trained model exists, an adapter for any ultralytics checkpoint including the
source paper's YOLOv10m weights, and a reader for externally computed
detections. Twelve unit tests pin it, one driving the rule layer from a
detector written against the documentation alone. Two coupling points are
documented rather than hidden: the support guard keys on the literal class
name `human`, and the fitted thresholds assume boxes of roughly the tightness
the annotators drew, so a detector with systematically different boxes should
re-run §3.8's calibration (twenty seconds offline from the cache). A worked
example is in `docs/CUSTOM_DETECTOR.md`.

## 3.10 Selecting frames by content

The images this project annotates are not independent photographs. They are
consecutive frames of a continuous robot capture, and the sequence they were
cut from oversamples the scene severely: across its 2,650 frames the mean
optical flow between neighbours is 0.08 px, with 0.9% of pixels moving more
than one pixel. A per-frame pipeline therefore spends a full perception pass
(detector, segmentation and depth) recomputing relations that have not moved.
The remedy is to annotate one frame per *viewpoint* rather than one per frame,
which requires deciding where one viewpoint ends and the next begins.

The standard tool does not apply. Shot-boundary detection thresholds the
difference between consecutive frames, which presumes cuts, and a robot
walking through a room produces none. The failure is structural rather than a
matter of threshold choice, as §4.14 verifies by sweeping it: motion arriving
at 0.08 px per frame is never larger than the noise at any single step, while
the same motion integrated over forty frames displaces the image by 13 px, so
only the accumulated drift carries the signal.

`segment_sequence` (`src/keyframes.py`) therefore measures drift from the
*anchor* of the current segment rather than the preceding frame, opening a new
segment when drift exceeds τ, so gradual motion accumulates instead of being
rounded away while a genuine cut still crosses in one step. Distances are mean
absolute differences over 64×48 mean-subtracted greyscale thumbnails, the
subtraction discarding the global exposure shifts of an auto-exposing camera,
which would otherwise fire boundaries of their own. Each segment nominates the
frame closest to its mean signature, a better representative on a moving
camera than the first, which is usually mid-transition.

A single parameter spans two uses. Small τ isolates near-duplicates, so each
segment is one viewpoint and the representative can stand for the rest; large
τ groups several viewpoints of one arrangement, which is what the
cross-viewpoint consistency measurement of §4.14 consumes. The threshold is
chosen from the data by sweep rather than assumed, and §4.14 reports both what
the segmentation recovers and what skipping the intervening frames costs.

## 3.11 Reproducibility by construction

Reproducibility here is a design property rather than a documentation
exercise, because three of the four requirements in §3.2 are unverifiable
without it: a threshold cannot be said to be fitted on groups 0-5 if nobody
else can refit it, and an ablation is an assertion unless the reader can
re-run the arm it removes.

**Configuration and caching.** Every threshold, seed and model identifier
lives in `configs/default.yaml`, so no constant is buried in a function, and
the runner caches each object's lifted geometry after the single GPU pass.
That separates the expensive stage from the cheap one: any rule or threshold
change re-evaluates the entire dataset offline in about 20 seconds with no GPU
(`scripts/reannotate_from_cache.py`) against roughly five minutes for a full
perception run, which is what made the audit-driven rule repairs of Chapter 4
affordable and let the ablation battery run as a sweep rather than a series of
overnight jobs.

**Test strategy.** The suite is 66 tests and runs in about a second, which is
deliberate: a test suite slow enough to skip does not constrain anything. It
has three layers. Worked examples from the predicate specification are
encoded as unit tests over the rule layer, so specification and code cannot
drift apart silently. A randomised invariant test fuzzes two thousand
synthetic scenes and asserts the structural guarantees §3.6 promises, which
hold by construction and would otherwise be easy for a later rule edit to
break quietly. The rest cover what fails for uninteresting reasons: format
writers, the detector adapters of §3.9, frame selection, and the parsers for
externally-produced replies.

**Environment.** The environment is pinned to Python 3.11 with CUDA torch
2.5.1, and the one genuinely awkward installation step is documented rather
than left to be rediscovered: installing SAM2 can silently replace the CUDA
build of torch with a CPU wheel, so the pipeline still runs, produces
identical labels, and takes an order of magnitude longer, which is the worst
class of failure because nothing reports it. A container definition applies
the same fix in the build, and a smoke test verifies on first setup that
both perception models load, that CUDA is actually being used, and that peak
memory sits inside the 6 GB budget of requirement 3. Appendix B gives the
full walk-through, and the repository is public.

## 3.12 Summary of design decisions

Every decision above shares one shape: an alternative was available, it was
rejected for a stated reason, and where the reason is a measurement rather
than a judgement the measurement is named. Four were settled by evidence
that arrived after the decision and could have overturned it. The Small
depth model was chosen for the 6 GB budget and its licence, and ablation A8
then found the Base model gives identical accuracy, so the limit is monocular
ambiguity rather than model capacity. The `near` metric was chosen as a
relative box gap over a 3D centroid distance because every centroid variant
transferred to held-out annotators at F1 no better than 0.024. Masks were
kept over box-only geometry because single-mask mode returned empty masks on
loose boxes, and box-only survives as an ablation that turns out to match the
full pipeline on every box-computable predicate. And the ground-plane
fallback was admitted only once it could be guarded by the tool's own contact
evidence. Appendix F.3 tabulates all eleven decisions with their rejected
alternatives.
