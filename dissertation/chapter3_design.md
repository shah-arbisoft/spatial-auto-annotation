# Chapter 3: Research Methodology and Design

This chapter sets out how the project was conducted and why the annotator is
built the way it is: the methodology (§3.1), the problem and its constraints
(§3.2), the design and the seven predicate rules that follow from it
(§3.3–§3.11), and every decision collected in one table (§3.12).

## 3.1 Research methodology

Three process models dominate data-science practice. **KDD** (Fayyad,
Piatetsky-Shapiro and Smyth, 1996) frames the work as a nine-step pipeline
for discovering patterns in existing databases, and **SEMMA** compresses
that into five tool-centred stages; **CRISP-DM** (Wirth and Hipp, 2000) adds
what both lack, an explicit *Business Understanding* phase before any data
is touched and a cycle in which evaluation feeds back into earlier phases,
and Azevedo and Santos (2008) find it effectively a superset of SEMMA with
stronger process guidance than KDD.

CRISP-DM was chosen for two concrete reasons. The motivating problem is an
engineering question posed by a dataset's own authors, so a phase pinning
the business problem before modelling is structurally necessary. Decisively,
the project's course followed CRISP-DM's evaluation-to-modelling loop in a
way the linear models do not describe: the audit of support-rule precision
sent the work back to modelling twice, a depth gate then a mask-contact
test, each pass re-validated on held-out annotators. §1.3 maps every phase
to a part of this dissertation, including the two findings that came out of
Data Understanding, and summarises the ethical considerations detailed in
Appendix A.

## 3.2 Problem analysis

The task is to produce, from a raw robot-acquired RGB image, the same
artefact a human annotator produces with SGDET-Annotate: a scene graph over
the six object classes with the seven spatial predicates, exported in Visual
Genome JSON, YOLO txt and h5. Four requirements shape the design:

1. **No human decides any label.** A rule may abstain and flag, but it may
   not ask.
2. **Comparability with the human labels.** RQ1 requires the outputs to
   align with the human annotations pair-for-pair and byte-for-byte in
   format, or no fair comparison is possible.
3. **Modest hardware.** A single 6 GB RTX 2060; anything heavier must be
   optional. (In practice the pipeline peaks at 0.65 GB and annotates the
   full dataset in about five minutes *(measured)*.)
4. **Reproducibility.** Every threshold in one config, every run seeded, and
   rule changes re-evaluable without re-running perception.

Three properties of the dataset, established in Chapter 2 and verified
directly on the released files, drive specific design responses:

- **Monocular RGB only.** No metric depth exists and estimated depth is
  *relative and per-image* (Yang, L. et al., 2024), so all depth comparisons
  are ordinal and within-image and no rule consumes absolute depth.
- **Sparse annotation.** Humans labelled a minority of object pairs
  *(measured: 6,458 of 42,440 unordered pairs, 15%; 8,790 of 84,880 ordered,
  10%, the two differing because most labelled pairs carry only one
  direction, §4.5)*, so the tool labels every pair, density being the value
  added, and the protocol treats human labels as a recall target, not an
  exhaustive gold standard.
- **Inconsistent `near`.** Only 3 of 9 annotator groups ever used `near`
  (461/160/93 labels of the 717 total, one further group supplying 3 and the
  rest none), each labelling a different fraction of equally-close pairs, so
  the threshold is fitted and annotator-aware (§3.8), not matched to a
  consensus that does not exist.

## 3.3 Design principle: compute, don't predict

Learned scene-graph generators (REACT++ and predecessors; Neau and Falomir,
2026) *predict* relations from visual patterns, so they need labelled
training data and sit downstream of annotation. This pipeline *computes*
relations from measured geometry with deterministic rules, so it runs before
any learned model and supplies what those models consume. That works only
because the seven predicates are spatial: each is decidable from positions,
extents and depth order, with one exception the design concedes rather than
hides. Support turns on whether a thing is resting or held, which geometry
cannot represent, so a class guard stands in for it (§3.6); ablation A10
tests whether geometry can take that job back and finds it cannot (Appendix
D.8). SAM2 and Depth Anything only *measure* where things are; no learned
component decides a relationship.

The alternative was to keep the human labels and stretch them, by
semi-supervised pseudo-labelling or active learning. Section 2.4 sets out
that rival and the three properties of this dataset that argue against it:
the seed is internally inconsistent, it is not a random sample of what it
would extend, and it is weakest exactly where the task is hardest, the
teacher such a loop would start from reaching 0.08–0.25 recall on the
sparsely-labelled predicates. A geometric labelling function inherits none
of those, because it does not derive from the seed at all: it is fitted on a
handful of thresholds, validated on held-out annotators, and exactly as
consistent on the unlabelled 90% as on the labelled 10%. None of this is
left as argument — §5.3 runs the self-training loop as a third arm of the
same controlled experiment, so the two routes meet on the humans' own
held-out annotations.

**Evaluation setting.** The relation stage is evaluated with ground-truth
boxes and classes (the SGG literature's *PredCls* setting, §2.7). This
isolates the contribution — the paper already establishes detection at 0.93
mAP@50 with YOLOv10m — and makes object indices line up one-to-one with the
human relationship records, so no fragile box-matching stage sits inside the
RQ1 comparison. Detector-in-the-loop operation (YOLOv10m vs. Grounding DINO)
is retained as an ablation and as the deployment mode for new images.

## 3.4 Pipeline architecture

```
image -- boxes+classes --> SAM2 masks --> depth map --> per-object geometry
                                                            |
        writers (VG JSON / YOLO / h5) <-- flags <-- correction <-- 7 rules
```

| Stage | Choice | Rejected alternative | Justification |
|---|---|---|---|
| Boxes | ground truth (study); YOLOv10m / Grounding DINO (deployment) | detector inside RQ1 | isolate the relation stage (§3.3) |
| Masks | SAM2 (Ravi et al., 2024), box-prompted, `sam2.1-hiera-small`; multimask + best score | boxes only | mask centroids and masked depth are robust to box slack; single-mask mode returned empty masks on loose boxes *(measured)*; box-only kept as an ablation |
| Depth | Depth Anything v2 **Small** (Yang, L. et al., 2024), HF pipeline | Base/Large; stereo/metric methods | 6 GB budget; Apache-2.0 (Base/Large are non-commercial); no metric depth exists for this data |
| Lift | centroid (x, y) + **median** depth over the mask | mean depth; full 3D reconstruction | median resists edge bleed where masks overlap background; reconstruction is unnecessary for ordinal tests |
| Fallback | empty mask → box region | image centre / drop object | a failed segmentation must not move the object; regression-tested |
| Rules | explicit thresholds, one function per predicate | learned relation head | the graded contribution; auditable and fittable |
| Correction | reject impossible label sets, demote to flags | emit everything | geometric consistency is checkable for free; adapted from Open3D-VQA's correction flow (Zhang et al., 2025) |
| Confidence | flag ambiguity bands for optional review | silent guesses | human-in-the-loop accelerator claim needs an explicit abstention mechanism |
| Writers | byte-compatible VG JSON / YOLO / h5 | own schema + converter | drop-in comparability (requirement 2); verified against real exports |

Coordinates are normalised by image size so thresholds transfer across
resolutions, and depth is inverted to "smaller is nearer" and min-max
normalised per image; the sign was fixed after front/behind agreement rose
from ~26% to ~74%. An EXIF-aware loader makes the 180°-rotated captures
upright before any box, mask or depth is read.

## 3.5 The seven rules

The complete specification is **Appendix C**: every rule with its
thresholds, the shipped values, the evidence behind each one, and the
correction and flagging policy. It is also maintained in the repository as
`docs/predicate_spec.md`, the copy the code and tests are checked against.
In brief:

- **on / under** encode *support*: subject above object, near-touching
  (vertical gap within ±0.05), horizontal extents overlapping (≥0.20 of the
  narrower), which keeps "floating in front of" from reading as "on".
  `under` is the strict inverse, so the pair can never contradict.
- **to the left of / to the right of** compare horizontal centres in the
  camera frame — the frame the annotators saw on screen, the faithful choice
  among the three RoboSpatial distinguishes (Song et al., 2025) — with an
  ambiguity band (0.02) that abstains when centres nearly coincide.
- **in front of / behind** is a two-stage cascade. Depth ordering decides
  first, with an abstention band (0.03), these being the hardest predicates
  because objects on the same surface often differ by less than the depth
  model can resolve. Where depth abstains a **ground-plane fallback**
  decides from pure projection, ordering two objects on the same floor by
  which box bottom sits lower in the image — a pixel-precise cue exactly
  where depth is noisiest. It is guarded by the tool's own support evidence,
  never firing when either object rests on another (mask contact ≥ 0.60),
  and has its own band (0.005). Pairs both stages abstain on are flagged,
  not guessed (recall 0.70/0.71 *(measured)*, from 0.52/0.55 depth-only).
- **near** is a size-relative proximity test: edge-to-edge box gap divided
  by mean object size, below a fitted threshold, and **never on contact
  pairs**, since `near` co-occurs with on/under on 0 of 469 human pairs.
  {{fig:near-T-sweep}} shows the sweep the threshold is read off. A
  3D-centroid metric was rejected on evidence: per-image depth normalisation
  makes centroid distances incomparable across scenes, and every centroid
  variant transferred to held-out annotators at F1 ≤ 0.024 against the
  relative gap's recall 1.0 (§3.8).

## 3.6 Correction and confidence

Three predicate families are mutually exclusive: on/under, left/right and
front/behind. Two of them cannot contradict, because the rule branches, so
exclusivity is a property of the control flow. Support is different: `on`
and `under` are independent tests over *different* contact evidence, so
noise in either can make both fire on the same pair. That case is demoted to
an `on_under_conflict` flag and neither label is emitted. Demoting instead
of resolving is the deliberate part: picking the stronger of two
contradictory signals would produce a label the evidence does not support
while looking exactly like one it does, and an annotator that fabricates
under uncertainty cannot be audited. Emitting everything for the consumer to
sort out (§3.4) was rejected because RQ2's consumer is a model, which has no
way to sort it out.

One further correction is class-aware, not geometric. Support is not
evaluated at all when either object is a person: the annotators never
recorded one, on **0 of 2,466 gold support triplets**, and mask contact
cannot distinguish an object *resting on* someone from one being *held* by
them. A rule that cannot represent the distinction its evidence turns on
should decline the pair instead of guessing; the guard is a configuration
entry (`no_support_classes`), not a special case buried in code. It is still
a class list standing in for geometry, and it would not cover a manipulator
or an animal holding something; ablation A10 tests whether contact height
can replace it and finds it cannot, at a cost of half the support recall
(Appendix D.8).

Ambiguity flags, four kinds, accompany the triplets: lateral tie, depth tie,
near-threshold edge, and the resolved contradiction above. They are the
design's honesty mechanism: the tool is offered as a human-in-the-loop
accelerator with a *measurable* residual cost, not as an oracle, and about a
third of ordered pairs carry a flag, which §4.7 decomposes into silent
abstention and a much smaller genuine review queue. The structural
guarantees this section promises are asserted in a randomised invariant test
over two thousand synthetic scenes (§3.11), because they hold by
construction and would otherwise be easy for a later rule edit to break
quietly.

## 3.7 Output compatibility

Byte-compatibility is a requirement of the research design: RQ2 compares two
label sources by training the same model on each, and if the automatic
labels arrived in a different container, every downstream difference would
confound the labels with the loader. Three formats are written for three
consumers: Visual Genome JSON, which a replication would diff against; YOLO
txt, which trains the detector the deployment mode and the benchmark share;
and the h5 layout Chapter 6's framework ingests. The writers reproduce the
SGDET-Annotate structure exactly — centre-form `boxes_1024`/`boxes_512`,
index-aligned `labels` and `attribute` arrays, `relationships` as
subject–object index pairs with a parallel `predicates` ID array, the same
six-dataset h5 layout with int64 attributes. The alternative, an internal
schema plus a converter (§3.4), would have left every comparison one
translation away from the thing it claims to measure. Compatibility is
verified by test: a load→write round trip reproduces `boxes_1024` and
`labels` with zero error, and the h5 matches a real export key-for-key and
dtype-for-dtype *(measured)*, both checks in the suite so a later writer
change cannot pass unnoticed.

## 3.8 Calibrating `near`: an annotator-aware protocol

The naive protocol (fit one threshold to all `near` labels, test on held-out
images) fails informatively: fitting on annotator groups 0–5 and testing on
6–8 yields held-out F1 = 0.009, because the label was applied by only three
of nine groups with very different exhaustiveness. The protocol therefore
fits on human-*annotated*, non-contact pairs only, since unannotated pairs
are not reliable negatives under sparse annotation and contact pairs are
never `near` by the measured convention; it uses only the training-split
groups that used the label at all (0 and 4); and it reports agreement on the
held-out near-user, group 8, which contributed nothing to the fit.

Results *(measured)*: fitted **T = 1.372** (gap/mean-size units); held-out
recall **1.000**, every pair the unseen annotator called near lying within
the threshold, with per-annotator precision 0.41 / 0.63 / 0.16 at the same
T. Since recall is 1.0 for all three annotators simultaneously, the human
labels are directionally consistent with a single threshold; what varies (by
~4×) is how exhaustively each applied it. The fitted threshold applies one
definition uniformly, which is exactly the "spatial thresholds for near" the
source paper's future work requests; whether the tool's extra near pairs are
genuinely near is checked by manual audit in the evaluation chapter.

## 3.9 Modularity: the detector as the replaceable part

The claim that the rules are detector-agnostic (§4.11) rests on the
architecture: the rule layer (`src/predicates.py`) imports nothing but
`numpy` and never receives an image, and the entry point takes boxes as an
argument, so no detector is wired in. Holding the boxes fixed, the relation
layer scores the same whether they came from a detector or ground truth, so
detector quality and relation quality are separately attributable. The
contract is explicit (`src/detectors.py`): one method returning pixel boxes,
class names and scores, with three implementations — open-vocabulary
prompting, an adapter for any ultralytics checkpoint including the source
paper's YOLOv10m weights, and a reader for external detections. Twelve unit
tests pin it, one driving the rule layer from a detector written against the
documentation alone. Two coupling points are documented: the support guard
keys on the literal class name `human`, and the fitted thresholds assume
boxes of roughly the tightness the annotators drew, so a detector with
systematically different boxes should re-run §3.8's calibration (twenty
seconds offline from the cache). A worked example is in
`docs/CUSTOM_DETECTOR.md`.

## 3.10 Selecting frames by content

The images this project annotates are consecutive frames of a continuous
robot capture, and the sequence oversamples the scene severely: across its
2,650 frames the mean optical flow between neighbours is 0.08 px, with 0.9%
of pixels moving more than one pixel, so a per-frame pipeline spends a full
perception pass recomputing relations that have not moved. The remedy is to
annotate one frame per *viewpoint*, which requires deciding where one
viewpoint ends and the next begins.

The standard tool does not apply. Shot-boundary detection thresholds the
difference between consecutive frames, which presumes cuts, and a robot
walking through a room produces none: 0.08 px per frame never exceeds the
noise at any single step, while the same motion over forty frames displaces
the image by 13 px, so only accumulated drift carries the signal.
`segment_sequence` (`src/keyframes.py`) therefore measures drift from the
*anchor* of the current segment, not the preceding frame, opening a new
segment when drift exceeds τ, so gradual motion accumulates while a genuine
cut still crosses in one step. Distances are mean absolute differences over
64×48 mean-subtracted greyscale thumbnails, the subtraction discarding the
exposure shifts of an auto-exposing camera. Each segment nominates the frame
closest to its mean signature, which on a moving camera beats taking the
first. A single parameter spans two uses — small τ isolates near-duplicates,
large τ groups several viewpoints of one arrangement, which is what §4.12's
cross-viewpoint measurement consumes — and the threshold is chosen by sweep,
§4.12 reporting what the segmentation recovers and what skipping frames
costs.

## 3.11 Reproducibility by construction

Reproducibility is a design property, because three of the four requirements
in §3.2 are unverifiable without it: a threshold is not fitted on groups 0–5
if nobody else can refit it, and an ablation is an assertion unless the
reader can re-run the arm it removes.

**Configuration and caching.** Every threshold, seed and model identifier
lives in `configs/default.yaml`, and the runner caches each object's lifted
geometry after the single GPU pass. That separates the expensive stage from
the cheap one: any rule or threshold change re-evaluates the whole dataset
offline in about 20 seconds with no GPU (`scripts/reannotate_from_cache.py`)
against roughly five minutes for a full perception run — what made the
audit-driven rule repairs of Chapter 4 affordable and let the ablation
battery run as a sweep.

**Test strategy.** The suite is 66 tests running in about a second,
deliberately: a suite slow enough to skip constrains nothing. Worked
examples from the predicate specification are encoded as unit tests over the
rule layer so the two cannot drift apart silently, a randomised invariant
test fuzzes two thousand synthetic scenes against the structural guarantees
§3.6 promises, and the rest cover the format writers, the detector adapters
of §3.9, frame selection and the reply parsers.

**Environment.** Python 3.11 with CUDA torch 2.5.1, pinned, and the one
genuinely awkward step documented: installing SAM2 can silently replace the
CUDA build of torch with a CPU wheel, so the pipeline still runs, produces
identical labels and takes an order of magnitude longer — the worst class of
failure because nothing reports it. A smoke test verifies on first setup
that both models load, that CUDA is in use, and that peak memory sits inside
the 6 GB budget. Appendix B gives the walk-through, and the repository is
public.

## 3.12 Summary of design decisions

Every decision above shares one shape: an alternative was available and was
rejected for a stated reason. Four were settled by evidence that arrived
*after* the decision and could have overturned it, which is the test of
whether a justification is real: the Small depth model, the relative-gap
`near` metric, masks over box-only geometry, and the ground-plane fallback.
Appendix F.3 tabulates all eleven with the alternative each displaced.

The decisions also carry the design's answer to the four objections §2.9
directs at the method, none added afterwards to fit. Predicates live in
configuration rather than code (§3.9), so the vocabulary extends without
touching the engine; the rules abstain and flag instead of guessing (§3.6),
so an unreliable case becomes a countable review item; randomised invariant
testing over synthetic scenes (§3.11) asserts the structural guarantees
without the author's judgement; and the camera frame is committed to
explicitly (§3.5), so a disagreement is locatable as a convention
difference. Section 7.7 returns to all four with the evidence, which is
where they are settled or conceded. Three are mitigations, not refutations;
the fourth the design could not settle alone, because invariant testing pins
rule *consistency* and says nothing about rule *truth* — that took §4.14's
instrument, built to attack the author's own verdicts, and it overturned
one.
