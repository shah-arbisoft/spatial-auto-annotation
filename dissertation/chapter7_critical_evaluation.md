# Chapter 7: Critical Evaluation

> Every quantitative claim below is established in Chapters 4–6
> and reproducible from the repository; this chapter interprets, connects and
> stress-tests them.

This chapter discusses the results rather than reporting new ones. Section
7.1 evaluates the achievement against the two research questions, §7.2
dissects what the remaining failures are made of, §7.3 examines the
dataset's annotation process in the light of the measurements, §7.4 reflects
on the methodology itself, §7.5 compares the outcome with the prior work of
Chapter 2, and §7.6 states limitations and threats to validity. The
objective-by-objective audit belongs with the conclusions and is given in
§9.1.

## 7.1 Achievement against the research questions

**RQ1** asked whether spatial-relationship annotation can be automated at a
quality comparable to human annotation. The answer is predicate-shaped rather
than singular. For the lateral and proximity predicates the tool is, by every
measure available, *at least* as good as the human process: 0.97–0.998 recall,
audited true precision ≈ 1.0, and for `near` a perfect held-out score against
the one annotator who used the label and never influenced the threshold. For
support, after two evidence upgrades motivated by measurement (depth
co-location, mask contact), recall is 0.88/0.81 with audited precision around
0.9, comfortably comparable. For the depth pair the cascade of relative depth
and the ground-plane fallback reaches 0.64/0.66 pooled (0.84 once the two
inverted-convention groups are aligned), and where the tool commits it agrees
at 0.95–1.00 with six of the seven same-convention annotators, the seventh
being the dataset's smallest sample at 65 triplets; the remaining shortfall
is calibrated abstention plus that inverted direction convention.
"Comparable to human quality" understates that case: on front/behind the tool
is more consistent than the human process it is measured against.

**RQ2** asked whether the automatic labels can train a relation model as well
as human labels. The controlled experiment answered more strongly than the
question was posed: with identical features, model, seed and split, the
auto-trained classifier reaches 0.76 mean recall against held-out *human* gold
versus 0.30 for its human-trained twin. At this dataset's annotation scale,
the automatic labels are better training material than the labels they were
validated against. The mechanism is not mysterious: density (20× more
triplets) and consistency (one definition, uniformly applied). But it converts
the project's premise from "removing the bottleneck loses little" to "removing
the bottleneck gains".

The third arm is what makes that claim hard to dismiss. Self-training on the
human labels, the standard semi-supervised remedy for exactly this problem,
reaches 0.36 and closes only 15% of the gap, and its bookkeeping shows why:
the teacher contributes about a thousand confident *negative* pseudo-labels
for every positive one, propagating the annotators' silence rather than their
judgement, and on `near` it drives recall below the human baseline it started
from. The comparison therefore is not against doing nothing, but against the
obvious alternative, under identical conditions.

The five objectives of §1.2.1 are audited against their evidence in §9.1; this
chapter is concerned with what the results *mean* rather than with whether
each objective was discharged.

## 7.2 What the remaining failures are made of

The failure gallery diagnoses every one of the 1,689 missed human triplets by
re-checking rule conditions, so the failure analysis is exhaustive rather than
anecdotal. Three observations matter most.

First, **genuine tool error is rare**: depth-ordering mistakes are 1–5% of
front/behind misses; across all predicates, misses attributable to avoidable
error are roughly 7%. The bulk of the miss mass is calibrated abstention
(depth ambiguity band, 52–61% of front/behind misses) and measured annotator
defects (38–42%, a share that *grew* as the ground-plane fallback shrank the
abstention share around it).

Second, **the support arc shows the method working as a method**. The box rule
shipped with ~0.27 true precision; the audit localised the failure (projection
adjacency), the gallery localised the misses (containment), one geometric
insight (stacked objects share a camera distance) fixed half the false fires,
and one perception upgrade (mask-bottom contact) fixed most of the rest while
*raising* recall, each step calibrated on train annotators and validated
held-out. The residual failure mode is itself precisely characterised: a
person *holding* an object satisfies pixel contact (3 of the 7 remaining
audited errors); a class-aware guard is the documented next refinement.

Third, **what looked like a depth-resolution ceiling was mostly a rules
ceiling, and it moved**. Two objects at similar camera distance cannot be
ordered by relative monocular depth (the `depth_eps` sweep bounds that trade:
recall up to ~0.71 at ε=0 for ~0.26–0.36 precision), but the ground-plane
fallback showed that most of the abstention band is recoverable *without*
depth, from pure projection, once the tool's own contact evidence guards
against elevated objects. The residual ceiling is narrower and precisely
characterised: objects resting on supports the detector has no box for
(the fallback's audited failure mode), and pairs whose bottom edges tie
within the band. Both operating points are documented, revisable decisions
rather than hidden constants (ablations A2, A7).

The viewpoint-stability test of §4.14 sharpens that third point into
something stronger than the ablations alone could establish. If what remains
of the front/behind gap were depth *noise*, meaning estimates jittering
either side of a decision boundary, then moving the camera would flip
verdicts, because that is precisely the perturbation such noise responds
to. Measured across viewpoints of the same scene, front/behind reproduces
itself 0.955 of the time, above `on` and `under`, and holds at 0.924 under
viewpoint changes
large enough to compress the sequence 89-fold. A predicate that recovers
0.64 of the human labels while agreeing with itself at that rate is not
guessing; it is applying a consistent criterion that the annotators did not
share. Read together with A8, where quadrupling the depth model changed
nothing, and §4.5, where two groups labelled the pair in the opposite
direction outright, the weight of the front/behind shortfall sits on
definitional disagreement rather than on perception. This does not dissolve
monocular ambiguity, which genuinely bounds the predicate for objects at
equal camera distance; it relocates most of the measured gap away from it.
The practical implication is unglamorous and worth stating plainly: the
intervention with the best expected return on this predicate is a written
annotation guideline, not a better depth network.

## 7.3 The dataset's annotation process, examined

The source paper flagged `near` as inconsistent and called for "clear
annotation guidelines (e.g., spatial thresholds for 'near')". This project
quantifies how much further the guideline problem goes:

1. `near` was used by 3 of 9 annotator groups, with ~4× variance in how
   exhaustively equally-close pairs were labelled. Yet all three annotators'
   labels sit inside one fitted threshold (held-out recall 1.0): consistent
   *notion*, non-exhaustive *application*.
2. Two annotator groups applied the **inverted direction convention** for
   in front of / behind (2–5% agreement where the tool commits; flipping
   recovers 0.93/0.71).
3. Support pairs were often labelled in **one direction only** (one group
   all-`on`, another all-`under`).
4. The official guidance, confirmed at the annotation tool's repository,
   consists of **vocabulary lists with no definitions**. Every defect above is
   the predictable consequence of labelling with undefined terms.

This reframes the evaluation itself: for several predicates there is no human
consensus to agree with, only per-annotator behaviours. The dissertation's
response (per-annotator reporting, annotator-aware calibration, and
operational definitions as the deliverable) is, to our knowledge, the first
time this dataset's label semantics have been made explicit.

The "tenth annotator" framing survives contact with the data, and §4.6 puts
numbers on it. Because the tool is deterministic it is the same labeller for
every group, so the 0.216 spread in its agreement across the seven
consistent annotators (0.717 to 0.933) measures the annotators' own
heterogeneity rather than the tool's inconsistency. Applying Fréchet bounds
with the tool as common reference places annotator-to-annotator agreement in
[0.74, 0.92], an interval containing the tool's own 0.869. The claim this
licenses is deliberately modest, because the bounds require assuming the
image batches are exchangeable: the automatic annotator cannot be shown to
agree with the human annotators any less well than they can be shown to
agree with one another. Absent overlapping assignments the quantity cannot
be measured outright, and that absence is itself a finding about the
dataset's construction, one a replication should design away by having two
annotators share a batch from the outset.

## 7.4 Methodological reflection

Choices that proved right: the **PredCls isolation** (without it, every rule
result would be confounded by detection; the SGDet decomposition shows the
relation layer at 0.85 conditional mean, invisible in the 0.38 end-to-end
number); the **sparse-gold protocol** (recall-primary plus restricted
precision plus audits; the audit overturned the naive reading of restricted
precision for five predicates and confirmed it for support); and **train-only
calibration with held-out annotators** (every fitted threshold generalised:
near recall 1.0, support F1 0.87 on annotators the thresholds never saw).

One choice proved its worth only after the fact. The benchmark of Chapter 6
originally ran a single seed per arm, and an early reading of its per-group
table reported that the auto arm wins on the one defect-free test annotator.
Retraining both arms at two further seeds showed the arms tied there, and
the claim was withdrawn (§6.3.1). Two lessons follow. The narrow one is that
a 0.011 margin on 73 images should never have been characterised as a win;
the honest description before replication was "indistinguishable". The
broader one is that the discipline this project applied faithfully to
thresholds, fitting on some annotators and validating on others, was applied
late to *model training variance*, and the fix was cheap: four extra runs on
a free GPU tier. A replication designed from the start would have trained
every arm at three seeds and reported ranges throughout, which is what the
final version does.

Choices a stricter replication should improve: the **audits were verdicted by
the author** (conservatively, with verdicts and rendered evidence published
for spot-checking; the independent validation study of §4.13 is the designed
remedy, and blind external verdicting should have been the instrument from
the first audit rather than the last); the
**support-rule iteration used the same audit machinery twice**, so the second
audit is confirmatory rather than fully independent; the SGDet **prompt/
threshold tuning used one disclosed iteration on a trial slice** that
over-estimated full-set detection quality, a small, instructive example of
trial-set optimism; and 2,000-scene invariant fuzzing pins rule consistency
but not rule *truth*, which only the audits address.

## 7.5 Synthesis against the geometry-to-label lineage

The pipeline borrows its skeleton from the SpatialVLM family (Chen et al.,
2024): lift perception to geometry, derive spatial facts deterministically.
The idea is older still, since CLEVR (Johnson et al., 2017) obtained exact
relations from a renderer; what neither can supply is the recovery of
geometry from real photographs, which is where the difficulty of this
project sits. What it adds is not the skeleton but the parts the lineage
leaves implicit. SpatialVLM and VQASynth (Remyx AI, 2024) generate *training
text* at internet scale and never confront a fixed predicate vocabulary with
human ground truth; SpatialRGPT (Cheng et al., 2024) curates region
representations with depth but validates downstream, not against annotators.
This project's contributions to the recipe are: **annotator-aware
calibration** (fit thresholds only on annotators who used a label; hold out
annotators, not just images), **contact as the support signature**
(mask-bottom adjacency, which the box-geometry lineage does not use and
which repaired both error directions at once, and which parallels the
argument for pixel-accurate grounding made by panoptic scene-graph
generation (Yang, J. et al., 2022)), and **loss attribution as methodology**
(every miss diagnosed to a cause; every gap decomposed into abstention vs
annotator vs error, detection vs relations). RoboSpatial's reference-frame
taxonomy (Song et al., 2025), cited in the design chapter to justify
camera-frame laterality, turned out to be the right lens for a *measured*
phenomenon: the front/behind convention inversion is a reference-frame
disagreement inside a single dataset's annotation team, and therefore an
instance of the frame-dependence of spatial language that Landau and
Jackendoff (1993) describe.

Two results connect this project to literatures outside its immediate
lineage. The benchmark finding of Chapter 6 is a specific case of the general
problem Northcutt, Athalye and Mueller (2021) demonstrated across ten
standard benchmarks, that errors in test annotation change model rankings and
can therefore select the wrong model; here the defect is not random label
noise but systematic annotator convention, and the ranking it distorts is
between two label *sources* rather than two architectures. The
diagnosis also matches the motivation for SpatialSense (Yang, K.,
Russakovsky and Deng, 2019) and Rel3D (Goyal et al., 2020), both of which
were built after their authors found that relation benchmarks could be
scored well without using spatial information; this dissertation observes the
mirror image, a model scoring well by reproducing annotator selection habits.
The RQ2 result, meanwhile, is the weak-supervision prediction of Ratner et
al. (2017) confirmed in a domain the original work did not address, with the
addition that the standard semi-supervised alternative (Lee, 2013) was
implemented and measured rather than argued away.

## 7.6 Limitations and threats to validity

**Internal.** Thresholds are fitted to six annotator groups of one dataset;
audits are author-verdicted (§7.4); the two-stage audit shares machinery with
the rule change it evaluates.

**External.** One laboratory domain, six object classes, one camera and
mounting; the fitted constants (`near_T`, ε values, contact threshold) are
dataset-specific by design. The *procedure* (fit on some annotators, validate
on held-out ones) is the transferable artefact, not the numbers. The one
piece of out-of-domain evidence is qualitative: the video clips of §4.12 run
the unretuned thresholds over different scenes, a different viewpoint and
objects almost entirely outside the six classes, and the support and lateral
relations behave correctly there, with the visible failures attributable to
detection rather than to the rules. That supports transfer of the rule layer
but measures nothing, since no labelled out-of-domain gold exists; a modest
labelled cross-domain sample is the missing experiment, and it is cheap
enough that a replication should simply include one. Full
automation is currently detection-bounded (0.38 end-to-end with a worst-case
zero-shot detector; the authors' trained detector would close most of that
gap, but this remains unverified without their weights).

**Construct.** "Agreement with human labels" is an imperfect proxy when the
humans disagree with each other; the per-annotator decompositions mitigate but
cannot eliminate this. The RQ2 result compares supervision *at this dataset's
annotation scale*. It does not claim automatic labels beat abundant,
guideline-driven human annotation, a regime this dataset does not contain.

**Ethics.** Scene images contain identifiable people; figures for publication
use the dataset as released (CC-BY 4.0) with faces blurred as a courtesy.

## 7.7 Aims, revisited

Both research questions are answered with evidence that survived held-out
validation, independent audits and exhaustive failure diagnosis. The
unexpected result is the strongest: the project set out to show automatic
labels are *not much worse* than human ones, and found conditions (sparse,
guideline-free human annotation, which is exactly the regime the source
dataset occupies) under which they are decisively better for downstream
learning. The scaling claim now has teeth beyond throughput: 836 images were
labelled in five minutes with 20× the human label density, and a model trained
on those labels reaches two and a half times the downstream recall of one
trained on the original annotations. The bottleneck this project removes was
not only slowing the dataset down; it was limiting what the dataset could
teach.
