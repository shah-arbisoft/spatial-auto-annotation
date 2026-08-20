# Chapter 7: Critical Evaluation

This chapter discusses the results instead of reporting new ones: what the
project achieved against its two research questions, what the remaining
failures are made of, what the measurements say about the dataset's own
annotation process, and, in §7.7, an answer to each of the five objections
§2.9 raised against the approach before any result was known. The
objective-by-objective audit belongs with the conclusions and is in §9.1.

## 7.1 Achievement against the research questions

**RQ1** asked whether spatial-relationship annotation can be automated at a
quality comparable to human annotation. The answer is predicate-shaped rather
than singular. For the lateral and proximity predicates the tool is, by every
measure available, *at least* as good as the human process: 0.97–0.998 recall,
blind-audited precision 0.92–1.00, and for `near` a perfect held-out score
against the one annotator who used the label and never influenced the
threshold. For
support, after two evidence upgrades motivated by measurement (depth
co-location, mask contact), recall is 0.81/0.75, but the claim stops there:
blind-audited precision is 0.535 [0.42, 0.65] after the threshold repair of
§4.14, up from 0.404 before it, so the labels the tool adds beyond the human
record are trustworthy for five predicates and right about half the time for
this one. For the depth pair the cascade of relative depth
and the ground-plane fallback reaches 0.70/0.71 pooled (0.84 once the two
inverted-convention groups are aligned), and where the tool commits it agrees
at 0.95–1.00 with six of the seven same-convention annotators, the seventh
being the dataset's smallest sample at 65 triplets; the remaining shortfall
is calibrated abstention plus that inverted direction convention.
"Comparable to human quality" understates that case: on front/behind the tool
is more consistent than the human process it is measured against.

**RQ2** asked whether the automatic labels can train a relation model as well
as human labels. The controlled experiment answered more strongly than the
question was posed: with identical features, model, seed and split, the
auto-trained classifier reaches 0.75 mean recall against held-out *human* gold
versus 0.30 for its human-trained twin. At this dataset's annotation scale,
the automatic labels are better training material than the labels they were
validated against. The mechanism is not mysterious: density (20× more
triplets) and consistency (one definition, uniformly applied). But it converts
the project's premise from "removing the bottleneck loses little" to "removing
the bottleneck gains".

That claim is scoped by the third measurement rather than confirmed by it.
Chapter 6 puts the same two label sources through the source paper's own
framework and cannot separate them, 0.292 against 0.293 over three seeds, so
the advantage the classifier measures does not appear on a ranked metric
scored against sparse human annotation. Both results are about the same
labels; what differs is what each yardstick asks for, and §6.4 spends the
chapter on that difference. The defensible sentence is that automatic labels
*teach better where correctness is the criterion and equally well where
resemblance to the annotators is*, not that they are simply better.

The third arm is what makes that claim hard to dismiss. Self-training on the
human labels, the standard semi-supervised remedy for exactly this problem,
reaches 0.36 and closes only 15% of the gap. Its bookkeeping shows why: the
teacher contributes about a thousand confident *negative* pseudo-labels for
every positive one, propagating the annotators' silence and not their
judgement, and on `near` it drives recall below the human baseline it started
from. The comparison therefore is not against doing nothing, but against the
obvious alternative, under identical conditions.

The six objectives of §1.2.1 are audited against their evidence in §9.1; this
chapter is concerned with what the results *mean* rather than with whether
each objective was discharged.

## 7.2 What the remaining failures are made of

The failure gallery diagnoses every one of the 1,689 missed human triplets by
re-checking rule conditions, so the failure analysis is exhaustive and not
anecdotal. Three observations matter most.

First, **genuine tool error is rare**: depth-ordering mistakes are 1–5% of
front/behind misses; across all predicates, misses attributable to avoidable
error are roughly 7%. The bulk of the miss mass is calibrated abstention
(depth ambiguity band, 52–61% of front/behind misses) and measured annotator
defects (38–42%, a share that *grew* as the ground-plane fallback shrank the
abstention share around it).

Second, **the support arc shows the method working as a method**, and it
ran one step further than the chapter originally recorded: the threshold that
survived those repairs was itself re-fitted after §4.14 found it over-emitting,
and the shipped value is now 0.85. The box rule shipped at 0.13 true precision; the audit localised the failure
(projection adjacency), the gallery localised the misses (containment), one
geometric insight (stacked objects share a camera distance) fixed half the
false fires, and one perception upgrade (mask-bottom contact) fixed most of
the rest while *raising* recall, each step calibrated on train annotators and
validated held-out. The residual failure mode is precisely characterised too:
a person *holding* an object satisfies pixel contact (3 of the 7 remaining
audited errors), and a class-aware guard is the documented next refinement.

Third, **what looked like a depth-resolution ceiling was mostly a rules
ceiling, and it moved**. Two objects at similar camera distance cannot be
ordered by relative monocular depth (the `depth_eps` sweep bounds the trade:
recall up to ~0.71 at ε=0 for ~0.26–0.36 precision), but the ground-plane
fallback recovers most of the abstention band *without* depth, from pure
projection, once the tool's own contact evidence guards against elevated
objects. What is left is narrower and equally well characterised: objects
resting on supports the detector has no box for, and pairs whose bottom edges
tie within the band. Both operating points are documented, revisable
decisions and not hidden constants (ablations A2, A7).

Section 4.12 settles what the ablations could not. If the front/behind gap
were depth *noise*, estimates jittering either side of a boundary, moving the
camera would flip verdicts, since that is the perturbation such noise responds
to. It does not: the predicate reproduces itself 0.955 of the time, above `on`
and `under`, and holds at 0.924 at 89-fold compression. A predicate recovering
0.64 of the human labels while agreeing with itself at that rate is not
guessing; it is applying a consistent criterion the annotators did not share.
With A8, where quadrupling the depth model changed nothing, and §4.5, where
two groups labelled the pair oppositely, the weight of the shortfall sits on
definitional disagreement rather than perception. That does not dissolve
monocular ambiguity, which genuinely bounds the predicate at equal camera
distance, but it relocates most of the measured gap away from it. The
implication is unglamorous: the best expected return here is a written
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
4. The official guidance, confirmed at the annotation tool's repository, is
   **vocabulary lists with no definitions** (§4.7), which is what makes
   every defect above predictable, not surprising.

This reframes the evaluation itself: for several predicates there is no human
consensus to agree with, only per-annotator behaviours. The dissertation's
response, per-annotator reporting, annotator-aware calibration and operational
definitions as the deliverable, is to our knowledge the first time this
dataset's label semantics have been made explicit.

The "tenth annotator" framing survives contact with the data, and §4.6 puts
numbers on it. The tool is deterministic, the same labeller for every group,
so the 0.216 spread in its agreement across the seven consistent annotators
(0.717 to 0.933) measures their heterogeneity, not its inconsistency. Fréchet
bounds with the tool as common reference place annotator-to-annotator
agreement in [0.74, 0.92], containing the tool's own 0.869. The claim is
deliberately modest, since the bounds assume the batches are exchangeable: the
automatic annotator cannot be shown to agree with the humans any less well
than they can be shown to agree with each other. Without overlapping
assignments the quantity cannot be measured outright, and that absence is
itself a finding about the dataset's construction, one a replication should
design away by having two annotators share a batch.

## 7.4 Methodological reflection

Three choices proved right. The **PredCls isolation**: without it every rule
result would be confounded by detection, and the SGDet decomposition shows the
relation layer at 0.85 conditional mean, invisible inside the 0.38 end-to-end
number. The **sparse-gold protocol**, recall-primary plus restricted precision
plus audits: the audit overturned the naive reading of restricted precision
for five predicates, and for support it took a third, blinded audit to
establish that the second had been confirming, not testing.
**Train-only calibration with held-out annotators**: every fitted threshold
generalised, with `near` recall 1.0 and support F1 0.87 on annotators the
thresholds never saw.

One choice proved its worth only afterwards. The withdrawn single-seed claim
of §6.3.1 shows the discipline applied faithfully to thresholds, fitting on
some annotators and validating on others, reaching *model training variance*
late. The fix was cheap, four extra runs on a free GPU tier, and a replication
designed from the start would have trained every arm at three seeds and
reported ranges throughout, which is what the final version does.

Choices a stricter replication should improve, the first of which is no longer
a suspicion but a measurement. The **audits were verdicted by the author**,
and §4.14 shows what that cost: the same rules on the same data score 0.77
when the auditor knows every item is a tool assertion and 0.404 when decoys
are mixed in unmarked. The **support-rule iteration used the same audit
machinery twice**, making the second audit confirmatory rather than
independent, and it confirmed a figure that a blind instrument does not
support. Blind verdicting with decoys should have been the instrument from
the first audit, not the third; it is cheap, and it is the only step
here that changed a headline number instead of tightening one.

One finding belongs here rather than in Chapter 6, because it is about method
and not about labels. The benchmark arms were originally trained weeks apart,
against whatever state the upstream framework happened to be in on the day;
retraining all nine runs in a single session on one clone moved the
human arm's pooled mR@100 from 0.326 to 0.293, while the vision-language arm
moved 0.001 and neither arm's labels had changed. **A third of the margin this
dissertation once reported between label sources was an artefact of *when*
each arm was trained.** Nothing in the original protocol was careless: seeds
were fixed, the detector frozen, the configuration shared. The confound
entered through an unpinned dependency, which is the one axis the protocol did
not name. A controlled comparison has to control the code as explicitly as it
controls the data, by pinning versions and training every arm in one session;
§6.3 now does, and its figures supersede the earlier ones.

That weakness was acted on rather than only recorded, and the sequence is the
part worth carrying forward. Section 4.14 traced the support result to a
threshold fitted on train F1 against gold covering a tenth of pairs, where a
false positive outside the gold was free. The threshold was re-fitted on the
audited claims from the training annotators, shipped, every experiment re-run
against the new labels, and a second pack drawn from those labels and audited
blind. Support precision moved from a measured 0.404 to a measured 0.535, and
the decoy control moved with it: 27 of 28 rejected against 19 of 28 first
time. The held-out fit had predicted 0.667, so the loop also priced its own
extrapolation at about 0.13. The lesson is not the number but the order — a
threshold worth changing is worth re-auditing on labels it produced, because
the estimate that justifies a change is not evidence for it. The
SGDet **threshold tuning used one
disclosed iteration on a trial slice** that over-estimated full-set detection
quality, an instructive case of trial-set optimism; and invariant fuzzing pins
rule consistency but not rule *truth*, which only the audits address.

## 7.5 Synthesis against the geometry-to-label lineage

The pipeline borrows its skeleton from the SpatialVLM family (Chen et al.,
2024) and, further back, from CLEVR (Johnson et al., 2017); neither supplies
the recovery of geometry from real photographs, where this project's
difficulty sits. What it adds is not the skeleton but the parts the lineage
leaves implicit, because SpatialVLM and VQASynth (Remyx AI, 2024) generate
*training text* at internet scale without ever confronting a fixed predicate
vocabulary with human ground truth, and SpatialRGPT (Cheng et al., 2024)
curates region representations with depth but validates downstream rather
than against annotators. This project contributes three. **Annotator-aware
calibration**: fit only on annotators who used a label, and hold out
annotators and not merely images. **Contact as the support signature**:
mask-bottom adjacency, unused by the box-geometry lineage, which repaired both
error directions at once and parallels the argument for pixel-accurate
grounding in panoptic scene-graph generation (Yang, J. et al., 2022). And
**loss attribution as methodology**: every miss diagnosed, every gap
decomposed into abstention against annotator against error, and detection
against relations. RoboSpatial's reference-frame taxonomy (Song et al.,
2025), cited in Chapter 3 to justify camera-frame laterality, proved the
right lens for a *measured* phenomenon: the front/behind inversion is a
reference-frame disagreement inside one annotation team, an instance of the
frame-dependence Landau and Jackendoff (1993) describe.

Two results connect this project to literatures outside its lineage. Chapter
6's benchmark finding is a case of the problem Northcutt, Athalye and Mueller
(2021) demonstrated across ten benchmarks, that errors in test annotation
change model rankings and can select the wrong model; here the defect is
systematic annotator convention rather than random noise, and it distorts a
ranking between two label *sources*, not two architectures. The
diagnosis matches the motivation for SpatialSense (Yang, K., Russakovsky and
Deng, 2019) and Rel3D (Goyal et al., 2020), both built after their authors
found relation benchmarks could be scored well without using spatial
information; this dissertation observes the mirror image, a model scoring
well by reproducing annotator selection habits. The RQ2 result is Ratner et
al.'s (2017) weak-supervision prediction confirmed in a domain the original
work did not address, with the standard semi-supervised alternative (Lee,
2013) implemented and measured and not argued away.

## 7.6 Limitations and threats to validity

**Internal.** Thresholds are fitted to six annotator groups of one dataset;
audits are author-verdicted (§7.4); the two-stage audit shares machinery with
the rule change it evaluates.

**External.** One laboratory domain, six object classes, one camera and
mounting; the fitted constants (`near_T`, ε values, contact threshold) are
dataset-specific by design, and the transferable artefact is the *procedure*,
fit on some annotators and validate on held-out ones, not the numbers. The
only out-of-domain evidence is qualitative (Appendix E.4) and measures nothing,
since no labelled out-of-domain gold exists; a modest labelled cross-domain
sample is the missing experiment, cheap enough that a replication should
simply include one. Full automation is currently detection-bounded (0.38
end-to-end with a worst-case zero-shot detector; the authors' trained
detector would close most of that gap, unverified without their weights).

**Construct.** "Agreement with human labels" is an imperfect proxy when the
humans disagree with each other; the per-annotator decompositions mitigate but
cannot eliminate this. The RQ2 result compares supervision *at this dataset's
annotation scale*. It does not claim automatic labels beat abundant,
guideline-driven human annotation, a regime this dataset does not contain.

**Ethics.** Scene images contain identifiable people; figures for publication
use the dataset as released (CC-BY 4.0) with faces blurred as a courtesy.

## 7.7 The objections of §2.9, answered

The literature review stated five objections to a rule-based annotator before
any result was reported, so that this chapter could be read as an attempt on
them. Each is answered here with the evidence that bears on it, including the
two the evidence does not settle.

**Vocabulary scale: not answered, and conceded.** Nothing in this
dissertation bears on predicates beyond the seven, and the objection is that
a hand-written rule set does not extend by learning. It stands in full. What
the work can offer is a boundary rather than a rebuttal: the rules are
decidable because the predicates are spatial, and §3.3 makes that
dependence explicit instead of hoping it generalises. Section 9.3 records
it as the limitation most likely to matter to anyone reusing the method.

**Systematic error: partly confirmed, partly refuted, and the split is
informative.** The objection predicts that a downstream model absorbs the
rule's blind spot as though it were fact. Chapter 6 is that prediction
coming true and is reported as such: the auto-trained arm ranks below the
human-trained one on the metric the field uses, and the gap is concentrated
where the annotation itself is defective. But the objection also predicts
that consistent-but-wrong supervision is worse than inconsistent human
supervision, and on this dataset that is refuted at every level where the
question was asked, by a factor of two and a half in the controlled
experiment and by a planner that clears the occluder in 22 of 25 scenes
against 0 with no relations at all. The reconciliation is that systematic
error is worse than random error only when it is *wrong*; §4.12's finding
that front/behind agrees with itself across viewpoints 0.955 of the time
while agreeing with the annotators 0.64 of the time is the shape of a
consistent rule meeting a different convention, not of a consistent mistake.

**Circular validation: conceded.** Section 7.4 already lists this first
among the choices a stricter replication should improve, and no result in
Chapters 4 to 6 removes it. What belongs here is the extent of the
concession, not a restatement of it. The circularity bounds the
*precision* estimates, because those rest on verdicts the author gave. It
does not reach the structural guarantees of §3.6, which are checkable
without any verdict at all, nor the downstream findings of Chapters 5 and 6,
which are scored against human annotation the author did not produce and in
which the automatic arm is judged by its rival's yardstick. The designed
remedy is the study of Appendix E.3 and it has no results at the time of
writing, which leaves
this the objection a reader should weight most heavily.

**Reference frame: answered as far as it can be, which is not all the way.**
The dissertation does not assert that the camera frame is correct; it shows
that the tool applies one frame consistently (§4.12), that two annotator
groups applied another (§4.5), and that recall rises from 0.70/0.71 to 0.84
once the convention is aligned. That establishes disagreement rather than
error, and identifies which party is consistent. It does not establish which
convention a robot should obey, and no measurement in this dissertation
could, because that is a question about the specification and not about the
data.

**Better collection, not cheaper labels: not attempted, and partly
answered sideways.** The project re-collected nothing, but it produced the
operational definitions the annotation process never had (§3.5) and quantified
three defects those definitions would have prevented, which is what a better
collection round needs. Cheap labels and correct definitions are complements,
and this work supplies the second while delivering the first.

The project set out to show that automatic labels are *not much worse* than
human ones, and found conditions under which they are decisively better, those
conditions being the sparse, guideline-free annotation this dataset occupies.
The bottleneck was therefore not only slowing the dataset down, it was
limiting what the dataset could teach.
