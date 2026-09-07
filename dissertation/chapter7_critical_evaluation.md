# Chapter 7: Critical Evaluation

What follows discusses the results and reports no new ones, covering what the
project achieved against its two research questions, what the remaining
failures are made of, what the measurements say about the dataset's annotation
process, and in §7.7 an answer to each of the five objections §2.9 raised
before any result. Section 7.8 covers what the result means socially and
professionally, and the objective-by-objective audit, which belongs with the
conclusions, is in §8.1.

## 7.1 Achievement against the research questions

**RQ1** asked whether annotation can be automated at a quality comparable to
human annotation, and the answer differs by predicate, with §8.1 stating it
against the criteria of §1.2.2. What matters here is its shape.

For the lateral and proximity predicates the tool is by every measure
available *at least* as good as the human process. For support it is not. Two
evidence upgrades lifted recall. But blind-audited precision stops the claim
at about half the labels the tool adds beyond the human record. For the depth
pair the shortfall is calibrated abstention plus an inverted direction
convention. Neither is error. Where the tool commits it agrees at 0.95–1.00
with six of the seven same-convention annotators. On front/behind the tool is
actually more consistent than the process it is measured against, which
"comparable to human quality" undersells.

**RQ2** asked whether the automatic labels can train a relation model as well
as human labels, and the controlled experiment answered more strongly than the
question was posed. With identical features, model, seed and split, the
auto-trained classifier reaches 0.75 mean recall against held-out *human* gold
versus 0.30 for its human-trained twin. At this dataset's annotation scale,
and in a feature space kin to the rules that wrote them (§5.5), the automatic
labels are better training material than the labels they were validated
against, on density (20× more triplets) and on consistency (one definition,
uniformly applied). So the premise converts from "removing the bottleneck
loses little" to "removing the bottleneck gains".

Chapter 6 scopes that claim without confirming it, since it cannot separate
the two label sources in a current SGG framework, so the advantage does not
appear on a ranked metric scored against sparse human annotation. What can be
defended is that automatic labels *teach better where correctness is the
criterion and equally well where resemblance to the annotators is*, and the
third arm makes the claim hard to dismiss. Self-training on the human labels,
the standard semi-supervised remedy for exactly this problem, reaches 0.36 and
closes only 15% of the gap. Its teacher contributes about a thousand confident
*negative* pseudo-labels for every positive one, propagating the annotators'
silence, not their judgement, and on `near` it drives recall below the human
baseline it started from. So the comparison is against the obvious alternative
under identical conditions, not against doing nothing. (The six objectives of
§1.2.1 are audited against their evidence in §8.1, and this chapter is about
what the results *mean*.)

## 7.2 What the remaining failures are made of

The failure gallery diagnoses every one of the 1,650 missed human triplets by
re-checking rule conditions, so the analysis is exhaustive, not anecdotal.
Three observations come out of it.

First, **genuine tool error is rare**. Section 4.10 puts avoidable error at
roughly 7% of the misses, the bulk being calibrated abstention (43–54% of
front/behind misses sit in the depth ambiguity band) and measured annotator
defects taking most of the rest. What the diagnosis gives is the rule system's
account of its own misses, not an independent adjudication, so the 7% is the
residual its own categories leave, and it still shapes what is left to do. A
tool whose misses are mostly abstention improves by deciding more often, not
better, and one whose misses are mostly the annotators' own conventions does
not improve by changing the tool.

Second, **the support arc shows the method working as a method**, and it ran
one step further than the chapter first recorded, since the threshold that
survived those repairs was re-fitted after §4.14 found it over-emitting, and
ships at 0.85. The box rule shipped at 0.13 true precision, and the audit
localised the failure to projection adjacency while the gallery localised the
misses to containment. One geometric insight, that stacked objects share a
camera distance, fixed half the false fires, and mask-bottom contact fixed
most of the rest while *raising* recall, with each step calibrated on train
annotators and validated held-out. The residual mode is characterised too: a
person *holding* an object satisfies pixel contact (3 of the 7 remaining
audited errors), and a class-aware guard is the documented next refinement.

Third, **what looked like a depth-resolution ceiling was mostly a rules
ceiling, and it moved**. Two objects at similar camera distance cannot be
ordered by relative monocular depth, and the `depth_eps` sweep bounds the
trade: recall up to ~0.71 at ε=0 for ~0.26–0.36 precision. But the
ground-plane fallback recovers most of the abstention band *without* depth,
once the tool's own contact evidence guards against elevated objects. What is
left is narrower and equally well characterised: objects resting on supports
the detector has no box for, and pairs whose bottom edges tie within the band.
Both operating points are documented, revisable decisions, not hidden
constants (ablations A2, A7).

Section 4.12 settles what the ablations could not. If the front/behind gap
were depth *noise*, moving the camera would flip verdicts. Instead the
predicate reproduces itself 0.958 of the time, above `on` and `under`. It
holds at 0.911 at 89-fold compression. Recovering 0.70 of the human labels
while agreeing with itself at that rate is not guessing, but applying a
consistent criterion the annotators did not share. Together with A8, where
quadrupling the depth model changed nothing, and §4.5, where two groups
labelled the pair oppositely, the weight of the shortfall sits on definitional
disagreement, not on perception. None of that dissolves monocular ambiguity,
which genuinely bounds the predicate at equal camera distance, but it moves
most of the measured gap away from it. What would pay best is a written
annotation guideline. That is a duller prescription than a better depth
network.

## 7.3 The dataset's annotation process, examined

The source paper flagged `near` as inconsistent and called for "clear
annotation guidelines (e.g., spatial thresholds for 'near')", and this project
quantifies how much further the guideline problem goes:

1. `near` was supplied in quantity by 3 of 9 annotator groups, a fourth
   contributing three labels, with ~4× variance in how exhaustively
   equally-close pairs were labelled. Yet all three annotators' labels sit
   inside one fitted threshold (held-out recall 1.0): consistent *notion*,
   non-exhaustive *application*.
2. Two annotator groups applied the **inverted direction convention** for in
   front of / behind (2–5% agreement where the tool commits; flipping
   recovers 0.94/0.82).
3. Support pairs were often labelled in **one direction only** (one group
   all-`on`, another all-`under`).
4. **No operational definition of the predicates survives in the released
   materials**: the annotation tool's repository carries vocabulary lists
   only (§4.7).

The fourth point needs stating precisely, because the obvious version of it is
weaker than the evidence. Wang et al. (2025) report that annotators were
trained on the tool and given predicate definitions. What is claimed here is
not that nothing was said, but that whatever was said did not survive contact
with nine annotators, since the three defects above are measured in the labels
themselves, not inferred from an absence. Prose definitions given at training
time were not enough, which argues for definitions of a different kind, not
for more of the same. No threshold a program applies can be applied two ways
by two annotators, and that is what Chapter 3's specification supplies, which
is what makes it a contribution, not documentation. So the evaluation itself
is reframed. For several predicates there is no human consensus to agree with,
only per-annotator behaviours, and the response is per-annotator reporting,
annotator-aware calibration, and operational definitions as the deliverable,
to our knowledge the first time this dataset's label semantics have been made
explicit.

The "tenth annotator" framing survives contact with the data, and §4.6 puts
numbers on it, since the tool is deterministic and the same labeller for every
group, so the 0.082 spread across the seven consistent annotators (0.851 to
0.933) carries no variance of its own. Even so it is an upper bound, not a
measurement, since the batches behind it differ threefold in density, so
annotator and batch cannot be separated, and the Fréchet bound §4.6 attempts
on it is not claimed here, needing exchangeable batches these are not. Without
overlapping assignments the quantity cannot be measured outright, which is
itself a finding about how the dataset was built. Any replication should
design that away by having two annotators share a batch.

## 7.4 Methodological reflection

Some choices proved right. Without the **PredCls isolation** every rule result
would be confounded by detection, and the SGDet decomposition shows the
relation layer at 0.85 conditional mean, which is invisible inside the 0.38
end-to-end number. Under the **sparse-gold protocol**, recall-primary plus
restricted precision plus audits, the audit overturned the naive reading of
restricted precision for five predicates, and for support it took a third,
blinded audit to establish that the second had been confirming, not testing.
With **train-only calibration on held-out annotators**, every fitted threshold
generalised, and `near` recall came out at 1.0 with support F1 0.87 on
annotators the thresholds never saw. One choice proved its worth only
afterwards, since the withdrawn single-seed claim of §6.3.1 shows the
discipline applied faithfully to thresholds while reaching *model training
variance* late. Fixing it was cheap, four extra runs on a free GPU tier, and a
replication designed from the start would have trained every arm at three
seeds and reported ranges throughout, as the final version does.

Then the choices a stricter replication should improve, the first no longer a
suspicion but a measurement. One is that the **audits were verdicted by the
author**, and §4.14 shows what that cost: the same rules on the same data
score 0.77 when the auditor knows every item is a tool assertion, and 0.404
when decoys are mixed in unmarked. The **support-rule iteration used the same
audit machinery twice**, leaving the second audit dependent on the first and
confirming a figure a blind instrument does not support. Blind verdicting with
decoys should have been the instrument from the first audit, not from the
third. Cheap to run, it is also the only step here that changed a headline
number, where the rest only tightened one. That weakness was acted on, not
only recorded, and §4.14 gives the sequence, since the support threshold had
been fitted where a false positive outside the gold was free. Re-fitted on
train annotators, it shipped, every experiment was re-run, and a second pack
was drawn from the new labels and audited blind. What matters is the order,
not the number, because a threshold worth changing has to be re-audited on the
labels it produced, since the estimate that justified the change cannot also
be the evidence for it. Another is that the **bootstrap resamples images** as
though they were independent draws, which §4.12 later shows they are not, the
dataset being one trajectory whose neighbouring frames persist at 0.90 to
0.92, and the unit was chosen before that was known, so the intervals it gives
are narrower than a block scheme over contiguous runs would give
(Supplementary F.8). Point estimates are unaffected. Any replication should
resample segments. Last comes the SGDet **threshold tuning**, which used one
disclosed iteration on a trial slice that over-estimated full-set detection
quality, an instructive case of trial-set optimism. And invariant fuzzing pins
rule consistency but not rule *truth*, which only the audits address. One
finding belongs here, not in Chapter 6, because it is about method: the
benchmark arms were originally trained weeks apart, against whatever state the
upstream framework was in on the day. Retraining all nine runs in one session
on one clone moved the human arm's pooled mR@100 from 0.326 to 0.293, and the
vision-language arm by 0.001, with no label changed. **A third of the margin
this dissertation once reported between label sources was an artefact of when
each arm was trained.** Nothing in the original protocol was careless. Seeds
were fixed, the detector frozen, the configuration shared, and the confound
entered through an unpinned dependency, the one axis the protocol did not
name. What moved was the external benchmark framework in a hosted session, not
the annotator: that runs from a pinned environment and reproduces `pairs.csv`
byte for byte against a committed digest (Supplementary B), which is why the
fidelity numbers were unaffected. A controlled comparison has to control the
code as explicitly as it controls the data, and a hosted run needs the
discipline the local one already had. Section 6.3 now does, and its figures
supersede the earlier ones.

## 7.5 Synthesis against the geometry-to-label work

The pipeline borrows its skeleton from the SpatialVLM family (Chen et al.,
2024) and, further back, from CLEVR (Johnson et al., 2017). Neither supplies
the recovery of geometry from real photographs, which is where this project's
difficulty sits. SpatialVLM and VQASynth (Remyx AI, 2024) generate *training
text* at internet scale without confronting a fixed predicate vocabulary with
human ground truth, and SpatialRGPT (Cheng et al., 2024) curates region
representations with depth but validates downstream.

What that work leaves implicit is where this project's contribution sits.
**Annotator-aware calibration** means fitting only on annotators who used a
label, and holding out annotators, not merely images. **Contact as the support
signature** is mask-bottom adjacency, unused by the box-geometry work, which
repaired both error directions at once and parallels the argument for
pixel-accurate grounding in panoptic scene-graph generation (Yang, J. et al.,
2022). And **loss attribution as methodology** means every miss diagnosed,
every gap decomposed into abstention against annotator against error, and
detection against relations. RoboSpatial's reference-frame taxonomy (Song et
al., 2025), cited in Chapter 3 to justify camera-frame laterality, proved the
right lens for a *measured* phenomenon, since the front/behind inversion is a
reference-frame disagreement inside one annotation team, an instance of the
frame-dependence Landau and Jackendoff (1993) describe.

Two results connect this project to literatures outside its own. Chapter 6's
benchmark finding is a case of the problem Northcutt, Athalye and Mueller
(2021) demonstrated across ten benchmarks, that errors in test annotation
change model rankings and can select the wrong model. Here the defect is
systematic annotator convention where the literature assumes random noise, and
it distorts a ranking between two label *sources*, not two architectures.
Matching that diagnosis is the motivation for SpatialSense (Yang, K.,
Russakovsky and Deng, 2019) and Rel3D (Goyal et al., 2020), both built after
their authors found relation benchmarks could be scored well without using
spatial information. Here the mirror image shows up, a model scoring well by
reproducing annotator selection habits. RQ2's result is Ratner et al.'s (2017)
weak-supervision prediction confirmed in a domain the original work did not
address, with the standard semi-supervised alternative (Lee, 2013) implemented
and measured, not argued away.

## 7.6 Limitations and threats to validity

**Internal.** Thresholds are fitted to six annotator groups of one dataset.
Audits are verdicted by the author and by a vision-language model, neither of
them a disinterested human, since the author built the tool, and the model is
a system of the kind §4.13 shows annotates poorly, admissible only because
judging a supplied claim needs the half of it that works, and the decoys
measure it doing so (§4.14, §7.4). No disinterested human estimate exists,
which is the sharpest limitation on every precision figure here. Both judges
disagree with each other often enough to show neither is echoing the other,
but agreement between two interested parties is not independence, and the
two-stage audit shares machinery with the rule change it evaluates.

**External.** One laboratory domain, six object classes, one camera and
mounting. Fitted constants (`near_T`, ε values, contact threshold) are
dataset-specific by design, and the transferable artefact is the *procedure*,
fit on some annotators and validated on held-out ones, not the numbers
themselves, while the only out-of-domain evidence is qualitative
(Supplementary E.3) and measures nothing, since no labelled out-of-domain gold
exists. What is missing is a modest labelled cross-domain sample. It is cheap
enough that a replication should simply include one. Full automation is
currently detection-bounded (0.38 end-to-end with a worst-case zero-shot
detector; the authors' trained detector would close most of that gap,
unverified without their weights).

**Construct.** "Agreement with human labels" is an imperfect proxy when the
humans disagree with each other, and the per-annotator decompositions mitigate
that without eliminating it. RQ2's result compares supervision *at this
dataset's annotation scale*, and claims nothing about automatic labels beating
abundant, guideline-driven human annotation, which is a regime this dataset
does not contain.

**Ethics.** Some frames contain identifiable people, so those images are
personal data under the Data Protection Act 2018, and anonymising them is data
minimisation, not a courtesy (§3.12). Figure 4.1 is the only figure here
showing scene imagery, with faces pixelated from the dataset's own human boxes
before the file is written, and the other nine are charts. Pixelation also
runs automatically over the audit pack. An item whose judgement that would
compromise is dropped, never shown.

## 7.7 The objections of §2.9, answered

The literature review stated five objections to a rule-based annotator before
any result was reported, so that this chapter could be read as an attempt on
them. Each is answered with the evidence that bears on it. That includes the
two the evidence does not settle.

**Vocabulary scale is not answered, and is conceded.** Nothing in this
dissertation bears on predicates beyond the seven, and the objection, that a
hand-written rule set does not extend by learning, stands in full. What the
work offers is a boundary, not a rebuttal, since the rules are decidable
because the predicates are spatial, and §3.3 makes that dependence explicit
without hoping it generalises. Section 8.3 records it as the limitation most
likely to matter to anyone reusing the method.

**Systematic error is partly confirmed and partly refuted, and the split is
informative.** The objection predicts that a downstream model absorbs the
rule's blind spot as fact, and Chapter 6 is that prediction coming true, since
the gap on the field's own metric is concentrated where the annotation itself
is defective. It also predicts that consistent-but-wrong supervision is worse
than inconsistent human supervision. That is not what this dataset shows.
Chapter 5's controlled classifier refutes it outright (§5.2), the benchmark
returns parity, not the predicted deficit (§6.3.1), and at the planner every
automatic-arm failure is a relation missing, never a wrong one acted on
(§5.7). The reconciliation is that systematic error is worse than random error
only when it is *wrong*, and §4.12's finding, front/behind agreeing with
itself across viewpoints 0.958 of the time while agreeing with the annotators
0.70 of the time, is the shape of a consistent rule meeting a different
convention, not of a consistent mistake.

**Circular validation is conceded.** Section 7.4 lists this first among the
choices a stricter replication should improve, and no result in Chapters 4 to
6 removes it. What belongs here is the extent of the concession, since the
circularity bounds the *precision* estimates, which rest on verdicts the
author gave. Circularity does not reach the structural guarantees of §3.6,
which are checkable without any verdict at all, nor the downstream findings of
Chapters 5 and 6, which are scored against human annotation the author did not
produce and in which the automatic arm is judged by its rival's yardstick.
What would remedy it is a panel of judges with no stake in the tool, scoring a
fresh draw from the shipped labels under the same blind instrument, and while
the sampling, rendering and scoring code are in the repository, the panel is
what is missing. Until that exists the precision figures remain
author-verdicted. The support rule in particular is unexamined by anyone
outside this project.

**Reference frame is answered as far as it can be, which is not all the way.**
The dissertation does not assert that the camera frame is correct. What it
shows is that the tool applies one frame consistently (§4.12), that two
annotator groups applied another (§4.5), and that pooled recall rises from
0.70 to 0.91 once the convention is aligned. So the finding establishes
disagreement, not error, and identifies which party is consistent. Which
convention a robot should obey is not established, and no measurement in this
dissertation could, because that is a question about the specification, not
about the data.

**Better collection, not cheaper labels, is not attempted, and is partly
answered sideways.** The project re-collected nothing, but it did produce the
operational definitions the annotation process never had (§3.5) and quantify
three defects those definitions would have prevented, which is what a better
collection round needs. Cheap labels and correct definitions are separate
problems. This work solves the definition problem while delivering the cheap
labels.

The project set out to show that automatic labels are *not much worse* than
human ones, and found conditions under which they are decisively better, those
conditions being the sparse, guideline-free annotation this dataset occupies.
So the bottleneck was not only slowing the dataset down. What it also limited
was what the dataset could teach.

## 7.8 Social and professional dimensions of the result

The evidence above is about accuracy. What follows is about what the accuracy
is for, and it sits here, not in a chapter of its own, because each point is a
reading of a measurement already reported.

**The human role, not the job.** The project automates paid human work, and
what the measurements support is narrower than displacement. At this scale the
manual process produced sparse, inconsistent labels at a cost of nine
annotators, and the realistic effect is a changed role, from labelling every
pair to reviewing a measurable flagged minority (§4.7). Dataset construction
also becomes affordable for groups that could not fund manual annotation at
all, since the pipeline runs on a consumer 6 GB GPU.

**Downstream consequence.** Annotation quality propagates, and a planner
consuming wrong spatial relations can act wrongly in physical space. So this
evaluation centres on audited precision, on abstention over guessing, and on
per-failure attribution, and Chapter 6 separates labels that are *correct*
from labels that are *human-like* for the same reason. Section 7.1 records
support's precision failure as a failure, not as a caveat, for the same
reason. Trust in automatic labels should be calibrated by that kind of
evidence, not assumed from a headline recall figure.

**Bias.** The geometric rules carry no demographic component, since they
compute from positions and extents, but the perception stack is learned, and
detection quality for the `human` class cannot be assumed uniform across
people, because published detectors have documented disparities. People in
this dataset are members of the collecting research group, so the question is
not testable here, and it is recorded as a deployment consideration, not a
resolved issue.

**Research integrity and professional practice.** Predictions were registered
before the benchmark ran and reported as they fell, with one confirmed, one
refuted, and one left unresolved once a better-controlled replication shrank
the margin it rested on (§6.6). One withdrawn hypothesis remains in the text
(§6.4), and two built refinements are reported as measured and declined
(Supplementary D.4). Throughout, the work follows the BCS Code of Conduct
(BCS, 2022) in the privacy safeguards of §3.12, in claims bounded by the
limitations of §7.6, and in documenting failures so others can build on them
as readily as on the successes. In the engineering, that means version control
with a clean history, a test suite run before every change ships, one
configuration file holding every threshold, and seeded reproducible runs
(Supplementary B). Without reproducible code, experimental results cannot be
verified, which is why the reproducibility package is a deliverable with the
same status as the results.

## 7.9 Summary

This chapter read the three iterations together. RQ1 is answered differently
by predicate, and RQ2's strong classifier result is scoped by a benchmark that
cannot separate the same two label sources, a disagreement §7.1 states and
does not resolve in the project's favour. Mostly the remaining failures are
calibrated abstention and measured annotator behaviour, not tool error, and
the support arc shows the method correcting itself twice under its own
evidence, while what looked like a depth ceiling proved largely a rules
ceiling. Three measured defects sit in the dataset's own annotation process,
which reframes "agreement with the humans" as a per-annotator quantity. Of the
five objections registered in §2.9 before any result, two are answered
empirically, one is conceded, one is answered as far as measurement can reach,
and one is not addressed at all. Chapter 8 closes the dissertation against its
objectives and research questions, states the contributions, and sets out what
is left undone.
