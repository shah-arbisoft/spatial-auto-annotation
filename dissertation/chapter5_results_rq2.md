# Chapter 5: Downstream Utility (RQ2)

This chapter answers RQ2 with a controlled experiment over three label
sources and a fourth added later (§5.1–§5.6), then follows the source
paper's robot-planning chain one link further, asking whether the label
source changes the plan an LLM planner produces (§5.7).

## 5.1 The controlled experiment

RQ2 asks whether the automatic labels are good enough to *train* a relation
model as effectively as human labels. The experiment isolates exactly that
variable: one lightweight classifier per predicate (a small MLP over pure
geometric pair features — relative position, depth difference, box geometry,
size-relative gap, mask-contact fractions), trained with **identical
features, architecture, seeds, positive-oversampling and group split**, then
evaluated against the **held-out human gold** (annotator groups 6–8, whose
data influenced no threshold, no calibration, and no training). The only
difference between the runs is the supervision source, under a fixed
60-iteration budget for every classifier — some do not fully converge,
identically for every source, so the comparison is between supervision
signals under equal compute, not between tuned models.

Three sources are compared in the core experiment (a fourth arrives with the
result it produced, §5.2): **human**, the ~10% of ordered pairs the
annotators chose to label; **automatic**, every pair the tool's rules fire
on, dense and rule-consistent; and **self-trained (pseudo-labelled)**, the
rival remedy from the semi-supervised literature (§2.4), implemented rather
than argued about — a teacher trained on the human labels exactly as in the
human arm, its confident predictions on the *unannotated* training pairs
(probability ≥ 0.90 either way) becoming pseudo-labels, and a student
retrained on the union (Lee, 2013), a pair the annotators touched at all
counting as labelled since their silence on its other predicates is
informative. The third arm answers the question that precedes the project's
premise: if the human labels are too sparse, why not simply stretch them?
Each source labels the same pairs its own way, and each model inherits its
source's character; that contrast is the experiment.

## 5.2 Result

Averaged over three seeds (42/43/44); each cell shows mean (min–max):

| predicate | human-trained | self-trained | vision-language | auto-trained | gold (held-out) |
|---|---|---|---|---|---|
| on | 0.84 (0.83–0.86) | 0.90 (0.89–0.92) | 0.76 (0.68–0.89) | 0.88 (0.87–0.89) | 348 |
| under | 0.44 (0.36–0.53) | 0.59 (0.59–0.60) | 0.84 (0.71–0.91) | 0.85 (0.85–0.85) | 192 |
| to the left of | 0.22 (0.18–0.29) | 0.31 (0.27–0.34) | 0.42 (0.41–0.43) | 0.95 (0.95–0.95) | 446 |
| to the right of | 0.25 (0.22–0.31) | 0.38 (0.37–0.39) | 0.35 (0.30–0.44) | 0.99 (0.99–0.99) | 550 |
| in front of | 0.09 (0.08–0.10) | 0.12 (0.12–0.13) | 0.08 (0.07–0.09) | 0.19 (0.19–0.19) | 609 |
| behind | 0.15 (0.12–0.17) | 0.22 (0.17–0.28) | 0.21 (0.21–0.23) | 0.37 (0.37–0.37) | 580 |
| near | 0.08 (0.00–0.19) | 0.03 (0.00–0.06) | 0.00 (0.00–0.00) | 1.00 (1.00–1.00) | 93 |
| **mean** | **0.30** | **0.36** | **0.38** | **0.75** | |

The fourth arm answers what §4.13 raises but cannot settle: if a
vision-language model is not a good enough *annotator*, is it a good enough
*teacher*? The same model labelled all 600 training images, used exactly as
the other sources are; every arm trains on precisely the pairs it covers,
and the other three columns are unchanged from the three-arm experiment to
the last decimal, which checks that this is an addition to the same
experiment. It does, with a qualification: at 0.38 the vision-language
labels teach better than the sparse human labels they would replace and
about as well as the standard remedy for scarce labels, while remaining
half as useful as computed geometry.

{{fig:rq2-with-vlm}} draws the four arms per predicate. Training on the
automatic labels multiplies downstream mean recall by ~2.5 against the human
annotators' own held-out labels: 0.75 vs 0.30. Self-training improves the
human baseline on six of seven predicates but lifts the mean only to 0.36,
closing **15% of the distance** to the automatic arm: stretching the
existing labels helps, and does not substitute for labelling every pair
consistently. The seed spreads carry a second result: the auto-trained
model's recall varies by at most 0.02 across seeds on every predicate, while
the human-trained model's varies by up to 0.19 (`near` spans 0.00–0.19) and
the self-trained model inherits the instability (`behind` spans 0.17–0.28).
Sparse supervision is not just weaker, it is *unstable*, its outcome hostage
to sampling noise, and self-training passes that on to its student.

### 5.2.1 The same experiment on four other indicators

A recall-only table invites the objection that the automatic arm labels
twenty times more densely, so of course it recovers more. Measured, the
answer does not flatter the tool: **on every indicator except recall the
automatic arm comes last** — macro precision 0.136 against the human arm's
0.252, F1 0.194 against 0.267, average precision 0.164 against 0.230, micro
F1 0.066 against 0.262. One number shows why that is not a verdict on label
quality: average precision is threshold-free, so no arm improves it by
committing to more pairs, yet the automatic arm scores **0.040 on `to the
left of`**, the predicate §4.4 audited at fifteen of fifteen extra
predictions correct. An arm cannot be both wrong and right about laterality,
so what the column measures is agreement with which pairs an annotator chose
to write down — §4.3's artefact one level down the chain. Appendix F.5 gives
the full table, and concedes what it cannot settle: §4.4 audited the rule
layer's extras, not the classifier's, so these precision figures are
uninterpretable rather than favourable. The automatic arm dominates at
teaching a model to recover the relations humans recorded and loses at
imitating which relations humans chose to record, and no metric computed
against this gold separates the two.

## 5.3 Why self-training does not rescue the human labels

The pseudo-label arm's own bookkeeping explains its ceiling. Of the 60,762
training pairs the annotators recorded just **6,026**; filling in the rest,
the teacher adds roughly **54,000 confident negative** pseudo-labels per
predicate against only **36 to 67 confident positives**, about 1,000 to 1.
Trained on annotation in which most pairs carry no label, the teacher has
learned above all that pairs usually have no relation, and self-training
feeds that conviction back as though it were evidence: what propagates is
not the annotators' knowledge but their silence. This is the failure mode
§2.4 predicted, now measured — pseudo-labelling is well behaved when the
seed is a representative sample of the pool, and this seed is not. The
`near` row makes it sharp: human-trained recall is already near collapse at
0.08, and self-training pushes it *down* to 0.03, confident pseudo-labels
from a weak, idiosyncratic notion burying what little positive signal
existed. Where the seed is defective, self-training amplifies the defect;
the comparison is therefore not "programmatic labels beat doing nothing" but
"programmatic labels beat the standard remedy, under identical conditions,
closing more than six times as much of the available gap". Active learning
is not tested because it fails for a simpler reason (§2.4): it still buys
human labels, lowering the bottleneck's cost without removing it.

## 5.4 Why the automatic labels win, and two consistency checks

The human arm shows the mechanism §5.3 traced, directly: where the annotation is thinnest
and least consistent (§4.5, §4.7), recall collapses (`near` 0.08, lateral
0.22–0.25); trained on dense rule-consistent labels the same model learns
the geometry (near 1.00, lateral 0.95–0.99, support 0.85–0.88) — §2.3's
weak-supervision prediction confirmed under controlled conditions. Two
checks argue the result is real: the auto-trained model's profile almost
exactly reproduces the rule layer's own held-out performance (mean 0.75
against the rules' 0.74; front/behind 0.19/0.37 against the rules'
0.20/0.37), so the classifier *distilled the annotator*, which is what "the
labels are learnable" means; and all three arms face identical features, the
same oversampling cap and the same held-out gold, including the
convention-inverted annotators, which penalises every arm's front/behind
equally.

## 5.5 Boundaries of the claim

The evaluation gold is itself sparse human annotation, so recall is primary,
mirroring RQ1, and §5.2.1 shows why none of the other columns reads as an
error rate here. The human-trained model's weakness is partly a property of
*any* sparse supervision at this scale; more human labels would improve it,
but producing them is the bottleneck this project removes, and the
self-trained arm shows the shortfall cannot be computed away instead. The
front/behind rows are depressed for every arm by the held-out groups'
inverted convention (Chapter 4), so the penalty is shared.

One structural caveat needs stating plainly. The classifier's features are
geometric and the automatic labels come from rules over closely related
geometry, so the auto-trained arm is partly re-learning its own generator.
Read alone, this chapter shows the automatic labels are *learnable* and the
human labels are not — not that they win under any featurisation. Three
things keep that meaningful: every arm gets identical features; being
learnable is itself the property RQ2 asks about, since a downstream consumer
must extract a consistent signal; and Chapter 6 removes the circularity by
repeating the comparison in a full scene-graph model with visual features.

## 5.6 From labels to robots: where this sits in the source paper's chain

The source paper's end goal is explicit: spatial understanding exists so
that robots can plan. Its motivating example shows an LLM planner failing to
remove a cube before grasping the book beneath it until SGG-derived
relations are supplied; its own evaluation of that chain stops at SGG
quality, six models benchmarked on the human labels, topping out at
mR@100 = 0.49 (VCTree), every model saturating by epoch 2–6 and `near`
stuck at 0.22–0.25 (§2.2). Each symptom has a measured cause and a built
remedy: saturation within six epochs is what training on the 8,926 sparse
triplets of §4.2 looks like, and the automatic labels give 20× the
supervision on the same images; the universal `near` failure is what a
three-annotator label with no operational definition looks like, and the
fitted-threshold labels are perfectly learnable (this chapter's proxy
reaches 1.00); the depth predicates were being taught two opposite
conventions, and the automatic labels apply one. Three predictions follow,
registered before the direct test that judges them in Chapter 6: later
saturation, a higher plateau, and the recovery of `near`.

## 5.7 The planner experiment: does the label source change what a robot would do?

The source paper's motivating example is asserted on one scene; here it is
run as an experiment. Twenty-five held-out scenes were selected in which a
target object has a second resting on it, so any safe plan must move the
occluder first. Each goes to an LLM planner in prompts differing only in
what they state: **A** lists the objects alone, **B** adds the human
relationships, **C** the automatically computed ones. One filter runs over
both relation conditions, so neither is offered a relation the other was
denied (density is not equalised, since matching a sparser source would mean
discarding true relations), and a plan is safe if it moves the occluder
before grasping the target, judged by published rules
(`eval/score_planner.py`), blind by construction; Appendix E.5 gives the
filter, the prompts, the scene-level forensics and a scoring defect
hand-reading caught. The experiment ran twice, on `gemini-flash-latest` and
the reasoning model `gemini-3.1-pro-preview`, with two conditions on the
larger planner only: **D**, the vision-language model's relations from
§4.13, and **E**, the union of C and D.

| Condition | Prompt states | Safe plans (flash) | Safe plans (pro) |
|---|---|---|---|
| A | objects only | 0 / 25 | 0 / 25 |
| B | human relationships | 25 / 25 | 25 / 25 |
| C | automatic relationships | 19 / 25 | 19 / 25 |
| D | vision-language relationships | not run | 20 / 25 |
| E | automatic and vision-language combined | not run | **25 / 25** |

{{fig:planner-sources}} shows the five conditions, and three findings sit in
them. **The two planners agree exactly**, and not only in the totals: C
fails on the same six scenes under both. A finding that survives replacing
the reasoning engine, down to which scenes fail, is a property of the prompt
and not of the model reading it — the claim the experiment exists to make —
and it disposes of the objection that a more capable planner would infer
support from the object list, which it does not, in twenty-five scenes of
twenty-five, twice. **Every failure in both automatic arms is a missing
support relation**, never a plan reasoning badly from what it was given: in
no scene was the relation present and ignored. Six misses in twenty-five is
24%, against support recall of 0.81/0.75 in §4.2, so the planner result is
the fidelity result one level higher in the chain — and it moves when that
fidelity does, the refit of §4.14 having cost C three scenes (22 of 25 on
the pre-refit labels) while costing E none. **The two automatic sources fail
on disjoint scenes.** D scores 20 of 25, marginally the stronger single
source, and its five failures do not intersect C's six, so a union supplying
more support relations repairs exactly the failing cases and cannot break
the working ones. It does: E clears the occluder in all 25, gaining six
scenes over C and losing none, and the disjointness survives the threshold
refit that widened C's own gap — the only measurement here on which
automatic labels *match* human annotation on a robot-relevant task, with no
human in the labelling loop. Throughout, `grasps_target` and `no_invented`
remain 1.00, so no plan failed for any other reason.

Twenty-five scenes is small, and the pairing is what makes it enough: exact
McNemar tests over the discordant scenes (`eval/planner_paired_tests.py`,
reported in full with the scene-level forensics in E.5) settle the
comparisons the argument rests on — relations against none at p < 10^-5, and
the union's 6-to-0 gain over the tool alone at p = 0.031, as is the human
arm's lead over the tool, the comparison that runs against this project —
and name what 25 scenes cannot settle, the tool against the vision-language
source being 6 discordant to 5 at p = 1.00, so that two-scene margin is not
read here. The tests are sharp exactly where the absolute rates are not, C's
own rate being 19 of 25 with a 95% interval of [0.55, 0.91].

Appendix E.5 carries five limits with the design; two belong here because
they bound what the result can mean. The scenes were selected to contain an
occluder, so it speaks to that situation and not to task planning at large,
and no robot moved, so this measures plans rather than executions. And
structurally, **the scoring rule cannot see a false positive**: a support
relation the tool asserts wrongly costs an unnecessary step and never a
failed plan, so with §4.14 measuring those at 0.40 precision this experiment
is insensitive to them by construction, testing whether the labels carry
*enough* and not whether they carry *too much*. A task penalising wasted
motion, or one where moving the wrong object is unsafe rather than merely
inefficient, would rank these label sources differently, and nothing here
predicts how.

What it settles is the question §5.6 could only frame. The tool's relations
alone carry 76% of the decision-relevant content human labels carry on this
task, against 0% for no labels at all, and combined with the
vision-language model's they carry all of it. The residual gap is not a
property of computed labels in general but of one predicate whose recall is
measured, whose precision was refitted, and whose failure modes §4.9 and
§4.14 diagnose — which is why tightening that predicate moved this number
and moved nothing else in the chapter.

## 5.8 Answer to RQ2

Yes at this level of test, and with more than was asked of it. At this
dataset's scale of human annotation the automatic labels are not merely
"good enough" but **substantially better training material than the human
labels themselves**, because density and consistency dominate raw human
authority when supervision is sparse. The self-trained arm rules out the
cheap alternative: the standard semi-supervised remedy recovers only 15% of
the gap, because it propagates the annotators' silence, not their knowledge.
That is the dissertation's core claim — removing the bottleneck can *grow*
dataset utility instead of approximating it — demonstrated on the dataset's
own held-out annotators and against the obvious rival.

The scope of that answer is set by what follows it. Chapter 6 repeats the
comparison inside a full scene-graph model and returns parity, 0.292 against
0.293, which no experiment of that size can separate. So the advantage
established here is an advantage on the controlled classifier, where
features are held identical, and it does not carry to the benchmark's ranked
metric. Section 6.4 measures why the two disagree, and the answer to RQ2
should be read as that pair of results and not as this one alone.
