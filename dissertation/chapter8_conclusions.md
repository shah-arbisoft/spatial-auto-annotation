# Chapter 8: Conclusions and Future Work

This chapter closes the dissertation: what each objective was met with and
how the two research questions are answered (§8.1), the contributions
(§8.2), each limitation paired with the experiment that would settle it
(§8.3), and a personal reflection on how the project was actually conducted
(§8.4).

## 8.1 Summary of the dissertation

The aim was to determine whether spatial-relationship annotation for
robot-acquired images can be automated, and whether the resulting labels are
good enough to replace the human ones. Six objectives carried it. **O1** is
the pipeline: 836 images annotated in about five minutes on a 6 GB GPU, no
human deciding any label, the dataset's formats written byte-compatibly.
**O2** is the specification, with every threshold fitted on groups 0 to 5
alone. **O3** is the fidelity study, mean recall 0.85 and 0.74 on annotators
no threshold saw, against three baselines, a vision-language model, ten
ablations and audited precision. **O4** attributes every one of the 1,650
missed triplets to a cause, leaving roughly 7% genuine tool error (§4.10,
§7.2). **O5** is the controlled experiment, 0.75 against 0.30 human and 0.36
self-trained. **O6** repeats it in the source paper's framework with a
frozen detector and three seeds per arm, and carries it one link further to
a planner; it is met in the sense that decides the answer, since the
heavyweight test does not agree with the lightweight one, the disagreement
is localised to the annotators whose labels Chapter 4 convicted of measured
defects, and both readings are reported and not one selected. Section 1.2.2
fixed what would count as an answer before any result was reported; what
follows is the verdict against those criteria.

**RQ1 is answered yes on five of seven predicates and qualified on two.**
The criterion was per-predicate recall on annotator groups no threshold ever
saw, judged against two references fixed in advance; the trivial baselines
are beaten by a wide margin, 0.74 held-out and 0.85 pooled against 0.14 for
both random and majority, and the second reference, how well two human
annotators would agree, is one this dataset cannot supply, for reasons
Supplementary F.12 sets out. Comparability therefore rests on the baselines and
the per-predicate audit, and that is a weaker footing than the criterion
intended. What the evidence establishes is that the tool reproduces the
human record far above any baseline and that the labels it adds beyond that
record survive blind audit on five predicates. What it cannot establish is
comparability with the human process itself, because no measurement of that
process's own consistency exists to compare against: *answered yes* below is
a verdict against the references that could be obtained, not against a human
ceiling. The second condition, that labels beyond the human
record survive audit, holds for the lateral, proximity and depth-decided
predicates at blind-audited precision 0.79–1.00. **It is not met for
support.** At 0.535 [0.42, 0.65] about half the support labels the tool adds
beyond the human record are wrong, beating chance but short of a standard
anything should be built on, so support answers RQ1 on recall and fails it
on precision, and the per-predicate form of the question exists so this
cannot be averaged away. For anything consuming the output, support is a
candidate set and not a label set. The qualification is `in front of` and
`behind` at 0.70/0.71 pooled, which is a disagreement about the words rather
than a failure of the criterion, §4.12 showing the tool reproducing its own
verdict across viewpoints 0.958 of the time while two annotator groups
applied the opposite convention.

**RQ2 is answered yes at the classifier, not met at the planner by the tool
alone, and undecided at the benchmark.** The controlled classifier gives
0.75 against 0.30 on held-out *human* gold, the rival source's own yardstick
and the harder direction the criterion specified. At the planner, §1.2.2
fixed the comparator as the human arm: against that arm's 25 of 25 the
tool's relations alone clear 19, which falls short, and what draws level is
the union with the vision-language source at 25 of 25, label-free but not
the treatment RQ2 names. So the planner establishes that human annotation
can be matched *without a human in the loop*, not that the tool matches it.
At the benchmark, parity satisfies *at least as well* as written, but a null
result is not a demonstration of equivalence, the paired difference being
-0.0006 with a 95% interval of [-0.070, +0.069], so the honest word for
0.292 against 0.293 is *undecided*, and since §1.2.2 required agreement
across all three for an unqualified yes, a tie is not agreement. What
Chapter 5 measures as a two-and-a-half-fold advantage the ranked metric
measures as parity; §6.3.1 localises the residual difference to the two test
annotators with measured defects, and the third arm suits the argument least
and is reported for that reason, vision-language labels leading at 0.329
against 0.293 and 0.292 (§6.3.2).

"Good enough to replace" is met at the classifier, unrefuted at the
benchmark, and reached at the planner only once a second automatic source
is added alongside;
"better" holds where ground truth means geometric consistency and not where
it means annotation practice. The evidence carries this conditional, on this
dataset: **automatic labels are better where ground truth means geometric
consistency, and do not overtake human labels where it means annotation
practice.** Robot planning needs the first, which is why the planner
separates the sources decisively and the ranked benchmark cannot separate
them at all.

## 8.2 Research contributions

**For the source dataset and its authors.** An annotator that labels the
existing images 20 times more densely in five minutes, in the dataset's own
formats, with the operational definitions the annotation process never had.
The fitted `near` threshold answers Wang et al.'s (2025) future-work request
for "spatial thresholds for near", and two annotation defects are quantified
for the first time: front/behind inverted in two of nine groups, and `near`
used by only three, with fourfold variation in exhaustiveness.

**For work on scene-graph benchmarks.** Evidence, by dissociation, that
ranked recall against sparse, guideline-free annotation partly measures
agreement with annotator habits: the model that ranks better memorises which
pairs annotators record, and the model that covers the relation types they
never recorded is the one trained on consistent computed labels.

**For weak supervision.** A controlled three-way comparison, on the same
features, model, split and seeds, showing that programmatic labels out-teach
both scarce human labels and the standard remedy for them. The mechanism is
measured, not inferred: self-training contributes roughly a thousand
confident negative pseudo-labels for every positive one.

**Methodologically.** Calibration held out by *annotator* and not by image;
a sparse-gold protocol pairing recall with audited precision; exhaustive
loss attribution; a way to probe what annotators would score against one
another without their ever having labelled the same images, together with
the finding that on batches this unbalanced it yields an upper bound and
no measurement (§4.6); and a
reliability check needing no labels at all, from recovering the fact that
the dataset was cut from a continuous capture (§4.12). The last is the most
likely to transfer, since many robotics datasets are sequences presented as
image sets, and there a predicate's agreement with itself across viewpoints
separates a rule that is wrong from one that is merely uncertain.

## 8.3 Limitations, and future research and development

Each limitation below is stated with the experiment that would settle it,
because that is more useful than an apology; Supplementary B specifies each of
those experiments.

**The chain reaches the plan, not the robot.** The planner experiment (§5.7)
carries the chain one link further, from labels to the plan a robot would
act on, but on 25 occluder-selected scenes and under a scorer that cannot
see a false positive it is supporting evidence rather than a closed link.
What is missing outright is execution: no robot moved. The dataset ships no
object meshes, so closing this needs a physics simulator (PyBullet or Isaac
Sim) with primitive-shape proxies substituted for the detected boxes, then
replaying the same 25 scenes' plans and scoring collision-free execution
rather than a text ordering. That would also make the redundant-step cost
of a hallucinated support relation visible instead of merely inferred from
its precision.

**No precision figure is verdicted by a disinterested human.** Every one
rests on the author's judgement and a vision-language model's. The decoys of
§4.14 measure both judges rather than assuming them, and the two disagree
often enough that neither is echoing the other, but two interested judges are
not independence. Support is where this matters most, since it is also the
predicate that fails. Settling it needs a panel with no stake in the tool,
scoring a fresh draw from the shipped labels under the same blind
instrument; the sampling, rendering and scoring code are all in the
repository.

**The benchmark is undecided, and the run that would decide it is written.**
Three seeds bound the paired difference to [-0.070, +0.069], so §6.7 reports
parity without establishing it; ten per arm would tighten that to
about ±0.020, and the notebook that would do it ships unrun for want of
GPU-hours.

**One domain, one camera, six object classes.** What transfers is the
calibration procedure, not the fitted constants, and the only out-of-domain
evidence is qualitative (E.3).

**A class list stands in for geometry on `on`/`under`, with a known blind
spot.** The support rule excludes `human` from either role because mask
contact alone cannot tell a person *holding* an object from a surface
*supporting* one (0 of 2,466 gold support triplets involve a person either
way); ablation A10 tested the geometric alternative, a drop-fraction
threshold, and the two populations do not separate (Supplementary D.8). The guard
is scoped to the classes this dataset labels, so an object held by any
unguarded class, whether a robot manipulator, a trolley or an animal, would
defeat it the same way a person does, which bounds every human-robot interaction
scene this pipeline has not been run on.

**Front/behind is bounded, but less by depth than the number suggests.**
Section 4.9 bounds the engineering: neither a larger depth model nor
multi-frame geometry moves the pair, so the limit is monocular ambiguity in
the scenes rather than model capacity. A predicate reproducing its own
verdict across viewpoints 0.958 of the time while recovering 0.70 of the
human labels is applying a criterion the annotators did not share, so the
intervention with the best expected return is a written annotation guideline
ahead of a better network, an uncomfortable conclusion for a
computer-vision project and the one the evidence supports. The routes that
stay open are a calibrated stereo pair or an RGB-D capture; wider surface
detection was built, measured and declined (Supplementary D.4).

**Detection bounds full automation.** End-to-end recall with a zero-shot
detector is 0.38 against the relation layer's 0.85 conditional on detection.
The gap is detection, not relations, and the source paper's own trained
detector (0.93 mAP@50) would close most of it; that check needs only the
released weights.

**Scale is shown without ground truth.** The pipeline was run over the 1,766
frames nobody has annotated: 562 keyframes after content-adaptive selection,
58 minutes, 185,242 triplets, a predicate distribution 0.032 in total
variation from the annotated portion (E.5). Capacity and stability on
unfamiliar input are therefore measured; correctness on that portion is not,
and cannot be without labels. Section 4.12 substitutes self-agreement for
truth and should be read as the weaker thing it is.

## 8.4 Personal reflections

The most useful thing this project taught me was to distrust a number until
I know how it was produced.

That lesson arrived early. My first depth results were poor and every unit
test passed: the dataset's images are stored rotated 180 degrees behind an
EXIF flag, so masks and depth were read from an upside-down image while the
boxes stayed upright, and no automated check could have caught it, every
component behaving as written. I found it by rendering an image with its
boxes drawn on and looking at it, and have inspected samples after every
significant change since, which has caught two more.

The second lesson was that "agreement with the humans" is not one target. I
began by treating the human labels as ground truth and my disagreements as
errors; measuring them properly showed the three annotator behaviours
Chapter 4 reports, and reframed the question from "how close can I get to
the humans" to "what do the humans mean, and where do they disagree". Almost
every later design decision, including calibrating on some annotators and
validating on others, follows from that shift.

The third lesson was about my own claims. On a single benchmark run I
reported that my labels won against the one test annotator with clean
conventions, by 0.011 over 73 images; two more seeds showed them tied and I
withdrew it (§6.3.1). I had been careful in one place, fitting thresholds on
some annotators and validating on others, and careless in another, treating
one training run as a result. Recording the retraction felt uncomfortable
and is, I think, right.

Given the time again I would front-load the experiments that answer the
question the project actually asks. I spent considerable effort proving the
labels are correct and train models well, and little on whether they help a
robot decide what to do, the entire motivation; the planner experiment ran
last and was the clearest single result, and run in week five it would have
pointed the intervening work at supply of the support relation. I would also
have had two people annotate the same fifty images in week two, since one
afternoon would have produced outright the inter-annotator agreement figure
I could only estimate indirectly. What I am most pleased with is not the
accuracy figure but the two refinements I built, measured and declined: both
were plausible and took real work, the data said neither helped, and
reporting that is where this started feeling like research.
