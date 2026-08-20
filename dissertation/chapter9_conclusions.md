# Chapter 9: Conclusions and Future Work

This chapter closes the dissertation: what each objective was met with and
how the two research questions are answered (§9.1), the contributions (§9.2),
each limitation paired with the experiment that would settle it (§9.3), and
a personal reflection on how the project was actually conducted (§9.4).

## 9.1 Summary of the dissertation

The aim was to determine whether spatial-relationship annotation for
robot-acquired images can be automated, and whether the resulting labels are
good enough to replace the human ones. Six objectives carried that aim, and
each is met with evidence, not assertion. That evidence is already reported.

**O1** is the pipeline itself: all 836 images annotated in about five minutes
on a 6 GB consumer GPU, no human deciding any label, the dataset's own formats
written byte-compatibly and verified by a zero-error load-write round trip.
**O2** is the specification: written operational definitions of all seven
predicates, every threshold fitted on annotator groups 0 to 5 alone, `near`
generalising to the held-out annotator at recall 1.00 and the support rules
reaching held-out F1 0.87. **O3** is the fidelity study: per-predicate
agreement with all 8,926 human triplets, against three trivial baselines, a
vision-language model under the same definitions (§4.13), nine ablations,
audited precision and cluster-bootstrap intervals, giving mean recall 0.85 and
0.76 on annotators no threshold ever saw. **O4** is the diagnosis: every one
of the 1,689 missed triplets attributed to a cause, leaving roughly 7%
attributable to genuine tool error (§4.10, §7.2). **O5** is the controlled
experiment isolating the label source, at 0.75 mean recall against held-out
human gold for the automatic arm against 0.30 human and 0.36 self-trained.
**O6** repeats that comparison in the source paper's own framework, with a
frozen detector and three seeds per arm, and carries it one link further to a
planner.

O6 is met in the sense that matters for an honest answer:
the heavyweight test does not agree with the lightweight one, the
disagreement is localised to the annotators whose labels Chapter 4 convicted
of measured defects, and both readings are reported rather than one selected.

Section 1.2.2 fixed what would count as an answer before any result was
reported, so both questions are settled against those criteria, not a
measure chosen afterwards. Section 7.1 argues the answers; what follows is
the verdict and its conditions.

**RQ1 is answered yes on five of seven predicates and qualified on two.** The
criterion was per-predicate recall on annotator groups no threshold ever saw,
judged against two references fixed in advance, and both are met: the trivial
baselines are beaten by a wide margin (0.85 against 0.14), and the tool's mean
agreement with the consistent annotators, 0.869, falls inside the [0.74, 0.92]
interval within which those annotators can be shown to agree with one another
(§4.6), so the tool is not distinguishable from a tenth annotator. The second
condition, that labels beyond the human record survive audit, holds for the
lateral, proximity and depth-decided predicates at blind-audited precision
0.92–1.00. **It is not met for support.** At 0.535 [0.42, 0.65] about half the
support labels the tool adds beyond the human record are wrong, which is well
above noise and well below a standard anything should be built on, and the
criterion of §1.2.2 asked for labels that survive audit rather than labels
that beat chance. Support therefore answers RQ1 on recall and fails it on
precision, and the per-predicate form of the question exists so that this
cannot be averaged away. The
contact rule repaired the box rule's failure but was left at a threshold
fitted where the cost of a false positive was invisible, the claim that it
reached 0.9 was an artefact of auditing unblinded, and refitting the threshold
lifted a measured 0.404 to a measured 0.535 rather than to the 0.667 the
held-out fit had predicted. The qualification is
`in front of` and `behind` at 0.70/0.71 pooled: not a failure of the criterion
but a disagreement about the words, since §4.12 shows the tool reproducing its
own verdict across viewpoints 0.958 of the time while two annotator groups
applied the opposite convention. A per-predicate answer was required precisely
so this could not hide inside a mean.

**RQ2 is answered yes at two of the three levels and is undecided at the
third.** The controlled classifier gives 0.75 against 0.30 and the planner 22
of 25 against 0 of 25 with no relations, both in the harder direction the
criterion specified: held-out *human* gold, the rival source's own yardstick.
The benchmark neither confirms the margin nor contradicts it, and the honest
word for 0.292 against 0.293 is *undecided* rather than *yes*: the automatic
arm's point estimate sits 0.001 below the human arm's, which is a difference
no experiment of this size could resolve in either direction, with overlapping
seed ranges on every slice. Section 1.2.2 required agreement across all three
for an unqualified yes, and a tie is not agreement. What Chapter 5 measures as
a two-and-a-half-fold advantage the ranked metric measures as parity, and
§1.2.2 committed to reporting that rather than resolving it favourably.
Chapter 6 does: what
difference remains sits on the two test annotators carrying measured
labelling defects and reverses on the one without, on a test gold whose
front/behind relations are 72% written by those two, while the auto arm
recovers five times more of the relation types the manual annotation never
recorded and reproduces itself across seeds nearly nine times more tightly.
"Good enough to replace" is met at two levels and unrefuted at the third;
"better", which Chapter 5 and the planner both support, holds where ground
truth means geometry and not where it means annotation practice.

The condition is legible, and it is the sentence the evidence supports:
**automatic labels are at least the equal of human ones wherever ground truth
means human annotation practice, and better wherever it means geometry.**
Robot planning needs the second, which is why the planner separates the
sources decisively while the ranked benchmark cannot separate them at all.

## 9.2 Research contributions

**For the source dataset and its authors.** An annotator that labels the
existing images 20 times more densely in five minutes, in the dataset's own
formats, and the written operational definitions the annotation process
never had. The fitted `near` threshold is a direct answer to the future-work
request in Wang et al. (2025) for "spatial thresholds for near". Two
annotation defects are quantified for the first time: an inverted
front/behind convention in two of nine annotator groups, and `near` used by
only three groups with fourfold variation in exhaustiveness.

**For work on scene-graph benchmarks.** Evidence that ranked recall against
sparse, guideline-free annotation partly measures agreement with annotator
habits rather than spatial correctness. The evidence is a dissociation:
the model that ranks better memorises which pairs annotators record, and the
model that covers the relation types they never recorded is the one trained
on consistent computed labels. The per-annotator decomposition localises the
effect precisely, and the seed replication shows which parts of it survive
training variance.

**For weak supervision.** A controlled three-way comparison, on the same
features, model, split and seeds, showing that programmatic labels out-teach
both scarce human labels and the standard remedy for scarce human labels.
The mechanism is measured, not inferred: self-training contributes
roughly a thousand confident negative pseudo-labels for every positive one,
propagating the annotators' silence and not their judgement.

**Methodologically.** Calibration held out by *annotator* rather than by
image; a sparse-gold evaluation protocol that pairs recall with audited
precision; exhaustive loss attribution; a way to estimate what annotators
would score against one another when they never labelled the same images, by
using a deterministic annotator as a fixed common reference (§4.6); and a
reliability check that needs no labels at all, obtained by recovering the
fact that an image dataset was cut from a continuous capture and asking
whether its labels survive the camera moving (§4.12). The last of these is
the one most likely to transfer. Many robotics datasets are sequences
presented as image sets, and wherever that is true, a predicate's agreement
with itself across viewpoints separates a rule that is wrong from a rule
that is merely uncertain, which is a distinction sparse human annotation
cannot draw.

## 9.3 Limitations, and future research and development

Each limitation below is stated with the specific experiment that would
settle it, because that is more useful than an apology.

**The chain reaches the plan, not the robot.** The planner experiment (§5.7)
closes one of the two remaining links: across 25 held-out scenes an LLM
planner never clears an occluding object when given objects alone (0/25),
always clears it when given the human relationships (25/25), and does so on 22
of 25 with the automatic ones, all three failures traced to a missing support
relation, not to faulty reasoning. What is missing is execution. No
robot moved during this project, so the evidence runs from labels to models to
plans and stops. Putting the same conditions on a physical Spot, or in a
simulator with contact physics, would close the last link, and it is now the
only one left in the chain.

**Precision estimates remain partly author-verdicted.** The independent
validation study (Appendix E.3) is deployed and collecting, with coverage stratified
so the audit-overlap items reach several raters first. Until it completes the
audited figures carry the author's own judgement, conservatively applied and
published for checking.

**One domain, one camera, six object classes.** What transfers is the
calibration procedure, not the fitted constants. A labelled cross-domain
sample of a few dozen images would turn Appendix E.4's qualitative evidence
into a
measurement, and is cheap enough that a replication should simply include one.

**Front/behind is bounded, but less by depth than the number suggests.**
Section 4.9 bounds the engineering: neither a larger depth model nor
multi-frame geometry moves the pair, so the limit is monocular ambiguity in
the scenes, not model capacity. What that leaves is a limitation of a
different kind, since a predicate reproducing its own verdict across
viewpoints 0.958 of the time while recovering 0.64 of the human labels is not
mismeasuring the scene but applying a criterion the annotators did not share.
The intervention with the best expected return is therefore a written
annotation guideline rather than a better network, which is an uncomfortable
conclusion for a computer-vision project and the one the evidence supports.
The measurement routes that stay open are a calibrated stereo pair or an
RGB-D capture; wider surface detection was built, measured and declined
(Appendix D.4).

**Detection bounds full automation.** End-to-end recall with a zero-shot
detector is 0.38 against the relation layer's 0.85 conditional on detection.
The gap is detection, not relations, and the source paper's own trained
detector (0.93 mAP@50) would close most of it. That check needs only the
released weights.

**Scale is demonstrated without ground truth.** The supervising group supplied
the full 2,650-frame capture the released images were cut from, and the
pipeline was run over the 1,766 frames nobody has annotated: 562 keyframes
after content-adaptive selection, 58 minutes, 185,242 triplets, a predicate
distribution 0.032 in total variation from the annotated portion
(Appendix E.6).
Capacity and stability on unfamiliar input are therefore measured, not
argued. Correctness on that portion is not and cannot be without labels;
§4.12's viewpoint-consistency check substitutes self-agreement for truth and
should be read as the weaker thing it is. Closing it needs a modest labelled
sample from those frames, a few hundred triplets, which is an afternoon of
annotation and not a research programme.

**The capture is stereo, and only one eye was used.** The supplied folder is
named `rightimg`, implying a left counterpart held by the supervising group.
True stereo would attack the front/behind bound directly, supplying disparity
at every frame from a known fixed baseline, which is exactly what the
multi-frame estimators of A9 lack: those must recover the camera's motion
first, and on small low-texture objects they answer for only 9% of pairs and
are 0.17 less accurate than the monocular cascade where they do (§4.9,
Appendix D.6). A calibrated stereo pair removes both problems at once and is
the cheapest experiment left on this predicate. It also keeps the method's
premise intact, since stereo is available at capture time, whereas depth
recovered from a robot walking twenty frames is not available to a
single-image annotator at all.

## 9.4 Personal reflections

The most useful thing this project taught me was to distrust a number until
I know how it was produced.

That lesson arrived early and painfully. My first depth-based results were
poor, and the unit tests all passed. The cause was that the dataset's images
are stored rotated 180 degrees behind an EXIF flag, so every mask and depth
value was being read from an upside-down image while the boxes stayed
upright. No automated check could have caught it, because every component
was behaving exactly as written. I found it by rendering an image with its
boxes drawn on and looking at it. Since then I have rendered and inspected
samples after every significant change, and it has caught things twice more.

The second lesson was that "agreement with the humans" is not one target. I
began by treating the human labels as ground truth and my disagreements as
errors. Measuring them properly showed that three of nine annotator groups
used `near` at all, that two labelled front and behind in opposite
directions, and that some labelled support in only one direction. That
reframed the project: the interesting question stopped being "how close can
I get to the humans" and became "what exactly do the humans mean, and where
do they disagree with each other". Almost every later design decision,
including calibrating on some annotators and validating on others, follows
from that shift.

The third lesson was about my own claims, and the one I would most like to
have learned sooner. On a single benchmark run I reported that my labels won
against the one test annotator with clean conventions, on a margin of 0.011
over 73 images; two more seeds showed them tied and I withdrew it (§6.3.1). I
had been careful in one place, fitting thresholds on some annotators and
validating on others, and careless in another, treating one training run as a
result. The discipline was not new to me, I had simply not applied it to model
training variance, and recording the retraction rather than quietly deleting
it felt uncomfortable and is, I think, right.

Given the time again I would front-load the experiments that answer the
question the project actually asks. I spent considerable effort proving the
labels are correct and train models well, and comparatively little on whether
they help a robot decide what to do, which is the entire motivation. The
planner experiment ran last and was the clearest single result: 0 of 25
against 25 of 25 needs no statistics, and run in week five it would have
pointed the intervening work at supply of the support relation, where its only
failures came from. I would also have had two people annotate the same fifty
images in week two, since one afternoon would have produced outright the
inter-annotator agreement figure I ended up estimating indirectly.

Technically, the thing I am most pleased with is not the accuracy figure but
the two refinements I built, measured and then declined. Both were
plausible, both took real work, and the data said neither helped. Reporting
them as null results instead of dropping them was the point at which the
project started feeling like research, not engineering.
