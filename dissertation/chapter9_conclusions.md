# Chapter 9: Conclusions and Future Work

This chapter closes the dissertation. Section 9.1 states how each objective
was met and what the two research questions were answered with, §9.2 sets
out the contributions and who can use them, §9.3 states the limitations
honestly and turns each into the work that would resolve it, and §9.4 is a
personal reflection on how the project was actually conducted.

## 9.1 Summary of the dissertation

The aim was to determine whether spatial-relationship annotation for
robot-acquired images can be automated, and whether the resulting labels are
good enough to replace the human ones. Six objectives carried that aim, and
each is met with evidence rather than assertion.

**O1, build.** A fully-automatic pipeline annotates all 836 images in about
five minutes on a 6 GB consumer GPU, with no human deciding any label, and
writes the dataset's own formats byte-compatibly (Chapter 3). Verified by a
load-write round trip reproducing boxes and labels with zero error.

**O2, specify and calibrate.** All seven predicates have written operational
definitions with explicit thresholds, every one fitted on annotator groups
0 to 5 alone and reported on the held-out groups (Chapter 3). The `near`
threshold generalises perfectly to the held-out annotator who used the label
(recall 1.00), and the support rules reach held-out F1 0.87.

**O3, validate.** Per-predicate fidelity against all 8,926 human triplets,
against three trivial baselines and a vision-language model given the same
boxes and the same definitions (§4.16), with nine ablations and manually
audited precision (Chapter 4): mean recall 0.85, and 0.76 on annotators no
threshold ever saw.
Cluster-bootstrap intervals accompany every headline figure, and an
independent validation study re-estimates precision with judges who have no
stake in the result (§4.13).

**O4, diagnose.** Every one of the 1,689 missed human triplets is attributed
to a cause by re-checking rule conditions, so the failure analysis is
exhaustive rather than illustrative. Genuine tool error accounts for
roughly 7% of misses; the remainder is calibrated abstention and measured
annotator behaviour (§4.10, §7.2).

**O5, test downstream utility.** A controlled experiment trains the same
classifier on three label sources under identical features, splits and seeds
(Chapter 5), so that the only thing differing between arms is where the
labels came from: automatic 0.76 mean recall against held-out human gold,
human 0.30, self-trained 0.36.

**O6, test at the level the field measures.** The same comparison is
repeated in the source paper's own benchmark framework with a shared frozen
detector and three seeds per arm (Chapter 6), and the chain is followed one
link further, to an LLM planner asked to produce a safe grasp plan for 25
held-out scenes under each label source (§5.7). The objective is met in the
sense that matters for an honest answer: the heavyweight test does not
agree with the lightweight one, the disagreement is localised to the
annotators whose labels Chapter 4 convicted of measured defects, and both
readings are reported rather than one being selected.

**RQ1, accuracy.** Yes, with one qualification. On five of seven predicates
the automatic labels match or exceed the human process (0.81 to 1.00 recall,
audited precision ≈ 1.0), and `near` is solved outright, resolving the one
predicate the source paper reports as failing for every model it
benchmarked. The depth pair reaches 0.64/0.66 pooled and 0.84 once two
annotator groups' inverted conventions are accounted for; where the tool
commits it agrees at 0.95 to 1.00 with six of the seven same-convention
annotators, the seventh being the smallest sample in the dataset. The
residual limit is monocular ambiguity, which ablation A8 shows a
four-times-larger depth model does not fix.

**RQ2, utility.** Yes, and by a wide margin at this dataset's annotation
scale. The auto-trained classifier reaches 0.76 mean recall against held-out
human gold versus 0.30 for its human-trained twin, and versus 0.36 when the
human labels are stretched by self-training, the standard semi-supervised
remedy. In the full benchmark the verdict is conditional and the condition
is legible: over three seeds per arm the human-trained model ranks better
against human test annotation (mR@100 0.326 against 0.278), the auto-trained
model recovers about sixty times more of the relation types the manual
annotation omits (zR@100 0.172 against 0.003, with disjoint per-seed ranges;
a difference in label coverage rather than in generalisation, §6.5), and the human
advantage is concentrated entirely on the two test annotators carrying
measured labelling defects, disappearing against the only one without
(0.308 against 0.307).

One link further down the chain the answer is unambiguous. Asked for a safe
grasp plan on 25 held-out scenes where an object rests on the target, an LLM
planner clears that object in 0 of 25 scenes when told only what objects are
present, 25 of 25 when given the human relationships, and 22 of 25 when
given the automatic ones, with every failure traced to a missing support
relation rather than to faulty reasoning (§5.7).

The single sentence the evidence supports: **automatic labels are the better
supervision wherever ground truth means geometry; human labels remain better
wherever ground truth means human annotation practice.** Robot planning
needs the former.

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
The mechanism is measured rather than inferred: self-training contributes
roughly a thousand confident negative pseudo-labels for every positive one,
propagating the annotators' silence rather than their judgement.

**Methodologically.** Calibration held out by *annotator* rather than by
image; a sparse-gold evaluation protocol that pairs recall with audited
precision; exhaustive loss attribution; a way to estimate what annotators
would score against one another when they never labelled the same images, by
using a deterministic annotator as a fixed common reference (§4.6); and a
reliability check that needs no labels at all, obtained by recovering the
fact that an image dataset was cut from a continuous capture and asking
whether its labels survive the camera moving (§4.14). The last of these is
the one most likely to transfer. Many robotics datasets are sequences
presented as image sets, and wherever that is true, a predicate's agreement
with itself across viewpoints separates a rule that is wrong from a rule
that is merely uncertain, which is a distinction sparse human annotation
cannot draw.

## 9.3 Limitations, and future research and development

Each limitation below is stated with the specific experiment that would
settle it, because that is more useful than an apology.

**The chain reaches the plan, not the robot.** The planner experiment
(§5.7) closes one of the two remaining links: across 25 held-out scenes an
LLM planner never clears an occluding object when given objects alone
(0/25), always clears it when given the human relationships (25/25), and
does so on 22 of 25 with the automatic ones, with all three failures traced
to a missing support relation rather than to faulty reasoning. What is still
missing is execution. No robot moved during this project, so the evidence
runs from labels to models to plans and stops there. Putting the same three
conditions on a physical Spot, or in a simulator with contact physics, is
the experiment that would close the last link, and it is now the only one
left in the chain.

**Precision estimates remain partly author-verdicted.** The independent
validation study (§4.13) is deployed and collecting, with coverage
stratified so the audit-overlap items reach several raters first. Until it
completes, the audited precision figures carry the author's own judgement,
conservatively applied and published for checking.

**One domain, one camera, six object classes.** The fitted constants are
dataset-specific by design; what transfers is the calibration procedure. The
only out-of-domain evidence is qualitative (§4.12), where unretuned
thresholds behaved correctly on stock video containing objects outside the
six classes. A labelled cross-domain sample of a few dozen images would turn
that into a measurement, and it is cheap enough that a replication should
simply include one.

**Front/behind is bounded, but less by depth than the number suggests.**
Two objects at similar camera distance cannot be ordered from one image, and
neither a larger depth model (A8) nor multi-frame geometry (A9) closes the
gap: two-view triangulation over the raw capture answers 9% of the depth
pairs and is 0.17 less accurate than the monocular cascade where it does
(§4.9, Appendix D.6). Read with the viewpoint stability of §4.14, where the predicate
reproduces its own verdict 0.955 of the time, the remaining shortfall is
mostly a disagreement about what the words mean rather than a measurement
failure, and the intervention with the best expected return is a written
annotation guideline. The measurement routes that stay open are a calibrated
stereo pair or RGB-D capture, and wider surface detection to extend the
ground-plane guard, the last of which was built, measured and declined on
this dataset (Appendix D.4).

**Detection bounds full automation.** With a zero-shot detector the
end-to-end recall is 0.38, while the relation layer conditional on detection
scores 0.85. The gap is detection, not relations, and the source paper's own
trained detector (0.93 mAP@50) would close most of it. That check needs only
the released weights.

**Scale is demonstrated without ground truth.** The supervising group
supplied the full 2,650-frame capture the released images were cut from, and
the pipeline was run over the 1,766 frames nobody has annotated: 562
keyframes after content-adaptive selection, 58 minutes, 185,242 triplets, a
predicate distribution 0.032 in total variation from the annotated portion
(§4.15). Capacity and stability on unfamiliar input are therefore measured
rather than argued. Correctness on that portion is not, and cannot be
without labels; §4.14's viewpoint-consistency check substitutes
self-agreement for truth and should be read as the weaker thing it is. The
experiment that would close this is a modest labelled sample from the
unannotated frames, a few hundred triplets, which is an afternoon of
annotation rather than a research programme.

**The capture is stereo, and only one eye was used.** The supplied folder is
named `rightimg`, which implies a left counterpart held by the supervising
group. True stereo would attack the front/behind bound directly, supplying
disparity at every frame from a known, fixed baseline, which is precisely
what the multi-frame estimators of A9 lack: those must recover the camera's
motion before they can use it, and on small low-texture objects they return
an answer for only 9% of pairs and are 0.17 less accurate than the monocular
cascade where they do (§4.9, Appendix D.6). A calibrated stereo pair removes both
problems at once and is the single cheapest experiment left on this
predicate. It also keeps the method's premise intact, since stereo is
available at capture time, whereas depth recovered from a robot walking
twenty frames is not available to a single-image annotator at all.

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

The third lesson was about my own claims, and it is the one I would most
like to have learned sooner. Reading a single benchmark run, I reported that
my labels won against the one test annotator with clean conventions. The
margin was 0.011 on 73 images. When I later retrained both arms at two more
seeds, the arms turned out to be tied, and I withdrew the claim (§6.3.1). I
had been careful about this in one place, fitting thresholds on some
annotators and validating on others, and careless about it in another,
treating one training run as a result. The discipline was not new to me; I
simply had not applied it to model training variance. Recording the
retraction in the chapter rather than quietly deleting it felt uncomfortable
and is, I think, the right thing to have done.

What I would do differently, given the time again, is front-load the
experiments that answer the question the project actually asks. I spent
considerable effort proving that the labels are correct and that they train
models well, and comparatively little on whether they help a robot decide
what to do, which is the entire motivation. The planner experiment ran, but
it ran last, and it turned out to be the clearest single result in the
dissertation: 0 of 25 against 25 of 25 needs no statistics. Had it run in
week five it would have pointed the intervening work at supply of the
support relation, which is where its only failures came from. I
would also have arranged for two people to annotate the same fifty images in
week two: that one afternoon would have produced the inter-annotator
agreement figure that the whole "comparable to human quality" claim needs,
and which I ended up estimating indirectly instead.

Technically, the thing I am most pleased with is not the accuracy figure but
the two refinements I built, measured and then declined. Both were
plausible, both took real work, and the data said neither helped. Reporting
them as null results rather than dropping them was the point at which the
project started feeling like research rather than engineering.
