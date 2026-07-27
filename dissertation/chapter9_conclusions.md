# Chapter 9: Conclusions and Future Work

This chapter closes the dissertation. Section 9.1 states how each objective
was met and what the two research questions were answered with, §9.2 sets
out the contributions and who can use them, §9.3 states the limitations
honestly and turns each into the work that would resolve it, and §9.4 is a
personal reflection on how the project was actually conducted.

## 9.1 Summary: the objectives and the answers

The aim was to determine whether spatial-relationship annotation for
robot-acquired images can be automated, and whether the resulting labels are
good enough to replace the human ones. Five objectives carried that aim, and
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
against three baselines, with eight ablations and manually audited precision
(Chapter 4): mean recall 0.85, and 0.76 on annotators no threshold ever saw.
Cluster-bootstrap intervals accompany every headline figure, and an
independent validation study re-estimates precision with judges who have no
stake in the result (§4.13).

**O4, diagnose.** Every one of the 1,689 missed human triplets is attributed
to a cause by re-checking rule conditions, so the failure analysis is
exhaustive rather than illustrative. Genuine tool error accounts for
roughly 7% of misses; the remainder is calibrated abstention and measured
annotator behaviour (§4.10, §7.2).

**O5, test downstream utility.** A controlled experiment trains the same
classifier on three label sources under identical conditions (Chapter 5),
and the comparison is repeated in the source paper's own benchmark
framework with a shared frozen detector and three seeds per arm (Chapter 6).

**RQ1, accuracy.** Yes, with one qualification. On five of seven predicates
the automatic labels match or exceed the human process (0.81 to 1.00 recall,
audited precision ≈ 1.0), and `near` is solved outright, resolving the one
predicate the source paper reports as failing for every model it
benchmarked. The depth pair reaches 0.64/0.66 pooled and 0.84 once two
annotator groups' inverted conventions are accounted for; where the tool
commits it agrees with every consistently-labelled annotator 95 to 100% of
the time. The residual limit is monocular ambiguity, which ablation A8 shows
a four-times-larger depth model does not fix.

**RQ2, utility.** Yes, and by a wide margin at this dataset's annotation
scale. The auto-trained classifier reaches 0.76 mean recall against held-out
human gold versus 0.30 for its human-trained twin, and versus 0.36 when the
human labels are stretched by self-training, the standard semi-supervised
remedy. In the full benchmark the verdict is conditional and the condition
is legible: over three seeds per arm the human-trained model ranks better
against human test annotation (mR@100 0.326 against 0.278), the auto-trained
model generalises about sixty times better to unseen relation compositions
(zR@100 0.172 against 0.003, with disjoint per-seed ranges), and the human
advantage is concentrated entirely on the two test annotators carrying
measured labelling defects, disappearing against the only one without
(0.308 against 0.307).

The single sentence the evidence supports: **automatic labels are the better
supervision wherever ground truth means geometry; human labels remain better
wherever ground truth means human annotation practice.** Robot planning
needs the former.

## 9.2 Contributions

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
model that generalises to unseen compositions is the one trained on
consistent computed labels. The per-annotator decomposition localises the
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
precision; exhaustive loss attribution; and a way to estimate what
annotators would score against one another when they never labelled the same
images, by using a deterministic annotator as a fixed common reference
(§4.6).

## 9.3 Limitations, and the work that would resolve them

Each limitation below is stated with the specific experiment that would
settle it, because that is more useful than an apology.

**The chain stops short of the robot.** The strongest evidence reaches
label quality and model quality; no experiment in this dissertation shows a
robot completing more tasks. The instrument exists: 75 prompts across 25
held-out scenes in three conditions (no relations, human relations,
automatic relations), with a blind scoring sheet, built and ready to run.
This is the single most valuable next experiment, and it is the one that
would convert a supported inference into a demonstration.

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

**Front/behind is bounded by monocular ambiguity.** Two objects at similar
camera distance cannot be ordered from one image, and a larger depth model
does not help (A8). The routes out are additional evidence rather than
better models: stereo or RGB-D capture, multi-frame structure, or wider
surface detection to extend the ground-plane guard, the last of which was
built, measured and declined on this dataset (§4.9.4).

**Detection bounds full automation.** With a zero-shot detector the
end-to-end recall is 0.38, while the relation layer conditional on detection
scores 0.85. The gap is detection, not relations, and the source paper's own
trained detector (0.93 mAP@50) would close most of it. That check needs only
the released weights.

**Scale is demonstrated in principle, not in practice.** Throughput and
density are measured on 836 images. Applying the pipeline to genuinely new
robot captures, which the supervising group can supply, is the natural next
step and the one that would make the scaling claim concrete rather than
extrapolated.

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
what to do, which is the entire motivation. The planner experiment should
have run in week five rather than sitting built and unrun at the end. I
would also have arranged for two people to annotate the same fifty images in
week two: that one afternoon would have produced the inter-annotator
agreement figure that the whole "comparable to human quality" claim needs,
and which I ended up estimating indirectly instead.

Technically, the thing I am most pleased with is not the accuracy figure but
the two refinements I built, measured and then declined. Both were
plausible, both took real work, and the data said neither helped. Reporting
them as null results rather than dropping them was the point at which the
project started feeling like research rather than engineering.
