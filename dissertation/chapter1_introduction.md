# Chapter 1: Introduction

> Chapter summary: the manual-annotation bottleneck, the compute-not-predict
> idea, the research questions and objectives, and the shape of the argument.

## 1.1 Background

A robot asked to pick up a book with a cube resting on it needs more than a
list of what is in the room: it needs the edge that says the cube is *on* the
book. Support, laterality, depth order and proximity are the scene-graph
edges a planner consumes to decide what to move first, and §5.7 measures the
consequence directly: given only the object list an LLM planner produces a
safe grasp plan in 0 of 25 held-out scenes, and 22 to 25 of 25 once the
relations are stated, depending on which source supplied them.

Learning to predict such relationships requires training data in which the
relationships are already labelled. Wang et al. (2025) introduced a spatial
relationship aware dataset captured by a Boston Dynamics Spot robot: nearly
a thousand indoor images (approximately 900 after cleaning; 838 annotated in
the released subset), annotated with seven spatial predicates: *behind, in
front of, on, to the left of, to the right of, under, near*. Every
relationship in it was labelled by hand. Nine trained annotators, working
independently in batches of 100, drew every bounding box, assigned every
class, and clicked subject then object to set each relationship, using a
manual tool (SGDET-Annotate), with a majority-vote cleaning pass.

### 1.1.1 The annotation bottleneck

Manual annotation is the bottleneck. It is slow, expensive, and does not
scale, which keeps the dataset small and limits what models trained on it
can learn. The dataset's authors themselves report that training saturates
early because of limited diversity, that the *near* predicate was
inconsistent between annotators, and that future work should "augment
under-represented relations" and adopt "spatial thresholds for near."
Crucially, **no automatic annotator exists for this dataset**.
SGDET-Annotate only accelerates manual labelling; a human still decides
every label.

The cost is structural rather than incidental, and it grows faster than the
data does. Objects in an image can be related pairwise, so an image holding
*n* annotated objects presents *n(n−1)* ordered pairs a conscientious
annotator would have to consider. This dataset averages 101 ordered pairs per
image, so a genuinely exhaustive pass would mean a hundred judgements per
photograph. Relationship annotation is therefore never exhaustive in
practice: Visual Genome, the reference corpus for the task, records about
eighteen relationships per image over scenes holding roughly twenty objects,
a small fraction of the pairs available (Krishna et al., 2017), and in the
dataset studied here humans recorded labels on about 10% of the ordered
pairs. Sparsity of this kind is not laziness but arithmetic,
and it has a consequence the field routinely absorbs without comment: a model
evaluated against such labels is rewarded for reproducing which pairs
annotators happened to record, not only which relationships actually hold.
Chapter 6 shows that this is measurable rather than theoretical.

A second cost is consistency. Nine annotators working independently in
batches, with no written definition of what "near" or "in front of" means,
produce nine slightly different labelling conventions. The source paper
identifies this for one predicate; Chapter 4 measures it for three, including
two annotator groups that recorded *in front of* and *behind* in opposite
directions from everyone else. Neither cost is solved by hiring more
annotators, because both scale with the number of humans involved.

### 1.1.2 Why the existing remedies do not remove it

Three families of method already exist for a shortage of labels, and each is
reviewed in Chapter 2 and tested somewhere in this dissertation rather than
dismissed on paper. Learned scene-graph generators predict relations from
visual patterns, but they are trained on labelled triplets and therefore sit
*downstream* of an annotator rather than replacing one. Semi-supervised
methods stretch the labels that exist, which presupposes a consistent
labelled seed; the seed here is 10% dense and internally contradictory, and
Chapter 5 measures what self-training actually does with such a seed. A
large vision-language model can be asked for the relations directly, which
is the most plausible modern shortcut and the one a reader is most likely to
propose; §4.16 runs it on the same images with the same definitions, and the
result is that it reproduces the *human* annotation's characteristic
failures rather than a geometric one's. What none of the three does is
produce a dense, self-consistent label for every ordered pair in an image
with no human deciding anything, which is the gap this project addresses.

## 1.2 Research aim and objectives

This project removes the human from the labelling loop. The seven predicates
are *spatial*, and spatial relationships are *computable from geometry*.
Given a raw RGB image, the proposed pipeline detects objects, segments them,
estimates monocular depth, lifts each object to a simple 3D position, and
then **computes** each predicate for every ordered pair of objects from
explicit geometric rules. The output is a scene graph in the dataset's own
formats (Visual Genome JSON, YOLO txt, h5).

A distinction at the heart of this work must be stated plainly. Scene-graph
generation (SGG) models such as REACT++ *predict* relationships from learned
visual patterns; they exist only because humans first labelled their
training data. This pipeline instead *computes* relationships from measured
geometry, with no human, and runs *before* any learned relation model. It is
therefore the **supplier** of the labelled data such models consume, not a
competitor to them. The perception components used (detector, segmentation,
depth) only *measure* where things are and how far away; a deterministic
rule decides the relationship. This is valid precisely because the
predicates are spatial.

The one predicate the authors found unreliable, *near*, is handled by
fitting a single proximity threshold (a size-relative gap between the two
objects) to the human labels and reporting it. A fixed, data-fitted
threshold is by construction more self-consistent than nine separate human
judgements, directly addressing the inconsistency the source paper flagged.

### 1.2.1 Research questions and objectives

- **RQ1 (accuracy).** Can spatial-relationship annotation for robot images be
  automated to a quality comparable to human annotation? Measured against the
  ~900 human-labelled images, per predicate.
- **RQ2 (utility).** Are the automatic labels good enough to train a
  relation-prediction model as effectively as human labels are? Measured with a
  controlled lightweight classifier trained once on each label source.

RQ1 asks whether the labels are *accurate*; RQ2 asks whether they are
*useful*.

The research questions decompose into six verifiable objectives:

- **O1 (build).** A fully-automatic pipeline (detection, segmentation, depth,
  geometric rules) that annotates the complete dataset in its native formats
  (VG JSON / YOLO / h5) with no human in the labelling loop. *(Chapter 3)*
- **O2 (specify and calibrate).** An operational geometric definition of all
  seven predicates, with every threshold fitted only on a subset of annotator
  groups and validated on held-out annotators. *(Chapter 3)*
- **O3 (validate).** Per-predicate fidelity against the human annotations,
  with trivial and box-only baselines, ablations, and manually audited true
  precision, answering RQ1. *(Chapter 4)*
- **O4 (diagnose).** Every disagreement with the human labels attributed to a
  cause: calibrated abstention, annotator behaviour, or genuine tool error.
  *(Chapters 4, 7)*
- **O5 (test downstream utility).** A controlled experiment in which the same
  classifier is trained on each label source under identical features,
  splits and seeds, isolating the label source, answering RQ2. *(Chapter 5)*
- **O6 (test at the level the field measures).** The same comparison repeated
  in the source paper's own scene-graph framework with a shared frozen
  detector and replicated seeds, and carried one link further to an LLM
  planner asked for a safe grasp plan under each label source, so that the
  answer to RQ2 does not rest on one lightweight model. *(Chapters 5, 6)*

### 1.2.2 What would count as an answer

Both research questions can be answered badly by choosing the measurement
after seeing the result, so the criteria are fixed here, before any of them
is reported.

RQ1 is answered **yes** if per-predicate recall of the human triplets is
comparable to what the human process itself achieves, on annotator groups
whose data influenced no threshold, and if the labels the tool emits beyond
the human record survive manual audit rather than turning out to be noise.
*Comparable* is given content by two references rather than by a number
chosen for convenience: the trivial random and majority baselines, which any
method must beat, and an estimate of how well the human annotators would
have scored against one another, which is the ceiling any annotator can
fairly be held to (§4.6). A per-predicate answer is required, not a mean,
because a mean over seven predicates can conceal one that fails outright.

RQ2 is answered **yes** if a model trained on the automatic labels performs
at least as well as the same model trained on the human labels, under
identical features, splits and seeds, and judged against held-out *human*
annotation. That direction is deliberately the harder one for the automatic
arm, because the yardstick is the rival source's own product. Since a single
lightweight model could produce such a result by accident, the question is
put three times at increasing cost, to a controlled classifier, to the
source paper's own benchmark framework, and to a planner acting on the
relations, with the standard semi-supervised remedy included as a third arm
in the controlled experiment. Agreement across all three would be required
for an unqualified yes; where they disagree, the disagreement is reported
and explained rather than resolved in the project's favour, and Chapter 6 is
where that obligation falls due.

### 1.2.3 Contributions

1. The first fully-automatic spatial-relationship annotator for this robot
   scene-graph dataset and its seven predicates.
2. A geometric specification and implementation of all seven predicates,
   including a correction step that rejects geometrically impossible labels and
   confidence flags that mark ambiguous cases for optional human review.
3. A *near* threshold fitted to the human labels, addressing a limitation the
   dataset authors explicitly named.
4. A fidelity study (RQ1) with multiple baselines and ablations, and a
   controlled downstream study (RQ2) that isolates the effect of the label
   source across three arms, including a self-training arm that tests the
   standard semi-supervised alternative to automatic labelling.
5. Two measurements of the dataset's own annotation process that its authors
   did not have: the quantified annotator behaviours of Chapter 4, and an
   estimate of how well the annotators would agree with one another, obtained
   by using the deterministic annotator as a common reference.
6. A label-free reliability check, obtained by recovering the fact that the
   released images are consecutive frames of one robot capture and asking
   whether a predicate survives the camera moving. It separates a rule that
   is wrong from one that is merely uncertain, a distinction sparse human
   annotation cannot draw, and it applies to any image dataset cut from a
   sequence.

### 1.2.4 Scope

In scope: the automatic annotator; the fidelity study with baselines and
ablations; the controlled downstream classifier; the direct benchmark test, in
which the source paper's own SGG framework (REACT++) is trained on each label
source (Chapter 6); the planner experiment that carries the comparison one
link further towards robot behaviour (§5.7); the vision-language baseline run
on the same images under the same definitions (§4.16); and a critical
evaluation chapter. Two items entered scope during the project rather than at
its start, and both are marked as such where they are reported. Scaling to
robot captures beyond the annotated release became possible when the
supervising group supplied the full capture the release was cut from (§4.15),
with the limits that follow from those frames having no ground truth; and the
vision-language comparison, originally deferred, was brought forward once it
became clear that a reader would treat it as the obvious alternative to the
whole approach. Deferred to future work and not attempted: copy-paste
augmentation of under-represented relations, and any revision of the
dataset's own predicate definitions.

**Delimitations and assumptions.** Five, each a decision rather than an
oversight, and each revisited in §7.6. The work covers **one indoor
environment and six annotated object classes**, so every fitted threshold is
dataset-specific by construction and it is the method, not the numbers, that
is claimed to transfer. Relations are computed in the **camera frame**,
which is a choice among the reference frames Chapter 2 sets out and not a
fact about the world; where an annotator used a different frame the two
disagree systematically, and §4.5 measures exactly that. Depth is
**monocular and relative**, so the depth predicates inherit an ambiguity no
threshold can remove and which ablation A8 shows a four-times-larger depth
model does not resolve. Fidelity is measured in the **PredCls setting**,
with ground-truth boxes and classes supplied, so detection error is held out
of the comparison and reported separately (§4.11). And the **seven
predicates are taken as given** from the source dataset; improving their
definitions would be a different project, and this one inherits whatever
they leave ambiguous.

## 1.3 Research approach

The project follows the CRISP-DM structure that organises data-science work
from problem understanding through to evaluation and deployment; the choice
among candidate methodologies is justified in Chapter 3. The mapping
below is descriptive, not decorative: two of the project's findings (the
dataset's stored image orientation and the three measured annotator
behaviours) came directly from the Data Understanding stage, and the
audit-driven repair of the support rules is a documented iteration between
Evaluation and Modelling. That is CRISP-DM's loop, made explicit rather than
hidden.

| CRISP-DM stage | In this project | Where |
|---|---|---|
| Business understanding | the manual-annotation bottleneck; RQ1/RQ2 | Ch. 1–2 |
| Data understanding | dataset audit: image-orientation defect, ~10% label sparsity, three measured annotator behaviours | Ch. 3–4 |
| Data preparation | orientation-corrected loading, geometry caching, native-format writers | Ch. 3 |
| Modelling | perception stack + geometric rule layer; threshold calibration; downstream classifiers | Ch. 3, 5 |
| Evaluation | fidelity protocol (baselines, ablations, audits), controlled label-source comparison, exhaustive failure attribution | Ch. 4–6 |
| Deployment | detector-in-the-loop mode, runtime/VRAM footprint, reproducibility package | Ch. 4, appendices |

The work was carried out under constraints that shaped the design as much as
the research questions did, and stating them makes several later choices
legible. All perception runs on a **single 6 GB consumer GPU**, which rules
out the largest segmentation and depth checkpoints and makes the small-model
choices of Chapter 3 a requirement rather than a preference; ablation A8
then asks what that requirement costs and finds it costs almost nothing on
the predicate it was expected to hurt. There was **no budget for paid
annotation**, so the independent re-estimate of precision is a volunteer
study (§4.13) rather than a commissioned one, and the audits that precede it
are the author's own, with the circularity that implies and that §2.9 states
as an objection before any result is reported. The project uses **one
dataset**, because it is the dataset whose annotation bottleneck the work
exists to address, and the price is that generalisation is argued rather
than demonstrated. And the benchmark chapter's training runs use **free
hosted GPU sessions**, which caps how many seeds are affordable and rules
out the hyper-parameter search a fully tuned comparison would want; the
replication reported in Chapter 6 is what that budget allows, and its width
is reported rather than smoothed over.

Ethical considerations are summarised here and detailed in Appendix A. The
work is a secondary analysis of a published, openly licensed dataset (CC-BY
4.0) collected by the supervising research group; no new personal data were
gathered for the annotation study. Some dataset frames contain identifiable
people, so faces are anonymised in every published figure. The independent
human validation of the automatic labels (Chapter 4) collects anonymous
true/false judgements only, with no names, contact details, or IP addresses,
under the University's ethics self-assessment process.

## 1.4 Dissertation outline

Chapter 2 reviews the literature with label quality as its organising
question and positions the research gap. Chapter 3 gives the research
methodology and the geometric design of the seven predicates with per-choice
justifications. Chapters 4, 5 and 6 are the analysis chapters, presented as
three iterations of increasing scope in the CRISP-DM sense: Chapter 4
presents the fidelity study (RQ1), its ablations, the independent validation
of its precision estimates, and two measurements that leave the annotated
gold behind: whether the labels survive the camera moving, and what the
pipeline does on robot frames nobody has labelled; Chapter 5 the controlled
downstream study (RQ2), including a self-training arm that tests the
standard rival remedy for scarce labels; and Chapter 6 the direct benchmark
test, in which the source paper's own SGG model is trained on each label
source and judged against three pre-registered predictions. Chapter 7 is a
critical evaluation tying all three iterations to causes and to prior work.
Chapter 8 assesses the legal, social, ethical and professional dimensions of
automating annotation. Chapter 9 concludes: it reports the objectives
against their evidence, states the contributions, turns each limitation into
the experiment that would resolve it, and reflects on how the project was
actually conducted.