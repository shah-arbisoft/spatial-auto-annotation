# Chapter 1: Introduction

> Chapter summary: the manual-annotation bottleneck, the compute-not-predict
> idea, the research questions and objectives, and the shape of the argument.

## 1.1 Background

For a robot to act usefully in a human environment it must understand not just
*what* objects are present but *how they are spatially related*: that a cup is
*on* a box, a book is *to the left of* a bottle, a person is *in front of* a
shelf. These spatial relationships are the substrate of scene understanding,
manipulation planning, and instruction following. The structured representation
that captures them is a **scene graph**: objects as nodes, spatial relationships
as labelled edges.

Learning to predict such relationships requires training data in which the
relationships are already labelled. Wang et al. (2025) introduced a spatial
relationship aware dataset captured by a Boston Dynamics Spot robot: nearly a
thousand indoor images (approximately 900 after cleaning; 838 annotated in the
released subset), annotated with seven spatial predicates: *behind, in front of,
on, to the left of, to the right of, under, near*. Every relationship in it was
labelled by hand. Nine trained annotators, working independently in batches of
100, drew every bounding box, assigned every class, and clicked subject then
object to set each relationship, using a manual tool (SGDET-Annotate), with a
majority-vote cleaning pass.

### 1.1.1 The annotation bottleneck

Manual annotation is the bottleneck. It is slow, expensive, and does not scale,
which keeps the dataset small and limits what models trained on it can learn.
The dataset's authors themselves report that training saturates early because of
limited diversity, that the *near* predicate was inconsistent between annotators,
and that future work should "augment under-represented relations" and adopt
"spatial thresholds for near." Crucially, **no automatic annotator exists for
this dataset**. SGDET-Annotate only accelerates manual labelling; a human still
decides every label.

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

## 1.2 Research aim and objectives

This project removes the human from the labelling loop. The seven predicates are
*spatial*, and spatial relationships are *computable from geometry*. Given a raw
RGB image, the proposed pipeline detects objects, segments them, estimates
monocular depth, lifts each object to a simple 3D position, and then **computes**
each predicate for every ordered pair of objects from explicit geometric rules.
The output is a scene graph in the dataset's own formats (Visual Genome JSON,
YOLO txt, h5).

A distinction at the heart of this work must be stated plainly. Scene-graph
generation (SGG) models such as REACT++ *predict* relationships from learned
visual patterns; they exist only because humans first labelled their training
data. This pipeline instead *computes* relationships from measured geometry, with
no human, and runs *before* any learned relation model. It is therefore the
**supplier** of the labelled data such models consume, not a competitor to them.
The perception components used (detector, segmentation, depth) only *measure*
where things are and how far away; a deterministic rule decides the relationship.
This is valid precisely because the predicates are spatial.

The one predicate the authors found unreliable, *near*, is handled by fitting a
single proximity threshold (a size-relative gap between the two objects) to
the human labels and reporting it. A fixed, data-fitted threshold is by
construction more self-consistent than nine separate human judgements, directly
addressing the inconsistency the source paper flagged.

### 1.2.1 Research questions and objectives

- **RQ1 (accuracy).** Can spatial-relationship annotation for robot images be
  automated to a quality comparable to human annotation? Measured against the
  ~900 human-labelled images, per predicate.
- **RQ2 (utility).** Are the automatic labels good enough to train a
  relation-prediction model as effectively as human labels are? Measured with a
  controlled lightweight classifier trained once on each label source.

RQ1 asks whether the labels are *accurate*; RQ2 asks whether they are *useful*.

The research questions decompose into five verifiable objectives:

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
  classifier is trained once on human and once on automatic labels, isolating
  the label source, answering RQ2. *(Chapter 5)*

### 1.2.2 Contributions

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

### 1.2.3 Scope

In scope: the automatic annotator; the fidelity study with baselines and
ablations; the controlled downstream classifier; the direct benchmark test, in
which the source paper's own SGG framework (REACT++) is trained on each label
source (Chapter 6); and a critical evaluation chapter. Scaling to robot
captures beyond the annotated release was planned as future work and became
possible mid-project when the supervising group supplied the full capture the
release was cut from; §4.15 reports it, with the limits that follow from
those frames having no ground truth. Deferred to future work: a
vision-language task-planning comparison and copy-paste augmentation. These
protect the timeline and strengthen the future-work discussion rather than
weakening the contribution.

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
pipeline does on robot frames nobody has labelled; Chapter 5 the controlled downstream
study (RQ2), including a self-training arm that tests the standard rival
remedy for scarce labels; and Chapter 6 the direct benchmark test, in which
the source paper's own SGG model is trained on each label source and judged
against three pre-registered predictions. Chapter 7 is a critical evaluation
tying all three iterations to causes and to prior work. Chapter 8 assesses the
legal, social, ethical and professional dimensions of automating annotation.
Chapter 9 concludes: it reports the objectives against their evidence, states
the contributions, turns each limitation into the experiment that would
resolve it, and reflects on how the project was actually conducted.
