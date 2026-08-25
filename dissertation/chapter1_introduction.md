# Chapter 1: Introduction

## 1.1 Background

A robot asked to pick up a book with a cube resting on it needs more than a
list of what is in the room: it needs the edge that says the cube is *on* the
book. Support, laterality, depth order and proximity are the scene-graph
edges a planner consumes to decide what to move first, and §5.7 measures the
consequence directly: given only the object list an LLM planner produces a
safe grasp plan in 0 of 25 held-out scenes, and 19 to 25 of 25 once the
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

That cost is structural and grows faster than the data. Objects relate
pairwise, so an image holding *n* annotated objects presents *n(n−1)*
ordered pairs a conscientious annotator must consider. This dataset averages
101 per image, so an exhaustive pass means a hundred judgements per
photograph. Relationship annotation is therefore never exhaustive in
practice: Visual Genome, the reference corpus, records about eighteen
relationships per image over scenes of roughly twenty objects (Krishna et
al., 2017), and here humans labelled about 10% of ordered pairs. That
sparsity is arithmetic and implies no inattention, and it has a consequence
the field absorbs without comment: a model evaluated against such labels is
rewarded for reproducing which pairs annotators happened to record as much
as which relationships hold, and Chapter 6 measures that.

A second cost is consistency. Nine annotators working independently in
batches, with no written definition of what "near" or "in front of" means,
produce nine slightly different labelling conventions. The source paper
identifies this for one predicate; Chapter 4 measures it for three, including
two annotator groups that recorded *in front of* and *behind* in opposite
directions from everyone else. Neither cost is solved by hiring more
annotators, because both scale with the number of humans involved.

### 1.1.2 Why the existing remedies do not remove it

Three families of method already exist for a shortage of labels, and each is
reviewed in Chapter 2 and tested here rather than dismissed on paper. Learned
scene-graph generators predict relations from visual patterns, but they
train on labelled triplets and so sit *downstream* of an annotator instead
of replacing one. Semi-supervised methods stretch the labels that exist,
which presupposes a consistent seed; this one is 10% dense and internally
contradictory, and Chapter 5 measures what self-training does with it. A
large vision-language model can be asked directly, the most plausible modern
shortcut and the one a reader will propose; §4.13 runs it on the same images
with the same definitions and finds it reproduces the *human* annotation's
characteristic failures, not a geometric one's. None of the three produces a
dense, self-consistent label for every ordered pair with no human deciding
anything, which is the gap this project addresses.

## 1.2 Research aim and objectives

This project removes the human from the labelling loop. The seven predicates
are *spatial*, so they are computable from geometry: given a raw RGB image
the pipeline detects objects, segments them, estimates monocular depth,
lifts each object to a 3D position and **computes** each predicate for every
ordered pair from explicit rules, writing a scene graph in the dataset's own
formats (Visual Genome JSON, YOLO txt, h5).

One distinction is central. Learned scene-graph models *predict*
relationships and therefore need labelled data; this pipeline *computes*
them from measured geometry and runs before any such model, making it the
**supplier** of what they consume rather than a competitor. Section 3.3 sets
out why that is possible for these seven predicates and what it costs.

The one predicate the authors found unreliable, *near*, is handled by
fitting a size-relative gap threshold to the human labels and reporting it.
A fitted threshold is by construction more self-consistent than nine
separate human judgements, which addresses the inconsistency the source
paper flagged.

### 1.2.1 Research questions and objectives

- **RQ1 (accuracy).** Can spatial-relationship annotation for robot images
  be automated to a quality comparable to human annotation? Measured per
  predicate against the ~900 human-labelled images.
- **RQ2 (utility).** Are the automatic labels good enough to train a
  relation-prediction model as effectively as human labels are? Measured
  with a controlled classifier trained once on each label source.

RQ1 asks whether the labels are *accurate*, RQ2 whether they are *useful*.
They decompose into six verifiable objectives:

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
the human record survive manual audit instead of turning out to be noise.
*Comparable* is given content by two references and not by a number chosen
for convenience: the trivial random and majority baselines, which any method
must beat, and an estimate of how well the human annotators would have
scored against one another, which is the ceiling any annotator can fairly be
held to (§4.6). The answer has to be given per predicate, because a mean
over seven predicates can conceal one that fails outright.

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
and explained and not resolved in the project's favour, and Chapter 6 is
where that obligation falls due.

### 1.2.3 Contributions

The deliverable is the first fully-automatic spatial-relationship annotator
for this dataset and its seven predicates, with a geometric specification of
each, a correction step that rejects impossible labels, confidence flags over
the ambiguous ones, and a fitted `near` threshold answering a limitation the
dataset's authors named. Around it sit a fidelity study with baselines and
ablations (RQ1), a controlled three-arm downstream study (RQ2), two
measurements of the dataset's own annotation process its authors did not
have, and a reliability check that needs no labels at all, obtained by
recovering the fact that the released images are consecutive frames of one
robot capture. Section 9.2 states each contribution against the evidence for
it, and says who can use it.

### 1.2.4 Scope

In scope: the automatic annotator; the fidelity study with baselines and
ablations; the controlled downstream classifier; the direct benchmark test,
in which the source paper's own SGG framework (REACT++) is trained on each
label source (Chapter 6); the planner experiment that carries the comparison
one link further towards robot behaviour (§5.7); the vision-language
baseline run on the same images under the same definitions (§4.13); and a
critical evaluation chapter. Two items entered scope during the project
rather than at its start, and both are marked as such where they are
reported. Scaling to robot captures beyond the annotated release became
possible when the supervising group supplied the full capture the release
was cut from (Appendix E.6), with the limits that follow from those frames
having no ground truth; and the vision-language comparison, originally
deferred, was brought forward once it became clear that a reader would treat
it as the obvious alternative to the whole approach. Deferred to future work
and not attempted: copy-paste augmentation of under-represented relations,
and any revision of the dataset's own predicate definitions.

**Delimitations and assumptions.** Five, each a deliberate decision, and each argued in §7.6 with the threat it carries. The work
covers **one indoor environment and six annotated object classes**, so it is
the method and not the fitted numbers that is claimed to transfer. Relations
are computed in the **camera frame**, a choice among the reference frames Chapter 2 sets out, and no fact about the world, and §4.5 measures what
it costs where an annotator chose differently. Depth is **monocular and
relative**. Fidelity is measured in the **PredCls setting**, so detection
error is held out of the comparison and reported separately (§4.11). And the
**seven predicates are taken as given**; improving their definitions would be
a different project.

## 1.3 Research approach

The project follows CRISP-DM, chosen over KDD and SEMMA for the reasons §3.1
gives. The mapping below earns its place: two findings (the
dataset's stored image orientation and the three measured annotator
behaviours) came straight out of Data Understanding, and the audit-driven
repair of the support rules is a documented iteration between Evaluation and
Modelling. That is CRISP-DM's loop, made explicit.

| CRISP-DM stage | In this project | Where |
|---|---|---|
| Business understanding | the manual-annotation bottleneck; RQ1/RQ2 | Ch. 1–2 |
| Data understanding | dataset audit: image-orientation defect, ~10% label sparsity, three measured annotator behaviours | Ch. 3–4 |
| Data preparation | orientation-corrected loading, geometry caching, native-format writers | Ch. 3 |
| Modelling | perception stack + geometric rule layer; threshold calibration; downstream classifiers | Ch. 3, 5 |
| Evaluation | fidelity protocol (baselines, ablations, audits), controlled label-source comparison, exhaustive failure attribution | Ch. 4–6 |
| Deployment | detector-in-the-loop mode, runtime/VRAM footprint, reproducibility package | Ch. 4, appendices |

Four constraints shaped the design as much as the research questions did,
and stating them makes several later choices legible. All perception runs on
a **single 6 GB consumer GPU**, which rules out the largest segmentation and
depth checkpoints and makes Chapter 3's small-model choices obligatory;
ablation A8 asks what that costs and finds almost nothing on the predicate
it was expected to hurt. There was **no budget for paid annotation**, so the independent re-estimate of precision is an unpaid volunteer study (Appendix E.3). It closed at 20 raters and carries a control arm of human-written claims, so §4.15 can report what the tool scores against what the annotators score on the same instrument; the audits around it remain the author's own, with the circularity §2.9 states as an objection before any result is reported. The project uses **one dataset**, the one
whose bottleneck the work exists to address, and the price is that
generalisation rests on argument. And the benchmark runs use **free hosted
GPU sessions**, which caps the affordable seeds and rules out a
hyper-parameter search; Chapter 6's replication is what that budget allows,
and its width is reported and not smoothed over.

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
question, and Chapter 3 gives the methodology and the geometric design of the
seven predicates. **Chapters 4 to 6 are three CRISP-DM iterations of
increasing scope on the same question**, and reading them in order is the
point: the fidelity study answers RQ1 against the human labels, the
controlled downstream study answers RQ2 against a lightweight model, and the
benchmark repeats it in the source paper's own framework and disagrees.
Chapter 7 ties all three to causes and to prior work, Chapter 8 to the legal,
social, ethical and professional dimensions of automating annotation, and
Chapter 9 to the objectives, the contributions and what is left undone.