# Chapter 1: Introduction

> Chapter summary: the manual-annotation bottleneck, the compute-not-predict
> idea, the research questions and objectives, and the shape of the argument.

## 1.1 Motivation

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

## 1.2 Problem statement

Manual annotation is the bottleneck. It is slow, expensive, and does not scale,
which keeps the dataset small and limits what models trained on it can learn.
The dataset's authors themselves report that training saturates early because of
limited diversity, that the *near* predicate was inconsistent between annotators,
and that future work should "augment under-represented relations" and adopt
"spatial thresholds for near." Crucially, **no automatic annotator exists for
this dataset**. SGDET-Annotate only accelerates manual labelling; a human still
decides every label.

## 1.3 Key idea and approach

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

## 1.4 Research questions

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

## 1.5 Contributions

1. The first fully-automatic spatial-relationship annotator for this robot
   scene-graph dataset and its seven predicates.
2. A geometric specification and implementation of all seven predicates,
   including a correction step that rejects geometrically impossible labels and
   confidence flags that mark ambiguous cases for optional human review.
3. A *near* threshold fitted to the human labels, addressing a limitation the
   dataset authors explicitly named.
4. A fidelity study (RQ1) with multiple baselines and ablations, and a controlled
   downstream study (RQ2) that isolates the effect of the label source.

## 1.6 Scope

In scope: the automatic annotator; the fidelity study with baselines and
ablations; the controlled downstream classifier; the direct benchmark test, in
which the source paper's own SGG framework (REACT++) is trained on each label
source (Chapter 6); a critical evaluation chapter. Deferred to future work: a
vision-language task-planning comparison, scaling on new robot captures, and
copy-paste augmentation. These protect the timeline and strengthen the
future-work discussion rather than weakening the contribution.

## 1.7 Methodology framing and ethics

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

## 1.8 Dissertation structure

Chapter 2 reviews the literature with label quality as its organising
question and positions the research gap. Chapter 3 gives the research
methodology and the geometric design of the seven predicates with per-choice
justifications. Chapters 4, 5 and 6 are the analysis chapters, presented as
three iterations of increasing scope in the CRISP-DM sense: Chapter 4
presents the fidelity study (RQ1), its ablations and the independent
validation of its precision estimates; Chapter 5 the controlled downstream
study (RQ2), including a self-training arm that tests the standard rival
remedy for scarce labels; and Chapter 6 the direct benchmark test, in which
the source paper's own SGG model is trained on each label source and judged
against three pre-registered predictions. Chapter 7 is a critical evaluation
tying all three iterations to causes and to prior work. Chapter 8 assesses the
legal, social, ethical and professional dimensions of automating annotation.
Chapter 9 concludes and sets out future work.
