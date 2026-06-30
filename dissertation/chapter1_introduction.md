# Chapter 1 — Introduction

> Draft, Week 1. Write-as-you-go. Prose to be tightened in Weeks 6–8; the
> argument and structure are the point at this stage.

## 1.1 Motivation

For a robot to act usefully in a human environment it must understand not just
*what* objects are present but *how they are spatially related* — that a cup is
*on* a box, a book is *to the left of* a bottle, a person is *in front of* a
shelf. These spatial relationships are the substrate of scene understanding,
manipulation planning, and instruction following. The structured representation
that captures them is a **scene graph**: objects as nodes, spatial relationships
as labelled edges.

Learning to predict such relationships requires training data in which the
relationships are already labelled. Wang et al. (2025) introduced a spatial
relationship aware dataset captured by a Boston Dynamics Spot robot — nearly a
thousand indoor images (approximately 900 after cleaning; 838 annotated in the
released subset), annotated with seven spatial predicates: *behind, in front of,
on, to the left of, to the right of, under, near*. Every relationship in it was
labelled by hand: nine trained annotators, working independently in batches of
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
this dataset** — SGDET-Annotate only accelerates manual labelling; a human still
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
single distance threshold to the human labels and reporting it. A fixed,
data-fitted threshold is by construction more self-consistent than nine separate
human judgements, directly addressing the inconsistency the source paper flagged.

## 1.4 Research questions

- **RQ1 (accuracy).** Can spatial-relationship annotation for robot images be
  automated to a quality comparable to human annotation? Measured against the
  ~900 human-labelled images, per predicate.
- **RQ2 (utility).** Are the automatic labels good enough to train a
  relation-prediction model as effectively as human labels are? Measured with a
  controlled lightweight classifier trained once on each label source.

RQ1 asks whether the labels are *accurate*; RQ2 asks whether they are *useful*.

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
ablations; the controlled downstream classifier; a critical evaluation chapter.
Deferred to future work: retraining a full SGG model (REACT++), a
vision-language task-planning comparison, scaling on new robot captures, and
copy-paste augmentation. These protect the timeline and strengthen the
future-work discussion rather than weakening the contribution.

## 1.7 Dissertation structure

Chapter 2 reviews the automatic-annotation lineage and positions the gap.
Chapter 3 gives the problem analysis and the geometric design of the seven
predicates with per-choice justifications. Chapter 4 presents the fidelity study
(RQ1) and ablations. Chapter 5 presents the downstream study (RQ2). Chapter 6 is
a critical evaluation tying results to causes and to prior work. Chapter 7
concludes and sets out future work.
