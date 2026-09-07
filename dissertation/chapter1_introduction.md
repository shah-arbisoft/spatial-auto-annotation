# Chapter 1: Introduction

## 1.1 Background

Ask a robot to pick up a book with a cube sitting on it, and a list of what is
in the room will not get it there. It needs the edge saying the cube is *on*
the book, which tells it to move the cube first. Support, left/right, depth
order and closeness are the scene-graph edges carrying that kind of fact, and
§5.7 puts a number on how much they matter. Given only the object list, an LLM
planner produced a safe grasp plan in 0 of 25 held-out scenes, and with the
relations stated it managed 19 to 25 of 25, depending on where they came from.

Any model that learns to predict these relationships has to train on data
somebody has already labelled. Wang et al. (2025) published a spatial
relationship aware dataset shot by a Boston Dynamics Spot robot, nearly a
thousand indoor images, about 900 after cleaning and 838 annotated in the
released subset, over seven spatial predicates: *behind, in front of, on, to
the left of, to the right of, under, near*. All of it, every box and every
edge, was done by hand. Working on their own in batches of 100, nine trained
annotators drew every box, assigned every class, and clicked subject then
object for each relationship in a manual tool called SGDET-Annotate, after
which a majority vote cleaned the result.

### 1.1.1 The annotation bottleneck

Hand labelling is the step that holds everything else up. It is slow, somebody
has to be paid for it, and so the dataset stays small, which caps what any
model trained on it can pick up.

Wang et al. (2025) name three problems of their own. Training saturates early
for want of variety in the data, *near* was used inconsistently between
annotators, and they suggest future work should "augment under-represented
relations" and adopt "spatial thresholds for near." No automatic annotator
exists for this dataset. SGDET-Annotate was built to make the manual pass
faster, and it does. But a person still decides every label.

Scale makes it worse, and the cost grows faster than the data does. Objects
relate in pairs, so an image with *n* annotated objects gives *n(n−1)* ordered
pairs, and this one averages 101 per image, which puts complete labelling out
of reach for nine people. Visual Genome, for comparison, records about
eighteen relationships per image across scenes of roughly twenty objects
(Krishna et al., 2017), while here the annotators covered about 10% of ordered
pairs, which the pair count forces and which says nothing about how carefully
they worked. The consequence is one the field mostly passes over. A model
judged against these labels picks up credit for guessing which pairs the
annotators wrote down as much as for getting the relationships right, which
Chapter 6 measures.

Then there is consistency, which spending more cannot fix. Nine people working
on their own batches end up with nine slightly different habits, and although
Wang et al. (2025) say annotators were trained and given definitions for each
predicate, no such definition survives in the released files. Whatever was
said at the time, Chapter 4 finds the *use* of three predicates drifted apart,
with two groups recording *in front of* and *behind* the opposite way round
from everyone else. Hiring more annotators helps with neither problem, because
both scale with the number of people.

### 1.1.2 Why the existing remedies do not remove it

Three families of fix already exist for a shortage of labels, and Chapter 2
goes through each. I tested all three here, in Chapters 4 and 5.

Learned scene-graph generators read relations off visual patterns, but since
they train on labelled triplets they sit downstream of an annotator and cannot
take one's place. Semi-supervised methods stretch whatever labels already
exist, which works when the seed is clean; this seed is 10% dense and
disagrees with itself, and Chapter 5 measures what self-training does with it.
The third route is to ask a large vision-language model outright, the obvious
modern shortcut, and when §4.13 puts one on the same images with the same
definitions, what comes back repeats the *human* annotation's typical
failures, not a geometric system's. None of the three gives a dense,
self-consistent label for every ordered pair with nobody deciding anything,
and that gap is what this project fills.

## 1.2 Research aim and objectives

The point of the project, stated once and tested for the rest of it, is to
take the human out of the labelling loop, and since the seven predicates are
*spatial* they can be worked out from geometry. Given a raw RGB image, the
pipeline finds objects, segments them, estimates monocular depth, lifts each
to a 3D position and **computes** each predicate for every ordered pair from
written-out rules, then saves a scene graph in the dataset's own formats,
Visual Genome JSON, YOLO txt and h5.

One distinction runs through the dissertation. Learned models *predict*
relationships, so they need labelled data, while this pipeline *computes* them
from measured geometry and runs before any such model, which makes it the
**supplier** of what those models consume, not a competitor. Section 3.3 sets
out why it works for these seven predicates, and what it costs. For *near*,
the one predicate the authors called unreliable, I fit a size-relative gap
threshold to the human labels and report the number, since a fitted threshold
is by construction more self-consistent than nine separate human judgements.

### 1.2.1 Research questions and objectives

- **RQ1 (accuracy).** Can spatial-relationship annotation for robot images
  be automated to a quality comparable to human annotation? Measured per
  predicate against the 8,926 human-recorded relationships in the released
  subset (§4.1 fixes the counts).
- **RQ2 (utility).** Are the automatic labels good enough to train a
  relation-prediction model as effectively as human labels are? Measured
  with a controlled classifier trained once on each label source.

The first is about whether the labels are right. The second, which took most
of the work, is about whether anything useful can be trained on them, and
between them they break into six checkable objectives:

- **O1 (build).** A fully-automatic pipeline (detection, segmentation,
  depth, geometric rules) that annotates the complete dataset in its native
  formats with no human in the labelling loop. *(Chapter 3)*
- **O2 (specify and calibrate).** A working geometric definition of all
  seven predicates, with every threshold fitted only on a subset of
  annotator groups and checked on held-out annotators. *(Chapter 3)*
- **O3 (validate).** Per-predicate fidelity against the human annotations,
  with trivial and box-only baselines, ablations, and manually audited
  precision, answering RQ1. *(Chapter 4)*
- **O4 (diagnose).** Every disagreement with the human labels traced to a
  cause: calibrated abstention, annotator behaviour, or tool error.
  *(Chapters 4, 7)*
- **O5 (test downstream utility).** A controlled experiment in which the
  same classifier is trained on each label source under identical features,
  splits and seeds, isolating the label source, answering RQ2. *(Chapter 5)*
- **O6 (test at the level the field measures).** The same comparison
  repeated in a current scene-graph framework with a shared frozen detector
  and replicated seeds, then carried one step further to an LLM planner
  asked for a grasp plan under each label source and scored on whether it
  clears the occluder first, so the answer to RQ2 does not rest on one
  lightweight model. *(Chapters 5, 6)*

### 1.2.2 What would count as an answer

Picking the measurement after seeing the result proves little, so I set the
tests out here first.

RQ1 gets a **yes** if per-predicate recall of the human triplets is about as
good as the human process manages, on annotator groups whose data touched no
threshold, and if the extra labels beyond the human record survive a manual
audit instead of turning out to be noise. *About as good* needed two things to
measure against, picked in advance. Trivial random and majority baselines are
the first, since any method has to beat those. The second was an estimate of
how well two human annotators would have agreed, roughly the ceiling an
annotator can fairly be held to.

The second does not exist. Because the batches are disjoint, §4.6 finds no way
to get it from this dataset, so the test falls back on the baselines and the
per-predicate audit, and I say so where it happens. Answers are given per
predicate as well, since a mean over seven can bury one that has failed badly.

RQ2 gets a **yes** if a model trained on the automatic labels does at least as
well as the same model trained on the human labels, under the same features,
splits and seeds, scored against held-out *human* annotation. That is the
harder direction, since the measure belongs to the rival source. One
lightweight model might return that by luck, so the question is asked three
times at rising cost, through a controlled classifier, a current scene-graph
benchmark framework, and a planner acting on the relations, with the usual
semi-supervised fix as a third arm. An unqualified yes needs all three to
agree. Chapter 6 is where they do not, and the disagreement is reported and
explained, not resolved in the project's favour.

### 1.2.3 Contributions

The literature search in Chapter 2, with its search terms, turned up no system
of this kind. What is delivered is the first fully-automatic
spatial-relationship annotator for this dataset's seven predicates, with a
geometric definition of each, a correction step that throws out impossible
labels, confidence flags on the doubtful ones, and a fitted `near` threshold
answering a limitation the authors named themselves. Around it sit a fidelity
study with baselines and ablations (RQ1), a controlled three-arm downstream
study (RQ2), two measurements of the dataset's own labelling process its
authors did not have, and a reliability check needing no labels, which fell
out of noticing that the released images are consecutive frames of one robot
capture. Section 8.2 states each contribution against its evidence and says
who can use it.

### 1.2.4 Scope

In scope: the automatic annotator; the fidelity study with baselines and
ablations; the controlled downstream classifier; the direct benchmark test in
a current SGG framework (Chapter 6); the planner experiment carrying the
comparison one step closer to robot behaviour (§5.7); the vision-language
baseline (§4.13); and a critical evaluation chapter.

Two things came into scope while the project was running, both marked where
they are reported, and once the supervising group handed over the full
capture, scaling to robot images beyond the annotated release became possible
(Supplementary E.5). The vision-language comparison was pulled forward once it
was clear a reader would treat it as the obvious alternative. Left for later
work: copy-paste augmentation of under-represented relations, and any rewrite
of the dataset's own predicate definitions.

**Limits and assumptions.** Five, each argued in §7.6 with the risk it
carries. Since the work covers **one indoor environment and six annotated
object classes**, what I claim transfers is the method, not the fitted
numbers. Relations are computed in the **camera frame**, one of the reference
frames Chapter 2 sets out, not a fact about the world, and §4.5 measures what
that costs where an annotator chose differently. Depth is **monocular and
relative**. Since fidelity is measured in the **PredCls setting**, detection
error is held out and reported on its own (§4.11). And the **seven predicates
are taken as given**, because rewriting their definitions would be a different
project.

## 1.3 Research approach

The project follows CRISP-DM, picked over KDD and SEMMA for the reasons in
§3.1. Two findings came out of Data Understanding, not Modelling, the way the
images are stored on their side and the three annotator behaviours I measured,
while the audit-driven repair of the support rules is a loop between
Evaluation and Modelling that the write-up keeps.

| CRISP-DM stage | In this project | Where |
|---|---|---|
| Business understanding | the manual-annotation bottleneck; RQ1/RQ2 | Ch. 1–2 |
| Data understanding | dataset audit: image-orientation defect, ~10% label sparsity, three measured annotator behaviours | Ch. 3–4 |
| Data preparation | orientation-corrected loading, geometry caching, native-format writers | Ch. 3 |
| Modelling | perception stack + geometric rule layer; threshold calibration; downstream classifiers | Ch. 3, 5 |
| Evaluation | fidelity protocol (baselines, ablations, audits), controlled label-source comparison, exhaustive failure attribution | Ch. 4–6 |
| Deployment | detector-in-the-loop mode, runtime/VRAM footprint, reproducibility package | Ch. 4, supplementary |

Four limits shaped the design about as much as the research questions did. I
had one 6 GB consumer GPU, no money for paid annotation, one dataset, and free
hosted GPU sessions for the benchmark runs, and each ruled something out and
is answered somewhere in the evidence. Ablation A8 answers the GPU budget.
Auditing samples, in place of re-labelling at scale, answers the annotation
budget. The single dataset is answered by argument, not by a second domain,
and the GPU-hour cap by a three-seed replication whose width I report.
Supplementary B sets out what each limit excluded.

Ethics is summarised here and covered in Supplementary A. The work is a
secondary analysis of a published, openly licensed dataset (CC-BY 4.0)
collected by the supervising research group, and I gathered no new personal
data. Some frames show identifiable people, so faces are blurred in every
published figure. Nothing comes from human participants, which makes the work
secondary analysis throughout and the module's Secondary Data Checklist the
route that applies.

## 1.4 Dissertation outline

Chapter 2 covers the literature with label quality as the organising question,
and Chapter 3 gives the methodology and the geometric design of the seven
predicates. **Chapters 4 to 6 are three CRISP-DM iterations of increasing
scope on the same question**, meant to be read in order, since the fidelity
study answers RQ1 against the human labels, the controlled downstream study
answers RQ2 against a lightweight model, and the benchmark repeats it in a
current SGG framework and disagrees. Chapter 7 ties all three to causes and to
prior work, and reads the result for its social and professional consequences,
while Chapter 8 closes against the objectives, contributions and what is left
undone. Where the legal and ethical limits on the data and the models bore on
the design, they are stated in §3.12. The chapters stand on their own, and the
supplementary material after the references is background.
