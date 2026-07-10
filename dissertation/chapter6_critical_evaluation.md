# Chapter 6 — Critical Evaluation

> Every quantitative claim below is established in Chapters 4–5
> and reproducible from the repository; this chapter interprets, connects and
> stress-tests them.

## 6.1 Achievement against the research questions

**RQ1** asked whether spatial-relationship annotation can be automated at a
quality comparable to human annotation. The answer is predicate-shaped rather
than singular. For the lateral and proximity predicates the tool is, by every
measure available, *at least* as good as the human process: 0.97–0.998 recall,
audited true precision ≈ 1.0, and for `near` a perfect held-out score against
the one annotator who used the label and never influenced the threshold. For
support, after two evidence upgrades motivated by measurement (depth
co-location, mask contact), recall is 0.88/0.81 with audited precision around
0.9 — comfortably comparable. For the depth pair the cascade of relative depth
and the ground-plane fallback reaches 0.64/0.66 pooled — 0.84 once the two
inverted-convention groups are aligned — and where the tool commits it agrees
with every consistently-labelled annotator 95–100% of the time; the remaining
shortfall is calibrated abstention plus that inverted direction convention.
"Comparable to human quality" understates that case: on front/behind the tool
is more consistent than the human process it is measured against.

**RQ2** asked whether the automatic labels can train a relation model as well
as human labels. The controlled experiment answered more strongly than the
question was posed: with identical features, model, seed and split, the
auto-trained classifier reaches 0.76 mean recall against held-out *human* gold
versus 0.30 for its human-trained twin. At this dataset's annotation scale,
the automatic labels are better training material than the labels they were
validated against. The mechanism is not mysterious — density (20× more
triplets) and consistency (one definition, uniformly applied) — but it converts
the project's premise from "removing the bottleneck loses little" to "removing
the bottleneck gains".

Against the five objectives set in §1.4, each is met and evidenced: **O1** —
the pipeline annotates all 836 images in ~5 minutes with no human decision,
byte-compatible with the dataset's formats (Chapter 3); **O2** — all seven
predicates have operational geometric definitions, every fitted threshold was
calibrated on train annotators only and generalised to held-out ones (near
recall 1.0, support F1 0.87 held-out); **O3** — per-predicate fidelity is
measured against baselines and ablations with audited true precision
(Chapter 4); **O4** — all 1,689 misses are attributed to a cause, with genuine
tool error bounded at ~6% of miss mass (§6.2); **O5** — the controlled
label-source experiment is the 0.76-versus-0.30 result above (Chapter 5).

## 6.2 What the remaining failures are made of

The failure gallery diagnoses every one of the 1,689 missed human triplets by
re-checking rule conditions, so the failure analysis is exhaustive rather than
anecdotal. Three observations matter most.

First, **genuine tool error is rare**: depth-ordering mistakes are 1–5% of
front/behind misses; across all predicates, misses attributable to avoidable
error are roughly 7%. The bulk of the miss mass is calibrated abstention
(depth ambiguity band, 52–61% of front/behind misses) and measured annotator
defects (38–42% — a share that *grew* as the ground-plane fallback shrank the
abstention share around it).

Second, **the support arc shows the method working as a method**. The box rule
shipped with ~0.27 true precision; the audit localised the failure (projection
adjacency), the gallery localised the misses (containment), one geometric
insight (stacked objects share a camera distance) fixed half the false fires,
and one perception upgrade (mask-bottom contact) fixed most of the rest while
*raising* recall — each step calibrated on train annotators and validated
held-out. The residual failure mode is itself precisely characterised: a
person *holding* an object satisfies pixel contact (3 of the 7 remaining
audited errors); a class-aware guard is the documented next refinement.

Third, **what looked like a depth-resolution ceiling was mostly a rules
ceiling — and it moved**. Two objects at similar camera distance cannot be
ordered by relative monocular depth (the `depth_eps` sweep bounds that trade:
recall up to ~0.71 at ε=0 for ~0.26–0.36 precision), but the ground-plane
fallback showed that most of the abstention band is recoverable *without*
depth, from pure projection, once the tool's own contact evidence guards
against elevated objects. The residual ceiling is narrower and precisely
characterised: objects resting on supports the detector has no box for
(the fallback's audited failure mode), and pairs whose bottom edges tie
within the band. Both operating points are documented, revisable decisions
rather than hidden constants (ablations A2, A7).

## 6.3 The dataset's annotation process, examined

The source paper flagged `near` as inconsistent and called for "clear
annotation guidelines (e.g., spatial thresholds for 'near')". This project
quantifies how much further the guideline problem goes:

1. `near` was used by 3 of 9 annotator groups, with ~4× variance in how
   exhaustively equally-close pairs were labelled — yet all three annotators'
   labels sit inside one fitted threshold (held-out recall 1.0): consistent
   *notion*, non-exhaustive *application*.
2. Two annotator groups applied the **inverted direction convention** for
   in front of / behind (2–5% agreement where the tool commits; flipping
   recovers 0.93/0.71).
3. Support pairs were often labelled in **one direction only** (one group
   all-`on`, another all-`under`).
4. The official guidance — confirmed at the annotation tool's repository —
   consists of **vocabulary lists with no definitions**. Every defect above is
   the predictable consequence of labelling with undefined terms.

This reframes the evaluation itself: for several predicates there is no human
consensus to agree with, only per-annotator behaviours. The dissertation's
response — per-annotator reporting, annotator-aware calibration, and
operational definitions as the deliverable — is, to our knowledge, the first
time this dataset's label semantics have been made explicit. The "tenth
annotator" framing survives contact with the data: where annotators are
self-consistent, the tool agrees with them at 0.72–0.93 overall and 95–100% on
committed depth directions, a range that plausibly brackets what the
annotators would score against each other — though, absent overlapping
assignments, inter-annotator agreement cannot be computed directly, and that
absence is itself a finding about the dataset's construction.

## 6.4 Methodological reflection

Choices that proved right: the **PredCls isolation** (without it, every rule
result would be confounded by detection — the SGDet decomposition shows the
relation layer at 0.85 conditional mean, invisible in the 0.38 end-to-end
number); the **sparse-gold protocol** (recall-primary plus restricted
precision plus audits — the audit overturned the naive reading of restricted
precision for five predicates and confirmed it for support); and **train-only
calibration with held-out annotators** (every fitted threshold generalised:
near recall 1.0, support F1 0.87 on annotators the thresholds never saw).

Choices a stricter replication should improve: the **audits were verdicted by
the author** (conservatively, with verdicts and rendered evidence published
for spot-checking, but blind double-verdicting would be stronger); the
**support-rule iteration used the same audit machinery twice**, so the second
audit is confirmatory rather than fully independent; the SGDet **prompt/
threshold tuning used one disclosed iteration on a trial slice** that
over-estimated full-set detection quality — a small, instructive example of
trial-set optimism; and 2,000-scene invariant fuzzing pins rule consistency
but not rule *truth*, which only the audits address.

## 6.5 Synthesis against the geometry-to-label lineage

The pipeline borrows its skeleton from the SpatialVLM family — lift perception
to geometry, derive spatial facts deterministically. What this project adds is
not the skeleton but the parts the lineage leaves implicit. SpatialVLM and
VQASynth generate *training text* at internet scale and never confront a fixed
predicate vocabulary with human ground truth; SpatialRGPT curates region
representations with depth but validates downstream, not against annotators.
This project's contributions to the recipe are: **annotator-aware
calibration** (fit thresholds only on annotators who used a label; hold out
annotators, not just images), **contact as the support signature** (mask-bottom
adjacency, which the box-geometry lineage does not use and which repaired both
error directions at once), and **loss attribution as methodology** (every miss
diagnosed to a cause; every gap decomposed — abstention vs annotator vs error,
detection vs relations). RoboSpatial's reference-frame taxonomy, cited in the
design chapter to justify camera-frame laterality, turned out to be the right
lens for a *measured* phenomenon: the front/behind convention inversion is a
reference-frame disagreement inside a single dataset's annotation team.

## 6.6 Limitations and threats to validity

**Internal.** Thresholds are fitted to six annotator groups of one dataset;
audits are author-verdicted (§6.4); the two-stage audit shares machinery with
the rule change it evaluates.

**External.** One laboratory domain, six object classes, one camera and
mounting; the fitted constants (`near_T`, ε values, contact threshold) are
dataset-specific by design — the *procedure* (fit on some annotators, validate
on held-out ones) is the transferable artefact, not the numbers. Full
automation is currently detection-bounded (0.38 end-to-end with a worst-case
zero-shot detector; the authors' trained detector would close most of that
gap, but this remains unverified without their weights).

**Construct.** "Agreement with human labels" is an imperfect proxy when the
humans disagree with each other; the per-annotator decompositions mitigate but
cannot eliminate this. The RQ2 result compares supervision *at this dataset's
annotation scale* — it does not claim automatic labels beat abundant,
guideline-driven human annotation, a regime this dataset does not contain.

**Ethics.** Scene images contain identifiable people; figures for publication
use the dataset as released (CC-BY 4.0) with faces blurred as a courtesy.

## 6.7 Aims, revisited

Both research questions are answered with evidence that survived held-out
validation, independent audits and exhaustive failure diagnosis. The
unexpected result is the strongest: the project set out to show automatic
labels are *not much worse* than human ones, and found conditions — sparse,
guideline-free human annotation, which is exactly the regime the source
dataset occupies — under which they are decisively better for downstream
learning. The scaling claim now has teeth beyond throughput: 836 images were
labelled in five minutes with 20× the human label density, and a model trained
on those labels reaches two and a half times the downstream recall of one
trained on the original annotations. The bottleneck this project removes was
not only slowing the dataset down; it was limiting what the dataset could
teach.
