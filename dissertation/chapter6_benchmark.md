# Chapter 6: The Direct Benchmark Test: Training REACT++ on Both Label Sources

This chapter is the third analysis iteration: the heavyweight version of the
RQ2 experiment, run in the source paper's own framework against three
predictions registered before it (§6.1). It reaches a different verdict from
Chapter 5, and §6.4 decomposes why.

## 6.1 Design

Chapter 5's controlled experiment used a deliberately small classifier; this
chapter runs the heavyweight version on the source paper's own terms: a
real-time scene-graph model (REACT++, the current model of the SGG-Benchmark
framework the dataset's tooling belongs to; Neau and Falomir, 2026) trained
once on the human relations and once on this tool's, evaluated on an
identical held-out test set. The isolation mirrors RQ2 exactly:

- **Shared detector.** A YOLOv8m backbone is trained once on the ground-truth
  boxes of the training split and frozen for both arms (test-time detection
  mAP is bit-identical between arms: 0.654). Only the relation labels differ.
- **Split** follows the calibration protocol: train = groups 0–4 (500 images;
  5,421 human vs 119,020 automatic relations; the 22× density difference is
  the treatment), validation = group 5, test = groups 6–8 (human gold in both
  arms; the framework's loader retains 210 of the 236 test images).
- **Three predictions were registered in §5.6 before the run**: (1) the
  human-label arm saturates early, replicating the source paper; (2) the
  auto-label arm reaches a higher plateau; (3) `near` recovers.

## 6.2 Training dynamics: prediction 1 confirmed

The human arm **peaks at epoch 4 of 25** and never improves again,
oscillating between 0.101 and 0.123 for the remaining twenty-one — mild
overfitting on 5,421 sparse triplets. That replicates the source paper's
central observation ("all predictors reached their peak mR@100 well before
the final epoch") with a current model, and adds the cause: sparse
supervision is exhausted early. The auto arm is still climbing at epoch 8,
does not reach 95% of its best until **epoch 14**, peaks at **epoch 22**,
and sees 213 distinct triplet types in training against the human arm's 94.
{{fig:sgg-training-curves}} plots both; each validation series uses its own
arm's labels, so only the *shapes* compare.

## 6.3 Test results: prediction 2 unresolved, prediction 3 refuted

Both best checkpoints evaluated on the identical test set (210 images,
human gold, groups 6–8), at seed 42:

| metric (test, sgdet) | human-trained | auto-trained |
|---|---|---|
| R@100 | **0.280** | 0.259 |
| mR@100 | 0.270 | **0.291** |
| F1@100 | **0.275** | 0.274 |
| zR@100, recall on types the shared reference omits (§6.4) | 0.073 | **0.309** |

At this seed the automatic arm leads mean recall, 0.291 against 0.270, the
human arm keeps raw recall, and the two are level on F1. The ordering
inverts predicate by predicate: the human arm is ahead on five of the seven,
the automatic arm on the support pair, by 0.077 on `on` and 0.284 on
`under`. For `near` both arms sit far below the source paper's 0.22–0.25
floor, 0.117 and 0.108, so prediction 3 is refuted as stated: the automatic
arm's dense, fitted `near` does not survive the trip through a learned
model. The sharpest separation is on triplet *types never seen in training*,
0.309 against 0.073, fourfold at this seed and fivefold over three (§6.3.1);
§6.4 sets out what that column measures here — coverage of relation types
the manual annotation never recorded, not compositional generalisation.
Prediction 2 is the one a single run cannot carry in either direction, the
margin sitting well inside the human arm's own seed spread, which is why
§6.3.1 exists. Appendix F.1 carries the per-predicate and per-slice
breakdown; aligning the front/behind convention of groups 6 and 8, one
disclosed bit per group as in §4.5, lifts both arms and leaves them level,
0.310 against 0.312.

### 6.3.1 Replication across seeds, and one claim withdrawn

Two of those margins are too small for one run to separate from training
noise. Both arms were retrained at seeds 43 and 44 and all six checkpoints
re-scored on every slice (`scripts/kaggle/`, aggregated by
`eval/seed_stats.py`) against the same frozen detector, so the spread below
is the relation model's own variance.

| slice | metric | conv. | human-trained | auto-trained | vision-language |
|---|---|---|---|---|---|
| full test | mR@100 | mixed | 0.293 (0.270–0.322) | 0.292 (0.290–0.296) | **0.329** (0.312–0.357) |
| full test | zR@100 | mixed | 0.052 (0.004–0.079) | 0.268 (0.225–0.309) | **0.307** (0.256–0.371) |
| group 6 | mR@100 | **inv.** | 0.327 (0.299–0.364) | 0.305 (0.296–0.312) | **0.337** (0.316–0.366) |
| group 6 | zR@100 | inv. | 0.000 (0.000–0.000) | **0.449** (0.348–0.530) | 0.202 (0.086–0.343) |
| group 7 | mR@100 | clean | 0.278 (0.254–0.298) | 0.289 (0.280–0.300) | **0.362** (0.357–0.364) |
| group 7 | zR@100 | clean | 0.098 (0.008–0.148) | 0.273 (0.234–0.346) | **0.456** (0.433–0.496) |
| group 8 | mR@100 | **inv.** | 0.147 (0.130–0.164) | 0.120 (0.108–0.131) | **0.164** (0.146–0.183) |
| group 8 | zR@100 | inv. | 0.007 (0.000–0.013) | 0.041 (0.033–0.048) | **0.090** (0.038–0.117) |
| aligned | mR@100 | corrected | 0.333 (0.310–0.369) | 0.316 (0.312–0.320) | **0.369** (0.354–0.390) |

All nine runs were trained in one session on one clone, one frozen detector
and one configuration, so the arms differ in their labels and nothing else.
An earlier set of figures, with the human arm at 0.326 pooled, came from
runs made weeks apart against different states of the upstream code; §7.4
reports what that cost. The **aligned** row re-scores the same models
against gold with the two inverted annotators corrected (§4.5); the
`zR@100` rows measure coverage of types omitted from one shared reference
(§6.4), and this dissertation makes no zero-shot claim on them. Cells give
the mean over three seeds with the per-seed range in brackets.

**On the metric this chapter is organised around, the two label sources are
indistinguishable.** The pooled mR@100 rows are level and no slice separates
them: every row's seed ranges overlap, the automatic arm's sitting *inside*
the human arm's on the full test set, and paired by seed the automatic arm
leads at 42 (+0.021) and 43 (+0.010) and trails at 44 (−0.032), so the
pooled difference of 0.001 rests on a single run. The correct statement is
parity, and it is about this metric, not the labels: on raw R@100 the human
arm keeps a real margin pooled, 0.295 against 0.255, and on zero-shot recall
the automatic arm leads fivefold, 0.268 against 0.052. The arms differ far
more in stability than in score: across seeds the automatic arm's pooled
mR@100 spans 0.006 against the human arm's 0.052 and the vision-language
arm's 0.044 — one definition applied uniformly lands in the same place
whatever the initialisation, nine annotators applying nine conventions do
not — and a margin of 0.001 between arms whose own seeds move by 0.052 is
not a result in either direction.

Where the arms do differ, they differ by annotator: the human arm leads on
the two annotators §4.5 convicts of inverting the front/behind convention
and *trails* on group 7, the one convicted of nothing. The ordering runs
with annotation quality and not with geometry, and Appendix F.9 gives the
three margins, none of them separable across seeds. That contamination is
large and measurable — **30% of the entire yardstick is
front/behind written by the two inverted annotators** — but it is a shared
ceiling rather than a differential, since both arms train where no inversion
is measured; Appendix F.9 gives the counts, what the ceiling does to each
metric, and the replication at the earlier `on_contact_min` 0.60 under which
the same ordering holds, so the pattern is a property of the test annotation
and not of one configuration of the tool. Every absolute figure in this
chapter is therefore a lower bound on both sides. What difference remains is
a separate effect, and §6.4 localises it to annotator *selection* rather
than convention: it appears on the lateral predicates too, which have no
direction to invert.

Second, **the zero-shot result is robust and larger than first reported.**
Pooled zR@100 is 0.268 against 0.052, roughly fivefold, with disjoint
ranges, and not an artefact of pooling: the auto arm leads on *every*
annotator separately (0.449 against 0.000 on group 6, 0.273 against 0.098 on
group 7, 0.041 against 0.007 on group 8), and the gap points the same way on
defective and clean annotators alike — what distinguishes a property of the
labels from a property of the gold.

### 6.3.2 A third label source

Chapter 5's vision-language labels went through the same benchmark as a
third arm: three seeds, same frozen detector, same human test gold, same
session. Pooled it leads both at 0.329 against 0.293 human and 0.292 auto,
and it is the only arm whose lead over the human one is consistent across
every slice — what §6.4's argument predicts, since §4.13 measures this model
annotating sparsely and human-like, so a metric rewarding resemblance to the
manual pass should reward it more than the manual pass rewards itself. The
sharpest form is on group 7, the one test annotator with no measured defect
and therefore the cleanest gold, where it reaches **0.362 against 0.278
human and 0.289 auto**, its per-seed range (0.357–0.364) touching neither. A
win on the cleanest annotator cannot be attributed to matching a defect, and
it is the one result in this chapter that no argument here explains away.
Appendix F.4 gives the per-slice table and the three readings the experiment
cannot separate.

## 6.4 Why the advantage disappears between Chapter 5 and this chapter

The same labels, the same held-out human gold, two very different verdicts:
per-pair recall says the automatic labels teach two and a half times better,
0.748 against 0.297; ranked evaluation against sparse gold says they are
level. Neither the labels nor the gold differ between the experiments, so an
advantage that size can only have evaporated into the **structure of the
metric**. (An earlier version of this chapter had a stronger claim to
explain — the human arm ahead at 0.326 against 0.278 — but those arms came
from runs made weeks apart against different states of the upstream
framework, and retraining all nine in one session removed the gap almost
entirely, the human arm falling 0.033 while the vision-language arm moved
0.001 (§7.4). The mechanisms below explained a reversal; they now explain
the erasure of a 2.5× advantage, which the evidence supports more
comfortably.)

**(i) Ranking dilution by true-but-unlabelled predictions.** R@K is a
per-image ranking budget: any prediction absent from the gold consumes
budget as a miss. The auto-trained model predicts densely, like its
supervision, and Chapter 4's audits established that such extras are
overwhelmingly *true*, so against gold annotating ~10% of pairs they outrank
the annotated ones and are scored as errors — §4.3's restricted-precision
artefact reproduced at benchmark level. The human-trained arm learned the
annotators' *labelling prior* instead, which is what a ranking metric
against human-selected gold rewards.

**(ii) Convention mismatch is a shared penalty, not, as first hypothesised,
a differential one.** Both arms were trained on consistent-convention
front/behind (the tool's by construction; groups 0–4's by measurement),
while groups 6 and 8, two thirds of the test gold, invert it (§4.5).
Re-scoring against aligned gold lifts both arms almost equally (+0.041
human, +0.035 auto; the human arm's *in front of* recall jumps 0.124 →
0.386, the auto arm's 0.101 → 0.248), so both pay the same tax, the gap
barely moves, and the initial hypothesis that the denser arm is punished
*harder* for its confidence is refuted and withdrawn. It replicates at the
shipped threshold — the retrained auto arm of §4.14 moves from 0.292 to
**0.316** mR@100 and 0.255 to **0.292** R@100 against aligned gold, +0.025
against the earlier +0.035 — so two label sets from two values of
`on_contact_min` pay a tax of the same order, and roughly a tenth of the
absolute mR@100 anywhere in this chapter is an artefact of that defect, on
both sides.

**(ii′) Where the gap actually lives: the two defective test groups.** The
per-group figures of §6.3.1 localise the human arm's lead to the two
annotators convicted of convention inversion, where it is four times what it
is on the clean one, and Appendix F.9 carries the fingerprint that separates
selection from convention: on group 6's *lateral* gold, which has no
direction to invert, the human arm recalls 0.49/0.69 against the auto arm's
0.12/0.21, because it learned which pairs that annotator chose to record.

**(iii) `near` gold is a single idiosyncratic annotator.** All 93 test
`near` labels come from group 8, whose usage is sparse and non-exhaustive
(§3.8). Neither arm can rank exactly those pairs highly; prediction 3
underestimated not the labels but the gold.

The zero-shot column does not measure what its name implies here. zR@100 is
recall on triplet types absent from a model's *own* training data, but both
arms are scored against one shared reference, the human training annotation.
Of the 25 test triplet types that annotation omits, the human arm saw none
in training and the auto arm saw 24, so its 0.172 against 0.003 records that
its labels *cover* relation types the manual pass never recorded — the
property the annotation bottleneck predicts, not compositional
generalisation, and this dissertation does not claim it as such; §6.5 weighs
what the column does establish.

## 6.5 What survives, read both ways

Two interpretations survive and neither is available without the other. The
benchmark result is real: a consumer *evaluated against human-annotated
scene graphs* is better supervised by human labels, which carry the
annotation prior the evaluation shares. The interpretation is equally real:
the ranking metric inherits every defect measured in the gold, and the
advantage is concentrated exactly where the annotation is defective and
absent where it is not, which is what annotation-prior agreement looks like.

Section 1.2.2 set a non-inferiority criterion — *at least as well*, not a
win — so parity is the shape a pass takes; but the numbers refuse a strong
claim in either direction. The paired mean difference is -0.0006 with a 95%
interval of [-0.070, +0.069], three seeds bound the gap to about a quarter
of the metric's own value, and a margin of ±0.01 would need roughly forty
runs per arm. The experiment establishes neither superiority nor
equivalence, and a reader entitled to say the automatic labels did not beat
the human ones is equally entitled to say this design could not have shown
it if they had. **What the same three seeds do resolve** is the part that is
not a null: zero-shot recall separates with disjoint ranges, 0.225–0.309
against 0.004–0.079, as does reproducibility, a 0.006 spread against 0.052,
both running the automatic arm's way. Against the cost of obtaining them —
nine annotators against five minutes on one consumer GPU —
indistinguishability on the ranking metric is close to the result the
project set out to obtain: RQ1 and RQ2 ask whether the human can be removed,
not whether the machine wins.

The critical reading is not novel to this project, which is what makes it
credible: Neural Motifs (Zellers et al., 2018) established that a frequency
baseline ignoring the image is hard to beat, Unbiased SGG (Tang et al.,
2020) formalised how thoroughly such models absorb the annotation
distribution, and Northcutt, Athalye and Mueller (2021) showed erroneous
test labels reorder rankings across ten benchmarks. What this chapter adds
is a case where the confound is *isolated by construction* — the arms differ
only in label source, share a frozen detector, and the per-annotator defects
were measured beforehand — so the advantage is attributed to annotation
practice, not inferred. The one remaining instrument is a manual audit of
the auto arm's top-ranked "false positives", the analogue of §4.4, left as
designed follow-up.

## 6.6 What Chapter 5's predictions got right and wrong

Registered before the run, judged after: prediction 1 (early human-arm
saturation) is **confirmed** and replicates the source paper. Prediction 2
(higher plateau) is **unresolved on mR@100**, and the word matters: an
earlier version of this chapter recorded it as refuted on the strength of a
0.048 gap that the retrained arms of §6.3.1 reduce to 0.001, which no
experiment of this size can call either way; where the plateau *is* higher
is the zero-shot component the prediction did not name, 0.268 against 0.052.
Prediction 3 (`near` recovery) is **refuted**: it wrongly assumed the test
gold could reward dense `near` prediction. The value of pre-registration is
that these verdicts are checkable, and that one had to be revised when the
measurement improved is a point in its favour. The replication adds a fourth
verdict, on a claim made *after* the run: §6.3.1 withdraws the single-seed
group-7 result on the record, a difference being worth naming only once its
size is compared with the variation of the procedure that produced it.

## 6.7 Answer, at the level the source paper measures

At the source paper's level of robot-readiness, automatic labels do not beat
human ones on the headline metric, 0.292 against 0.293; what they do is
train a model that trains longer, covers five times more of the relation
types the manual annotation never recorded, reproduces across seeds more
than eight times more tightly, and ranks slightly ahead on the one test
annotator with no measured defect. The evidence therefore supports a
conditional claim: **automatic labels are the better training material
wherever ground truth means geometric consistency, and do not overtake human
labels wherever it means annotation habits.** The first condition is the
operative one for a robot, which needs relations that are *correct* before
they are *human-like*, and §5.7 tests that one link further down, where it
survives. Chapter 7 takes the three iterations together and asks what they
mean: what the remaining failures are made of, what they say about the
dataset's own annotation process, and how far the objections raised in
advance survive the evidence.
