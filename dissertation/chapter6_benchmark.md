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
once on the human relations and once on this tool's, and evaluated on an
identical held-out test set. The isolation mirrors RQ2 exactly:

- **Shared detector.** A YOLOv8m backbone is trained once on the ground-truth
  boxes of the training split and frozen for both arms (test-time detection
  mAP is bit-identical between arms: 0.654). Boxes and classes never differ;
  only the relation labels do.
- **Split** follows the calibration protocol: train = groups 0–4 (500 images;
  5,421 human vs 119,020 automatic relations; the 22× density difference is
  the treatment), validation = group 5, test = groups 6–8 (human gold in both
  arms; the framework's loader retains 210 of the 236 test images).
- **Three predictions were registered in §5.6 before the run**: (1) the
  human-label arm saturates early, replicating the source paper; (2) the
  auto-label arm reaches a higher plateau; (3) `near` recovers.

## 6.2 Training dynamics: prediction 1 confirmed

The human arm **peaks at epoch 4 of 25** and never improves again, oscillating
between 0.101 and 0.123 for the remaining twenty-one, mild overfitting on
5,421 sparse triplets. That replicates the source paper's central observation
("all predictors reached their peak mR@100 well before the final epoch") on
its own dataset with a current model, and adds the cause: sparse supervision
is exhausted early. The auto arm is still climbing at epoch 8, does not reach
95% of its best until **epoch 14** and peaks at **epoch 22**, and sees 213
distinct triplet types in training against the human arm's 94. Where one arm
has learned everything its labels contain in four epochs, the other is still
finding structure at twenty-two.
{{fig:sgg-training-curves}} plots both: each validation
series uses its own arm's labels, so only the *shapes* compare, never the
heights.

## 6.3 Test results: prediction 2 unresolved, prediction 3 refuted

Both best checkpoints evaluated on the identical test set (210 images,
human gold, groups 6–8), at seed 42:

| metric (test, sgdet) | human-trained | auto-trained |
|---|---|---|
| R@100 | **0.280** | 0.259 |
| mR@100 | 0.270 | **0.291** |
| F1@100 | **0.275** | 0.274 |
| zero-shot recall zR@100 | 0.073 | **0.309** |

At this seed the automatic arm leads the mean-recall metric, 0.291 against
0.270, while the human arm keeps raw recall and the two are level on F1. The
ordering inverts predicate by predicate: the human arm is ahead on five of
the seven and the automatic arm on the support pair, by 0.077 on `on` and
0.284 on `under`. For `near` both arms sit far below the source paper's
0.22–0.25 floor, 0.117 and 0.108, so prediction 3 is refuted as stated and
the automatic arm's dense, fitted `near` does not survive the trip through a
learned model. Appendix F.1 carries the per-predicate breakdown. The
sharpest separation is on triplet *types never seen in training*, where the
automatic arm recalls 0.309 against the human arm's 0.073, a fourfold gap at
this seed and fivefold over three (§6.3.1). What that column measures here
is set out in §6.4, and it is coverage of the relation types the manual
annotation never recorded, not compositional generalisation.

Prediction 2 is the one a single run cannot carry, in either direction: the
margin above is well inside the human arm's own seed spread, which is why
§6.3.1 exists and why the heading calls it unresolved and not confirmed.
Re-scoring the same checkpoints on individual test slices decomposes the
headline, and the seed replication below repeats that decomposition as a
spread, so the single-run slice figures are in Appendix F.1. Aligning the
front/behind convention of groups 6 and 8, one disclosed bit per group as in
§4.5, lifts both arms and leaves them level, 0.310 against 0.312.

### 6.3.1 Replication across seeds, and one claim withdrawn

Every number above comes from one run per arm, and two of the margins are
too small for one run to separate from training noise. Both arms were
retrained at seeds 43 and 44 and all six checkpoints re-scored on every
slice (`scripts/kaggle/`, aggregated by `eval/seed_stats.py`), against the
same frozen detector, so the spread below is the relation model's own
variance.

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
against gold with the two inverted annotators corrected (§4.5). Cells give
the mean over three seeds with the per-seed range in brackets.

**On the metric this chapter is organised around, the two label sources are
indistinguishable.** Pooled mR@100 is 0.293 human against 0.292 automatic,
and no slice separates them: every row's seed ranges overlap, with the
automatic arm's sitting *inside* the human arm's on the full test set. Paired by seed the automatic arm leads at 42 (+0.021) and
43 (+0.010) and trails at 44 (−0.032), so the pooled difference of 0.001 rests
on a single run. The correct statement is parity, and it is a statement about this metric and not about the labels: on raw R@100 the human arm keeps a
real margin, 0.295 against 0.255, and on zero-shot recall the automatic arm
leads fivefold, 0.268 against 0.052.

The arms differ far more in stability than in score. Across seeds the
automatic arm's pooled mR@100 spans 0.006 against the human arm's 0.052 and
the vision-language arm's 0.044, eight and seven times wider. One definition
applied uniformly produces a model that lands in the same place whatever the
initialisation, and nine annotators applying nine conventions do not. A margin
of 0.001 between arms whose own seeds move by 0.052 is not a result in either
direction, which is the reading the spread column exists to force.

Where the arms do differ, they differ by annotator. The human arm is
ahead on the two annotators §4.5 convicts of inverting the front/behind
convention, by 0.022 on group 6 and 0.027 on group 8, and *behind* on group 7,
the one annotator this dissertation convicts of nothing, by 0.011. The
ordering runs with annotation quality and not with geometry, and the sign
change on the clean annotator is the part worth noting: whatever advantage
human labels carry here does not survive contact with an annotator who
followed the stated convention. None of these three differences is separable
across seeds, so the ordering is offered as a consistent direction and not as
three measured effects.

The size of that contamination can be measured. Of the 2,818 relations in
the test gold, **1,189 (42%) are front/behind, and 859 of those (72%) come
from the two inverted annotators** — so **30% of the entire yardstick is a
predicate labelled in the opposite direction to the convention every
training group used**. Both arms train on groups 0–5, where no inversion is
measured, so neither can score those relations and the penalty falls on them
equally. The inversion therefore sets a *ceiling*, not a bias. It caps what
any model trained on this data can reach on 30% of the test gold, which
compresses the range the two arms can differ across and makes every absolute
mR@100 in this chapter a lower bound on both sides.

What difference remains is a separate effect, and §6.4 localises it to
annotator *selection* rather than annotator convention: it appears on the
lateral predicates too, which have no direction to invert. The two defects
co-occur in the same annotators without one causing the other.

Two labelling rules support this, not one. The figures above are the shipped
`on_contact_min` of 0.85 (§4.14); the same experiment at the earlier 0.60
gave 0.278, 0.286, 0.307 and 0.109 across the same four slices. Raising the
threshold improved three slices and cost the fourth, and the ordering by
annotator defect held under both. A pattern that survives changing the
labelling rule is a property of the test annotation rather than of one
configuration of the tool.

Second, **the zero-shot result is robust and larger than first reported.**
Pooled zR@100 is 0.172 against 0.003, roughly sixty-fold, with disjoint
ranges. It is not an artefact of pooling: the auto arm leads on *every*
annotator separately (0.428 against 0.000 on group 6, 0.286 against 0.005 on
group 7, 0.033 against 0.000 on group 8), and the human arm recalls nothing
outside its training combinations on two of the three. Unlike the ranking
metrics this gap is not near-run at any seed or slice, and it points the same
way on defective and clean annotators alike, which is what distinguishes a
property of the labels from a property of the gold. §6.5 sets out what the
column establishes: which relation types each source covers, not
compositional generalisation, because both arms score against one shared
reference.

### 6.3.2 A third label source

Chapter 5's vision-language labels were put through the same benchmark as a
third arm, three seeds, same frozen detector, same human test gold, trained in
the same session as the other two. Pooled it leads both at 0.329, against
0.293 human and 0.292 auto, and it is the only arm whose lead over the human
one is consistent across every slice. That is what §6.4's argument predicts:
§4.13 measures this model annotating sparsely and human-like, so a metric
rewarding resemblance to the manual pass should reward it, and it does — more
than the manual pass rewards itself.

The sharpest form of that result is on group 7, the one test annotator with no
measured defect and therefore the cleanest gold, where the vision-language arm
reaches **0.362 against 0.278 human and 0.289 auto**, with a per-seed range
(0.357–0.364) touching neither. A win on the cleanest annotator cannot be
attributed to matching a defect, and it is the one result in this chapter that
no argument here explains away. Appendix F.4 gives the per-slice table and
the three readings the experiment cannot separate.


## 6.4 Why the advantage disappears between Chapter 5 and this chapter

The same labels, the same held-out human gold, two very different verdicts.
Per-pair recall says the automatic labels teach two and a half times better,
0.748 against 0.297; ranked evaluation against sparse gold says they are
level. Neither the labels nor the gold differ between the experiments, so an
advantage that size can only have evaporated into the **structure of the
metric**, and decomposing it yields the chapter's real findings.

An earlier version of this chapter had a stronger claim to explain: the human
arm ahead at 0.326 against 0.278. That comparison drew its arms from training
runs made weeks apart against different states of the upstream framework, and
retraining all nine runs in one session removed the gap almost entirely, with
the human arm falling 0.033 while the vision-language arm moved 0.001 (§7.4).
The mechanisms below explained a reversal; they now explain the erasure of a
2.5× advantage, which is the same phenomenon at a different magnitude and a
claim the evidence supports more comfortably.

**(i) Ranking dilution by true-but-unlabelled predictions.** R@K is a
per-image ranking budget: only the top-K predictions count, and any prediction
absent from the gold consumes budget as a miss. The auto-trained model behaves
like its supervision and predicts densely, and Chapter 4's audits established
that such extras are overwhelmingly *true*. Against gold annotating ~10% of
pairs those true predictions rank above the annotated ones and are scored as
errors, which is the restricted-precision artefact of §4.3 reproduced at
benchmark level. The human-trained arm learned the annotators' *labelling
prior* instead, which is exactly what a ranking metric against human-selected
gold rewards.

**(ii) Convention mismatch is a shared penalty, not, as first
hypothesised, a differential one.** Both arms were trained on
consistent-convention front/behind (the tool's by construction; groups
0–4's by measurement), while groups 6 and 8, two thirds of the test gold,
invert it (§4.5). Re-scoring against aligned gold lifts both arms almost equally (+0.041 human, +0.035 auto; the human arm's *in front
of* recall jumps 0.124 → 0.386, the auto arm's 0.101 → 0.248): both models
learned the consistent convention, both pay the same tax on inverted gold,
and the gap between them barely moves. The initial hypothesis that the
denser arm is punished *harder* for its confidence is refuted by this
measurement and withdrawn.

That measurement replicates at the shipped threshold. Re-scoring the
retrained auto arm of §4.14 against the same convention-aligned gold moves it
from 0.292 to **0.316** mR@100 and from 0.255 to **0.292** R@100, a gain of
+0.025 against the +0.035 recorded at the earlier threshold. The size of the
correction is therefore a property of the test annotation rather than of the
labelling rule: two different label sets, produced by two different values of
`on_contact_min`, pay a tax of the same order for the same annotator defect.
Roughly a tenth of the absolute mR@100 reported anywhere in this chapter is
an artefact of that defect, on both sides of the comparison.

**(ii′) Where the gap actually lives: the two defective test groups.** The
per-group figures of §6.3.1 localise the human arm's lead to the two
annotators already convicted of convention inversion, where it is four times
what it is on the clean one, and 30% of the test gold is front/behind written
by those two. Group 6 shows the clearest fingerprint: its *lateral* gold,
geometrically unambiguous relations both models predict freely, is recalled at
0.49/0.69 by the human arm against 0.12/0.21 by the auto arm. Laterals have no
convention to invert, so what differs is *which* pairs the annotator selected,
and the human-trained model ranks exactly those highly because it learned human
selection habits, not because it knows more geometry.

**(iii) `near` gold is a single idiosyncratic annotator.** All 93 test
`near` labels come from group 8, whose usage is sparse and non-exhaustive
(§3.8). Neither arm can rank exactly those pairs highly; prediction 3
underestimated not the labels but the gold.

The zero-shot column does not measure what its name implies here. zR@100
is recall on triplet types absent from a model's *own* training data, but both
arms are scored against one shared reference, the human training annotation,
since that is the only way their numbers sit in one column. Of the 25 test
triplet types the human annotation omits, the human arm saw none in training
and the auto arm saw 24, so its 0.172 against 0.003 records that its labels
*cover* relation types the manual pass never recorded. That is the property
the annotation bottleneck predicts, and it is not compositional
generalisation; this dissertation does not claim it as such.

## 6.5 What survives, read both ways

Two interpretations survive and neither is available without the other. The
benchmark result is real: a consumer *evaluated against human-annotated scene
graphs* is better supervised by human labels, which carry the annotation prior
the evaluation shares. The interpretation is equally real: the ranking metric
inherits every defect measured in the gold, and the advantage is concentrated
exactly where the annotation is defective and absent where it is not, which is
what annotation-prior agreement would look like.

The criterion was non-inferiority, and the arithmetic of that belongs here.
Section 1.2.2 asked whether a model trained on automatic labels performs *at
least as well* as one trained on human labels, not whether it wins, so
parity is the shape a pass takes and not an advantage evaporating. That
reframing is worth only as much as the numbers behind it, and the numbers
refuse a strong claim in either direction. Paired by seed the automatic arm
leads at 42 and 43 and trails at 44; the mean difference is -0.0006 with a
95% interval of [-0.070, +0.069]. Three seeds bound the gap to about a
quarter of the metric's own value, and a margin of ±0.01 would need roughly
forty runs per arm. The experiment therefore establishes neither superiority
nor equivalence, and a reader entitled to say the automatic labels did not
beat the human ones is equally entitled to say this design could not have
shown it if they had.

**What the same three seeds do resolve** is the part that is not a null.
Zero-shot recall separates with disjoint ranges, 0.225-0.309 against
0.004-0.079, and so does reproducibility, a 0.006 spread against 0.052.
Those are the comparisons this sample size can make and both run the
automatic arm's way. Set against the cost of obtaining them, nine annotators
against five minutes on one consumer GPU, indistinguishability on the
ranking metric is close to the result the project set out to obtain: RQ1 and
RQ2 ask whether the human can be removed, not whether the machine wins.

The critical reading is not novel to this project, which is what makes it
credible. Neural Motifs (Zellers et al., 2018) established that a frequency
baseline ignoring the image is hard to beat; Unbiased SGG (Tang et al.,
2020) formalised how thoroughly such models absorb the annotation
distribution; Northcutt, Athalye and Mueller (2021) showed erroneous test
labels reorder rankings across ten benchmarks. What this chapter adds is a
case where the confound is *isolated by construction*: the arms differ only
in label source and share a frozen detector, and the per-annotator defects
were measured beforehand in Chapter 4, so the advantage is attributed to
annotation practice, not inferred. The one remaining instrument is a manual
audit of the auto arm's top-ranked "false positives", the analogue of §4.4,
left as designed follow-up.


## 6.6 What Chapter 5's predictions got right and wrong

Registered before the run, judged after: prediction 1 (early human-arm
saturation) is **confirmed** and replicates the source paper. Prediction 2
(higher plateau) is **unresolved on mR@100**, and the word matters: an
earlier version of this chapter recorded it as refuted on the strength of a
0.048 gap that the retrained arms of §6.3.1 reduce to 0.001, which no
experiment of this size can call in either direction. Where the plateau *is*
higher is the zero-shot component the prediction did not name, 0.268 against
0.052. Prediction 3 (`near` recovery) is **refuted**: it wrongly assumed the
test gold could reward dense `near` prediction. The value of
pre-registration is that these verdicts are checkable, and that one of them
had to be revised when the measurement improved is a point in its favour and
not against it.

The replication adds a fourth verdict, on a claim made *after* the run, not
before: §6.3.1 withdraws the single-seed group-7 result, not quietly editing
it away. The lesson is the one already applied to the pre-registered
predictions and not to that one, that a difference is worth naming only once
its size is compared with the variation of the procedure that produced it.

## 6.7 Answer, at the level the source paper measures

At the level the source paper evaluates robot-readiness, automatic labels
train a model that trains longer, covers five times more of the relation
types the manual annotation never recorded, reproduces itself across seeds
more than eight times more tightly, ranks level overall, and ranks slightly
ahead on the one test annotator with no measured defect. The evidence
supports a conditional claim: **automatic labels are better training
material wherever ground truth means geometry; human labels remain better
wherever it means human annotation habits.** The first condition is the
operative one for a robot, which needs relations that are *correct* before
they are *human-like*, and §5.7 tests that one link further down, where it
survives.