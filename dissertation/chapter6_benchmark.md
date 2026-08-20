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

The human arm reaches 95% of its best validation score by **epoch 4**, peaks
at epoch 12 and declines, mild overfitting on 5,421 sparse triplets. That
replicates the source paper's central observation ("all predictors reached
their peak mR@100 well before the final epoch") on its own dataset with a
current model, and adds the cause: sparse supervision is exhausted early. The
auto arm is still improving at epoch 9, peaks at epoch 18 without declining,
and sees 213 distinct triplet types in training against the human arm's 94.
{{fig:sgg-training-curves}} plots both: each validation
series uses its own arm's labels, so only the *shapes* compare, never the
heights.

## 6.3 Test results: predictions 2 and 3 refuted on the ranking metric

Both best checkpoints evaluated on the identical test set (210 images,
human gold, groups 6–8):

| metric (test, sgdet) | human-trained | auto-trained |
|---|---|---|
| R@100 | **0.348** | 0.260 |
| mR@100 | **0.346** | 0.277 |
| F1@100 | **0.347** | 0.268 |
| zero-shot recall zR@100 | 0.004 | **0.157** |

The human-trained arm wins the headline ranking metrics (mR@100 0.346 vs
0.277) and leads on six of the seven predicates individually, the exception
being `under` at 0.762 against 0.731; `near` fails for both arms (0.054 and
0.032), close to the source paper's 0.22–0.25 floor, not recovered.
Predictions 2 and 3 are refuted as stated. Appendix F.1 carries the
per-predicate breakdown. One result points sharply the other way: on triplet *types never
seen in training*, the auto-trained arm recalls 0.157 against the human
arm's 0.004, a 39× gap at this seed (replicated at ~60× over three seeds;
§6.3.1). What that column measures here is set out in §6.4, and it is
coverage of the relation types the manual annotation never recorded rather
than compositional generalisation.

Re-scoring the same checkpoints on individual test slices decomposes that
headline, and the seed replication below repeats the decomposition with a
spread, not a point, so the single-run slice figures are in Appendix
F.1. One of them has no counterpart there: aligning the front/behind
convention of groups 6 and 8, one disclosed bit per group as in §4.5, lifts
both arms without changing the ordering, 0.387 against 0.312.

### 6.3.1 Replication across seeds, and one claim withdrawn

Every number above comes from one training run per arm. Two of the margins
are small enough that a single run cannot distinguish them from training
noise, so both arms were retrained at seeds 43 and 44 and all six
checkpoints re-scored on every slice (`scripts/kaggle/`, aggregated by
`eval/seed_stats.py`). The detector is the same frozen backbone throughout,
so the spread below is the relation model's own variance and nothing else.

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

All nine runs — three arms at three seeds — were trained in one session on one
clone of the framework, one frozen detector and one configuration, so the arms
differ in their relation labels and in nothing else. An earlier set of figures,
in which the human arm scored 0.326 pooled, was assembled from runs made
weeks apart against different states of the upstream code; §7.4 reports what
that cost and why these supersede it. The **aligned** row re-scores the same
models against test gold with the two inverted annotators corrected (§4.5).

Mean over three seeds, with the per-seed range in brackets; "separable"
records whether the two arms' ranges are disjoint.

**On the metric this chapter is organised around, the two label sources are
indistinguishable.** Pooled mR@100 is 0.293 for the human arm and 0.292 for
the automatic one. No slice separates them: on every row above the two seed
ranges overlap, and the automatic arm's range sits *inside* the human arm's on
the full test set. Paired by seed the automatic arm leads at 42 (+0.021) and
43 (+0.010) and trails at 44 (−0.032), so the pooled difference of 0.001 rests
on a single run. The correct statement is parity, and it is a statement about
this metric rather than about the labels: on raw R@100 the human arm keeps a
real margin, 0.295 against 0.255, and on zero-shot recall the automatic arm
leads fivefold, 0.268 against 0.052.

**The arms differ far more in stability than in score.** Across seeds the
automatic arm's pooled mR@100 spans 0.006; the human arm's spans 0.052 and the
vision-language arm's 0.044, nearly nine and seven times wider. One definition
applied uniformly produces a model that lands in the same place whatever the
initialisation, and nine annotators applying nine conventions do not. A margin
of 0.001 between arms whose own seeds move by 0.052 is not a result in either
direction, which is the reading the spread column exists to force.

**Where the arms do differ, they differ by annotator.** The human arm is
ahead on the two annotators §4.5 convicts of inverting the front/behind
convention, by 0.022 on group 6 and 0.027 on group 8, and *behind* on group 7,
the one annotator this dissertation convicts of nothing, by 0.011. The
ordering runs with annotation quality rather than with geometry, and the sign
change on the clean annotator is the part worth noting: whatever advantage
human labels carry here does not survive contact with an annotator who
followed the stated convention. None of these three differences is separable
across seeds, so the ordering is offered as a consistent direction and not as
three measured effects.

The size of that contamination is measurable rather than rhetorical. Of the
2,818 relations in the test gold, **1,189 (42%) are front/behind, and 859 of
those (72%) come from the two inverted annotators** — so **30% of the entire
yardstick is a predicate labelled in the opposite direction to the convention
every training group used**. Both arms train on groups 0–5, where no inversion
is measured, so neither can score those relations and the penalty falls on
them equally. What the inversion sets is therefore a *ceiling*, not a bias: it
caps what any model trained on this data can achieve on 30% of the test gold,
compressing the range in which the two arms can differ at all, and it makes
the absolute mR@100 figures in this chapter lower bounds on both sides rather
than estimates of spatial competence.

What difference remains is a separate effect, and §6.4 localises it to
annotator *selection* rather than annotator convention: it appears on the
lateral predicates too, which have no direction to invert. The two defects
co-occur in the same annotators without one causing the other.

Two labelling rules support this rather than one. The figures above are the
shipped `on_contact_min` of 0.85 (§4.14); the same experiment at the earlier
0.60 gave 0.278, 0.286, 0.307 and 0.109 across the same four slices. Raising the
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

The same labels, the same held-out human gold, and two very different
verdicts. Per-pair recall (Chapter 5) says the automatic labels teach two and
a half times better, 0.748 against 0.297. Ranked evaluation against sparse
gold says the two are level, 0.292 against 0.293. Nothing about the labels or
the gold differs between the experiments, so an advantage of that size cannot
evaporate for any reason other than the **structure of the metric**, and
decomposing it yields the chapter's real findings.

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
invert it (§4.5). Re-scoring against convention-aligned gold lifts the two
arms almost equally (+0.041 human, +0.035 auto; the human arm's *in front
of* recall jumps 0.124 → 0.386, the auto arm's 0.101 → 0.248): both models
learned the consistent convention, both pay the same tax on inverted gold,
and the gap between them barely moves. The initial hypothesis that the
denser arm is punished *harder* for its confidence is refuted by this
measurement and withdrawn.

That measurement replicates at the shipped threshold. Re-scoring the
retrained auto arm of §4.14 against the same convention-aligned gold moves it
from 0.291 to **0.316** mR@100 and from 0.255 to **0.293** R@100, a gain of
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

**The zero-shot column does not measure what its name implies here.** zR@100
is recall on triplet types absent from a model's *own* training data, but both
arms are scored against one shared reference, the human training annotation,
since that is the only way their numbers sit in one column. Of the 25 test
triplet types the human annotation omits, the human arm saw none in training
and the auto arm saw 24, so its 0.172 against 0.003 records that its labels
*cover* relation types the manual pass never recorded. That is the property
the annotation bottleneck predicts, and it is not compositional
generalisation; this dissertation does not claim it as such.

## 6.5 An honest reading, both ways

Two interpretations survive and neither is available without the other. The
benchmark result is real: a consumer *evaluated against human-annotated scene
graphs* is better supervised by human labels, which carry the annotation prior
the evaluation shares. The interpretation is equally real: the ranking metric
inherits every defect measured in the gold, and the advantage is concentrated
exactly where the annotation is defective and absent where it is not, which is
what annotation-prior agreement would look like.

The critical reading is not novel to this project, which is what makes it
credible rather than self-serving. Neural Motifs (Zellers et al., 2018)
established that a frequency baseline ignoring the image is hard to beat;
Unbiased SGG (Tang et al., 2020) formalised how thoroughly such models absorb
the annotation distribution; Northcutt, Athalye and Mueller (2021) showed
erroneous test labels reorder rankings across ten benchmarks. What this
chapter adds is a case where the confound is *isolated by construction*: the
arms differ only in label source and share a frozen detector, and the
per-annotator defects were measured beforehand in Chapter 4, so the advantage
is attributed to annotation practice, not inferred. The one remaining
instrument is a manual audit of the auto arm's top-ranked "false positives",
the analogue of §4.4, left as designed follow-up.


## 6.6 What Chapter 5's predictions got right and wrong

Registered before the run, judged after: prediction 1 (early human-arm
saturation) is **confirmed** and replicates the source paper. Prediction 2
(higher plateau) is **refuted on mR@100 as stated**, the plateau being higher
only on the zero-shot component the prediction did not name. Prediction 3
(`near` recovery) is **refuted**: it wrongly assumed the test gold could
reward dense `near` prediction. The value of pre-registration is that these
verdicts are checkable, and the mechanism analysis above is what the misses
taught.

The replication adds a fourth verdict, on a claim made *after* the run rather
than before: §6.3.1 withdraws the single-seed group-7 result, not
quietly editing it away. The lesson is the one already applied to the
pre-registered predictions and not to that one, that a difference is worth
naming only once its size is compared with the variation of the procedure
that produced it.

## 6.7 Answer, at the level the source paper measures

At the level the source paper evaluates robot-readiness, automatic labels
train a model that trains longer, covers the relation types the manual
annotation never recorded five times more fully (zR@100 0.268 against 0.052),
reproduces itself across seeds nearly nine times more tightly (spread 0.006
against 0.052), ranks level overall (0.292 against 0.293), and ranks slightly
ahead on the one test annotator with no measured defect. The claim the evidence supports is conditional:
**automatic labels are better training material wherever ground truth means
geometry; human labels remain better wherever it means human annotation
habits.** The first condition is the operative one for a robot, which needs
relations that are *correct* rather than *human-like*, and §5.7 tests that a
link further down, where it survives. The chain still lacks execution on a
physical robot (§9.3).
