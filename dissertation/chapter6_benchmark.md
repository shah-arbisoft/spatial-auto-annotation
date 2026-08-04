# Chapter 6: The Direct Benchmark Test: Training REACT++ on Both Label Sources

> Numbers generated on Kaggle (T4 GPU) with SGG-Benchmark; training logs and
> the exact per-epoch series in `outputs/sgg_benchmark/`; conversion by
> `scripts/export_sgg_benchmark.py`; run recipe in `scripts/kaggle/`.

This chapter is the third analysis iteration: the heavyweight version of the
RQ2 experiment, run in the source paper's own framework. Section 6.1 gives
the design and the three predictions registered in advance, §6.2–6.3 the
training dynamics and test results, §6.4 the decomposition of why the verdict
differs from Chapter 5, §6.5 both readings of the evidence, and §6.6–6.7 the
prediction verdicts and the conditional answer.

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

The human-label arm reaches 95% of its best validation score by **epoch 4**,
peaks at epoch 12, and then declines, showing mild overfitting on 5,421 sparse
triplets. This replicates the source paper's central training observation
("all predictors reached their peak mR@100 well before the final epoch") on
its own dataset with a current model, and adds the cause: sparse supervision
is exhausted early. The auto-label arm is still improving at epoch 9 and
peaks at epoch 18 with no decline; it also exposes the model to 213 distinct
triplet types during training against the human arm's 94. (Figure:
`sgg_training_curves.png`. The two validation series use each arm's own
label source, so only their *shapes* are compared, never their heights.)

## 6.3 Test results: predictions 2 and 3 refuted on the ranking metric

Both best checkpoints evaluated on the identical test set (210 images,
human gold, groups 6–8):

| metric (test, sgdet) | human-trained | auto-trained |
|---|---|---|
| R@100 | **0.348** | 0.260 |
| mR@100 | **0.346** | 0.277 |
| F1@100 | **0.347** | 0.268 |
| zero-shot recall zR@100 | 0.004 | **0.157** |

| per-predicate mR@100 | human-trained | auto-trained |
|---|---|---|
| on | **0.741** | 0.637 |
| under | 0.731 | **0.762** |
| to the left of | **0.203** | 0.120 |
| to the right of | **0.373** | 0.180 |
| in front of | **0.124** | 0.101 |
| behind | **0.195** | 0.109 |
| near | **0.054** | 0.032 |

The human-trained arm wins the headline ranking metrics (mR@100 0.346 vs
0.277); `near` fails for both arms (0.05/0.03), close to the source paper's
0.22–0.25 floor rather than recovered. Predictions 2 and 3 are refuted as
stated. One result points sharply the other way: on triplet *types never
seen in training*, the auto-trained arm recalls 0.157 against the human
arm's 0.004, a 39× gap in compositional generalisation at this seed
(replicated at ~60× over three seeds; §6.3.1).

Two follow-up evaluations decompose the headline (same checkpoints, test
slices re-scored):

| test slice (mR@100) | human-trained | auto-trained |
|---|---|---|
| full test, as annotated | **0.346** | 0.277 |
| full test, conventions aligned* | **0.387** | 0.312 |
| group 6 alone (inverted convention) | **0.382** | 0.261 |
| group 7 alone (consistent annotator) | 0.323 | 0.334 |
| group 8 alone (inverted, dense `near` user) | **0.190** | 0.116 |

\* groups 6/8's front/behind gold flipped, one disclosed bit per group, as in
§4.5.

### 6.3.1 Replication across seeds, and one claim withdrawn

Every number above comes from one training run per arm. Two of the margins
are small enough that a single run cannot distinguish them from training
noise, so both arms were retrained at seeds 43 and 44 and all six
checkpoints re-scored on every slice (`scripts/kaggle/`, aggregated by
`eval/seed_stats.py`). The detector is the same frozen backbone throughout,
so the spread below is the relation model's own variance and nothing else.

| slice | metric | human-trained | auto-trained | separable? |
|---|---|---|---|---|
| full test | mR@100 | **0.326** (0.303–0.347) | 0.278 (0.268–0.289) | yes |
| full test | zR@100 | 0.003 (0.000–0.004) | **0.172** (0.157–0.196) | yes |
| group 6 | mR@100 | **0.366** (0.343–0.382) | 0.286 (0.261–0.304) | yes |
| group 6 | zR@100 | 0.000 (0.000–0.000) | **0.300** (0.152–0.379) | yes |
| group 7 | mR@100 | 0.308 (0.298–0.323) | 0.307 (0.289–0.334) | **no** |
| group 7 | zR@100 | 0.005 (0.000–0.008) | **0.165** (0.094–0.221) | yes |
| group 8 | mR@100 | **0.171** (0.142–0.197) | 0.109 (0.087–0.125) | yes |
| group 8 | zR@100 | 0.000 (0.000–0.000) | **0.036** (0.018–0.061) | yes |

Mean over three seeds, with the per-seed range in brackets; "separable"
records whether the two arms' ranges are disjoint.

One caveat belongs with the decimals. Re-scoring the *same* checkpoints a
second time does not reproduce them exactly: an independent re-evaluation
pass moved the pooled human mR@100 by 0.001 and the group-8 figures, drawn
from the smallest slice at 37 images, by up to 0.008. Inference here is not
bit-deterministic, so a margin of that order is not a result. The numbers
above are those of the re-evaluation committed with this repository, and no
claim in this chapter turns on a difference smaller than the spread already
shown in brackets.

Two things follow, and the first is a correction. **The single-seed group-7
result does not replicate.** At seed 42 the auto arm led 0.334 to 0.323, and
that margin was reported as observed rather than tested precisely because
73 images and one run could not support more. Across three seeds the arms
are indistinguishable: 0.307 against 0.308, with ranges that overlap almost
completely. The claim that the auto arm *wins* on the clean annotator is
therefore withdrawn.

What survives is the pattern it was evidence for, and it survives with a
spread rather than a point. The human arm's advantage is large on both
annotators carrying a measured defect (group 6: 0.366 vs 0.286; group 8:
0.171 vs 0.109) and disappears entirely on the one annotator whose labels
this dissertation did not convict of anything (0.308 vs 0.307). The
gradient runs with annotation quality, not with geometry, which is the
claim §6.4 develops. The weaker version is also the more defensible one:
parity on clean gold needs no argument about whose labels are better, while
the human arm's lead on defective gold demands an explanation.

Second, **the zero-shot result is robust and larger than first reported.**
Pooled zR@100 is 0.172 for the auto arm against 0.003 for the human arm, a
factor of roughly sixty, with disjoint ranges. The pattern is not an artefact
of pooling: the auto arm leads on *every* annotator taken separately (0.300
against 0.000 on group 6, 0.165 against 0.005 on group 7, 0.036 against
0.000 on group 8), and the human arm recalls literally nothing outside its
training combinations on two of the three. Unlike the ranking metrics, this
gap is not a near-run thing at any seed or on any slice, and unlike the
ranking metrics it points the same way on defective and clean annotators
alike, which is what distinguishes a property of the labels from a property
of the gold. What that column does and does not establish is set out in §6.5:
it measures which relation types each source covers, not compositional
generalisation, because both arms are scored against one shared reference.

## 6.4 Why the verdict flipped between Chapter 5 and this chapter

The same labels, the same held-out human gold, and two opposite outcomes:
per-pair recall (Chapter 5) says automatic labels teach better (0.76 vs
0.30); ranked evaluation against sparse gold says the reverse. Because the
labels and gold are identical across the two experiments, the flip localises
in the **structure of the metric**, and decomposing it yields the chapter's
real findings.

**(i) Ranking dilution by true-but-unlabelled predictions.** R@K is a
per-image ranking budget: only the top-K predictions count, and any
prediction absent from the gold consumes budget as a miss. The auto-trained
model behaves like its supervision. It predicts relations densely, and
Chapter 4's audits established that such dense extras are overwhelmingly
*true* (sampled true precision ≈1.0 for lateral/proximity/depth, ≈0.9 for
support). Against gold that annotates ~10% of pairs, those true predictions
rank above the annotated ones and are scored as errors: the
restricted-precision artefact of §4.3, reproduced at benchmark level. The
human-trained arm, by contrast, learned the annotators' *labelling prior*
(which pairs a human bothers to record), which is exactly what a ranking
metric against human-selected gold rewards.

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

**(ii′) Where the gap actually lives: the two defective test groups.** The
per-group decomposition localises the human arm's lead precisely, and the
seed replication of §6.3.1 is what makes the localisation trustworthy. On
**group 7, the one test annotator with consistent conventions, the two arms
are indistinguishable**: 0.308 against 0.307 over three seeds, with
overlapping ranges. On the two annotators this dissertation had already
convicted of convention inversion (and, for group 8, idiosyncratic `near`
usage and one-directional support) the human arm wins by margins far larger
than seed variance: 0.366 against 0.286 on group 6, and 0.171 against 0.109
on group 8. The human arm's entire headline lead is therefore manufactured
on the two defective annotators and vanishes on the clean one. Ranking
parity on clean gold is a weaker statement than the single-seed run
suggested, and a sturdier one: it needs no claim about which labels are
better, only the observation that the human arm's advantage tracks
annotation defects rather than geometry. The group-7 zero-shot gap is not
marginal at any seed (zR@100 0.165 against 0.005).
Group 6 shows the clearest fingerprint of annotation-prior matching: its
*lateral* gold, geometrically unambiguous relations both models predict
freely, is recalled at 0.49/0.69 by the human arm against 0.12/0.21 by the
auto arm. Laterals have no convention to invert; what differs is *which*
pairs the annotator selected, and the human-trained model ranks exactly
those pairs highly because it learned human selection habits, not because
it knows more geometry.

**(iii) `near` gold is a single idiosyncratic annotator.** All 93 test
`near` labels come from group 8, whose usage is sparse and non-exhaustive
(§3.8). Neither arm can rank exactly those pairs highly; prediction 3
underestimated not the labels but the gold.

**The zero-shot column needs reading carefully, because it does not measure
what its name implies here.** zR@100 is defined as recall on triplet types
absent from a model's *own* training data, but both arms are scored against
one shared reference, the human training annotation, since that is the only
way their numbers sit in one column. The two are not the same question. Of
the 25 test triplet types the human annotation omits, the human arm saw none
in training and the auto arm saw 24, so the auto arm's 0.172 against 0.003
(three seeds, disjoint ranges) records that its labels *cover* relation types
the manual pass never recorded. That is a real and relevant property of the
label source, and it is the one the annotation bottleneck predicts. It is not
evidence of compositional generalisation, and this dissertation does not
claim it as such.

## 6.5 An honest reading, both ways

Two interpretations survive the evidence, and the dissertation records both.
Read charitably toward the benchmark: if the downstream consumer will be
*evaluated against human-annotated scene graphs*, human labels remain the
better supervision, because they carry the annotation prior the evaluation
shares, a genuine, practical advantage the automatic labels do not
replicate. Read critically: the ranking metric inherits every defect this
dissertation measured in the gold (sparsity that penalises true
predictions, inverted conventions in two of three test groups, a
single-annotator `near`), so it partly measures agreement with those defects
rather than spatial understanding. The per-group decomposition, replicated
across seeds, adjudicates between the readings: against the only test
annotator whose labels this dissertation's earlier chapters did *not*
convict of a measured defect, the two arms rank equally well and the auto
arm covers far more of the omitted relation types, while against the two
convicted annotators the human arm wins in proportion to their
idiosyncrasy. That is weaker than the single-seed run implied, and it is
the version the evidence supports.

Neither reading is available without the other. The benchmark result is
real: on this dataset's test annotation the human-trained model ranks
better, and a practitioner scoring against such annotation should expect
that. The interpretation is equally real: the advantage is concentrated
exactly where the annotation is defective and absent where it is not, which
is what a metric measuring annotation-prior agreement would look like.

The critical reading is not novel to this project, which is what makes it
credible rather than self-serving. Neural Motifs (Zellers et al., 2018)
established that a frequency baseline ignoring the image is hard to beat on
scene-graph benchmarks, and Unbiased SGG (Tang et al., 2020) formalised how
thoroughly such models absorb the annotation distribution. Northcutt,
Athalye and Mueller (2021) then showed that erroneous test labels reorder
model rankings across ten standard benchmarks. What this chapter adds is a
case where the confound is *isolated by construction*: because the two arms
differ only in label source and share a frozen detector, and because the
per-annotator defects were measured beforehand in Chapter 4, the advantage
can be attributed to annotation practice rather than inferred from it. The
one remaining instrument is a manual audit of the auto arm's top-ranked
"false positives" (the direct analogue of §4.4), left as designed follow-up.

## 6.6 What Chapter 5's predictions got right and wrong

Registered before the run, judged after: prediction 1 (early human-arm
saturation) is **confirmed**, and it replicates the source paper. Prediction
2 (higher plateau) is **refuted on mR@100 as stated**; the plateau is higher
only on the zero-shot component the prediction did not name. Prediction 3
(`near` recovery) is **refuted**; it wrongly assumed the test gold could
reward dense `near` prediction. The value of pre-registration is precisely
that these verdicts are checkable; the mechanism analysis above is what the
misses taught.

The replication adds a fourth verdict, this time on a claim made *after* the
run rather than before it. Reading the single-seed per-group table, an
earlier draft of this chapter reported that the auto arm wins on group 7.
Three seeds show the arms tied there, and §6.3.1 withdraws the claim. The
episode is worth recording rather than quietly editing away: a margin of
0.011 on 73 images was always inside the noise the replication went on to
measure (per-seed ranges spanning 0.025 and 0.045), and the honest label at
the time would have been "indistinguishable" rather than "ahead". The
zero-shot gap, by contrast, was large enough to survive every seed and
every slice. The general lesson is the one the design already applied to
the pre-registered predictions and should have applied here too: a
difference is worth naming only once its size is compared with the
variation of the procedure that produced it.

## 6.7 Answer, at the level the source paper measures

At the level the source paper itself evaluates robot-readiness (SGG metrics
on human-annotated gold), automatic labels train a model that trains longer,
covers relation types the manual annotation never recorded far more fully
(zR@100 0.172 vs 0.003 over three seeds, ranges disjoint; §6.5 sets out what
that column does and does not establish), ranks equally well
against the one test annotator with no measured annotation defect, and
ranks lower against the two that have them, for reasons this dissertation
can attribute line by line to measured properties of those annotations. The
claim the evidence supports is therefore conditional:
**automatic labels are better training material wherever ground truth means
geometry; human labels remain better wherever ground truth means human
annotation habits.** For the robot chain, where the planner consumes
relations that must be *correct* rather than *human-like*, the first
condition is the operative one, and testing it end-to-end (relations into an
LLM planner, plans scored for validity) is the designed next step.
