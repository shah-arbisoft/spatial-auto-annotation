# Chapter 7 — The Direct Benchmark Test: Training REACT++ on Both Label Sources

> Numbers generated on Kaggle (T4 GPU) with SGG-Benchmark; training logs and
> the exact per-epoch series in `outputs/sgg_benchmark/`; conversion by
> `scripts/export_sgg_benchmark.py`; run recipe in `scripts/kaggle/`.

## 7.1 Design

Chapter 5's controlled experiment used a deliberately small classifier; this
chapter runs the heavyweight version on the source paper's own terms: a
real-time scene-graph model (REACT++, the current model of the SGG-Benchmark
framework the dataset's tooling belongs to) trained once on the human
relations and once on this tool's, and evaluated on an identical held-out
test set. The isolation mirrors RQ2 exactly:

- **Shared detector.** A YOLOv8m backbone is trained once on the ground-truth
  boxes of the training split and frozen for both arms (test-time detection
  mAP is bit-identical between arms: 0.654). Boxes and classes never differ;
  only the relation labels do.
- **Split** follows the calibration protocol: train = groups 0–4 (500 images;
  5,421 human vs 119,020 automatic relations — the 22× density difference is
  the treatment), validation = group 5, test = groups 6–8 (human gold in both
  arms; the framework's loader retains 210 of the 236 test images).
- **Three predictions were registered in §5.5 before the run**: (1) the
  human-label arm saturates early, replicating the source paper; (2) the
  auto-label arm reaches a higher plateau; (3) `near` recovers.

## 7.2 Training dynamics — prediction 1 confirmed

The human-label arm reaches 95% of its best validation score by **epoch 4**,
peaks at epoch 12, and then declines — mild overfitting on 5,421 sparse
triplets. This replicates the source paper's central training observation
("all predictors reached their peak mR@100 well before the final epoch") on
its own dataset with a current model, and adds the cause: sparse supervision
is exhausted early. The auto-label arm is still improving at epoch 9 and
peaks at epoch 18 with no decline; it also exposes the model to 213 distinct
triplet types during training against the human arm's 94. (Figure:
`sgg_training_curves.png`. The two validation series use each arm's own
label source, so only their *shapes* are compared, never their heights.)

## 7.3 Test results — predictions 2 and 3 refuted on the ranking metric

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
arm's 0.004 — a 39× gap in compositional generalisation.

## 7.4 Why the verdict flipped between Chapter 5 and this chapter

The same labels, the same held-out human gold, and two opposite outcomes:
per-pair recall (Chapter 5) says automatic labels teach better (0.76 vs
0.30); ranked evaluation against sparse gold says the reverse. Because the
labels and gold are identical across the two experiments, the flip localises
in the **structure of the metric**, and decomposing it yields the chapter's
real findings.

**(i) Ranking dilution by true-but-unlabelled predictions.** R@K is a
per-image ranking budget: only the top-K predictions count, and any
prediction absent from the gold consumes budget as a miss. The auto-trained
model behaves like its supervision — it predicts relations densely, and
Chapter 4's audits established that such dense extras are overwhelmingly
*true* (sampled true precision ≈1.0 for lateral/proximity/depth, ≈0.9 for
support). Against gold that annotates ~10% of pairs, those true predictions
rank above the annotated ones and are scored as errors — the restricted-
precision artefact of §4.3, reproduced at benchmark level. The human-trained
arm, by contrast, learned the annotators' *labelling prior* — which pairs a
human bothers to record — which is exactly what a ranking metric against
human-selected gold rewards.

**(ii) Convention mismatch on two of three test groups.** Both arms were
trained on consistent-convention front/behind (the tool's by construction;
groups 0–4's by measurement), while groups 6 and 8 — two thirds of the test
gold — invert it (§4.5). The dense arm commits harder to the consistent
convention and is punished more systematically for it; the sparse arm's
weaker, noisier direction preferences scatter across both conventions and
are accidentally rewarded. Depth predicates: 0.124/0.195 (human) vs
0.101/0.109 (auto), both far below what either model knows.

**(iii) `near` gold is a single idiosyncratic annotator.** All 93 test
`near` labels come from group 8, whose usage is sparse and non-exhaustive
(§3.7). Neither arm can rank exactly those pairs highly; prediction 3
underestimated not the labels but the gold.

**The zero-shot flip is the counter-evidence in the benchmark's own terms.**
zR@100 scores triplet types absent from training — the one sub-metric that
cannot reward memorising the annotators' labelling prior. There the
auto-trained arm's 39× advantage (0.157 vs 0.004) shows what density and
consistency actually bought: geometry that composes to unseen combinations,
rather than a lookup of previously-labelled ones.

## 7.5 An honest reading, both ways

Two interpretations survive the evidence, and the dissertation records both.
Read charitably toward the benchmark: if the downstream consumer will be
*evaluated against human-annotated scene graphs*, human labels remain the
better supervision, because they carry the annotation prior the evaluation
shares — a genuine, practical advantage the automatic labels do not
replicate. Read critically: the ranking metric inherits every defect this
dissertation measured in the gold — sparsity that penalises true
predictions, inverted conventions in two of three test groups, a
single-annotator `near` — so it partly measures agreement with those defects
rather than spatial understanding; the zero-shot result, the audits and the
Chapter 5 experiment jointly indicate the auto-trained model knows the
geometry at least as well while generalising far better. Distinguishing the
readings empirically is the designed follow-up: per-group test evaluation
(the consistent group 7 versus the inverted 6/8) and a manual audit of the
auto arm's top-ranked "false positives", both direct analogues of Chapter
4's instruments.

## 7.6 What Chapter 5's predictions got right and wrong

Registered before the run, judged after: prediction 1 (early human-arm
saturation) — **confirmed**, and it replicates the source paper. Prediction
2 (higher plateau) — **refuted on mR@100 as stated**; the plateau is higher
only on the zero-shot component the prediction did not name. Prediction 3
(`near` recovery) — **refuted**; it wrongly assumed the test gold could
reward dense `near` prediction. The value of pre-registration is precisely
that these verdicts are checkable; the mechanism analysis above is what the
misses taught.

## 7.7 Answer, at the level the source paper measures

At the level the source paper itself evaluates robot-readiness — SGG metrics
on human-annotated gold — automatic labels train a model that trains longer,
generalises to unseen relation compositions dramatically better, and scores
lower on ranked recall against sparse human test annotations, for reasons
this dissertation can attribute line by line to measured properties of those
annotations. The claim the evidence supports is therefore conditional:
**automatic labels are better training material wherever ground truth means
geometry; human labels remain better wherever ground truth means human
annotation habits.** For the robot chain — where the planner consumes
relations that must be *correct*, not *human-like* — the first condition is
the operative one, and testing it end-to-end (relations into an LLM planner,
plans scored for validity) is the designed next step.
