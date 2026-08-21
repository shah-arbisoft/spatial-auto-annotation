# Chapter 2: Literature Review

This chapter reviews the work this project builds on and the work it must be
distinguished from. Its organising question is **label quality**, because
that is what the project's evidence ultimately turns on: not whether spatial
relations can be computed, which the literature already settles, but whether
computed labels are better or worse than the human labels they replace, and
how anyone would know.

It reaches the project's own approach only after the rival remedies that
stretch scarce labels instead of replacing them (§2.4), and it ends on the
two sections that argue against itself: what the field's metrics miss (§2.8)
and the strongest case against a rule-based annotator (§2.9).

## 2.1 Scene graphs and spatial relationships in robotics

The seven spatial predicates this project computes (*on*, *under*,
*left/right of*, *in front of / behind*, *near*) are the scene-graph edges
that carry the geometry an agent must respect to act. A planner told only
"cube, book, table" cannot decide what to move first; a planner told "the
cube is on the book" can, and §5.7 turns that into a measurement and not
an illustration. Language-driven robot planners ground their instructions in
exactly such a structured account of scene state (Ahn et al., 2022), and 3D
scene graphs were proposed as its unifying form, tying semantics, space and
camera into one queryable representation (Armeni et al., 2019). This review
concentrates on how those edges are **produced**: by hand, by learned
prediction, or, as this project argues, by computation from measured
geometry.

## 2.2 The source dataset and its annotation bottleneck

Wang et al. (2025) introduce *A Spatial Relationship Aware Dataset for
Robotics*, captured by a Boston Dynamics Spot robot. Verified specifics from
the paper:

- **Scale:** "nearly 1,000 robot-acquired indoor images" collected; after quality
  control "approximately 900" remain. (The released GitHub subset used here
  contains 884 images / **838 annotated**; see [DATASET_NOTES.md](../docs/DATASET_NOTES.md).)
- **Annotation:** "Nine trained annotators worked independently in batches of 100
  images," using the manual **SGDET-Annotate** tool (every box drawn, every class
  and relationship clicked), with a majority-vote cleaning pass. The released data
  is organised into nine groups of ~100, matching this exactly.
- **Vocabulary:** exactly seven predicates (*behind, in front of, on, to the left
  of, to the right of, under, near*) over six object classes (book, bottle, box,
  cube, human, remote).
- **Detector baseline:** a YOLOv10m backbone (Wang, A. et al., 2024) reaches
  ≈0.92 precision, 0.90 recall and **0.93 mAP@50** (mAP@50-95 ≈0.68).

Three limitations the authors themselves flag motivate this project. First,
**early saturation**: "all predictors reached their peak mR@100 well before
the final epoch," which they attribute to the fact that "the dataset's
limited diversity exhausts relational learning capacity early." Second, the
**`near` predicate is unreliable**: they report "inconsistencies,
particularly with the 'near' predicate," and it "remains challenging for all
models (0.2247–0.2494)." Third, their future-work prescription is explicit:
"augment under-represented relations while enforcing clear annotation
guidelines (e.g., spatial thresholds for 'near')."

The common cause is the **manual annotation bottleneck**. SGDET-Annotate
only *accelerates* human labelling (a human still decides every edge), so
the dataset cannot grow cheaply, diversity stays low, and inter-annotator
disagreement (worst on `near`) is baked in. **No automatic annotator exists
for this dataset.** This is the gap the project fills, and the fitted `near`
threshold is a direct, data-driven realisation of the authors' own "spatial
thresholds for near."

## 2.3 Label quality: weak supervision and annotator disagreement

The premise, replacing scarce human labels with dense computed ones, has an
established name: **weak supervision**. **Snorkel** (Ratner et al., 2017)
formalised *data programming*: experts write labelling functions instead of
labelling examples, and their noisy overlapping votes are combined into
training labels, trading per-label authority for coverage and consistency.
Its deployments repeatedly matched or beat hand-labelled baselines wherever
the labelled set, not the model, was the bottleneck. This project's
geometric rules are labelling functions in that sense, deterministic,
auditable and dense, with two departures. Measured geometry gives
near-exact, not noisy votes for five of the seven predicates (blind-audited
precision 0.79–1.00) and demonstrably noisy ones for the other two, where
§4.14 puts support at 0.40; on the five, no probabilistic aggregation is
needed, and on the two the objection Snorkel answers with aggregation is
answered here by abstention instead. The second departure is that the
computed labels are *validated against* the human labels they replace (RQ1),
not assumed comparable.

The complementary literature dismantles the premise that human annotation is
a single reliable gold standard. **Uma et al.'s (2021) survey of learning
from disagreement** documents systematic annotator disagreement across
vision and language, driven by ambiguous guidelines, subjective category
boundaries and annotator-specific conventions, and reviews methods that
treat disagreement as signal to be modelled and not noise to be averaged
away. The frame fits exactly: Chapter 4 measures three annotator behaviours,
selective `near` usage, an inverted front/behind convention in two groups,
and one-directional support labelling, which make "agreement with the
humans" per-annotator, not global. Two design decisions follow in Chapters
3–4: evaluation is reported per annotator group, never only pooled, and
thresholds are calibrated only on annotators who used a label, with the rest
held out.

Where the human judgements must themselves be evaluated, the measurement
tradition supplies the instruments: **Cohen's kappa** (Cohen, 1960) for
chance-corrected agreement between two verdict sets, and coefficients such as
Krippendorff's alpha for a pool of raters. Artstein and Poesio (2008) survey
both and their pitfalls, including the prevalence effect that depresses kappa
when one answer dominates. Chapter 4's independent validation applies exactly
these, scoring crowd judgements of sampled predictions against the author's
own audit verdicts. It also sharpens RQ2 into a question the
weak-supervision literature predicts but rarely tests directly: can
consistent computed labels *out-teach* inconsistent human ones on the humans'
own held-out annotations?

Label quality is not only a training concern; it contaminates *evaluation*.
Northcutt, Athalye and Mueller (2021) measured label errors across ten
heavily-used benchmarks (3.3% average, including 6% of ImageNet's validation
labels) and showed that correcting them changes model rankings: practitioners
may have been selecting the wrong model because the yardstick was wrong.
Spatial-relation benchmarks met the same problem from the collection side.
SpatialSense (Yang, Russakovsky and Deng, 2019) used *adversarial*
crowdsourcing, instructing annotators to find relations that defeat naive
predictors, because relations collected without that pressure are dominated
by guessable co-occurrences, not spatial reasoning; Rel3D (Goyal et
al., 2020) rebuilt the task on 3D scenes with minimally contrastive pairs
after finding 2D datasets let models score well without using spatial
information at all. Both respond to the fact this dissertation measures in
its own dataset: what a benchmark appears to test and what its annotation
rewards can diverge, invisibly, until someone measures the labels. Chapter 6
shows the consequence here, and Northcutt et al.'s conclusion, that rankings
flip when gold is corrected, is the shape of that result.

## 2.4 The rival family: semi-supervised and active learning

Computing labels from geometry is not the only established answer to
expensive annotation, and an honest review must position the project against
the stronger rival: use the labels that exist and stretch them.
**Active learning** (Settles, 2009) reduces annotation cost by choosing
*which* examples a human labels next, concentrating effort where the model is
most uncertain. **Semi-supervised learning** trains on a small labelled set
plus a large unlabelled one (van Engelen and Hoos, 2020); its simplest and
most widely used instrument is **pseudo-labelling** (Lee, 2013), in which a
model trained on the labelled seed labels the unlabelled data for its own
retraining, and its strongest modern form is noisy self-training (Xie et al.,
2020), which scaled the idea to ImageNet with a teacher–student loop.

Applied here the recipe would be: train on the ~10% of pairs the annotators
labelled, pseudo-label the remaining 90%, retrain. Three properties of this
dataset argue against it, each of them measured. First, self-training
*amplifies its seed*, and this seed is sparse and internally inconsistent
(selective `near`, two inverted front/behind conventions, one-directional
support; §2.2, Chapter 4); a teacher trained on contradictory conventions
teaches them on, with added confidence. Second, the seed is not merely small
but *selectively* small: annotators labelled what they found salient, so the
labelled 10% is not an unbiased sample of the 90% to be filled in, which is
the assumption pseudo-labelling needs. Third, the empirical anchor: Chapter
5's human-trained classifier is exactly the seed such a loop would start
from, and it collapses on the sparsely-labelled predicates (recall
0.08–0.25) and is unstable across seeds. A loop built on that teacher has
little reliable to amplify. Active learning fails differently: it still buys
*human* labels, reducing the bottleneck's slope without removing it, and it
rations inconsistency between annotators instead of fixing it.

The geometric route sidesteps all three failure modes because its labelling
function does not derive from the flawed seed at all: the rules are fitted to
a handful of thresholds (with the fit itself validated on held-out
annotators) and are exactly as consistent on the 90% as on the 10%.

None of this is left as argument. Chapter 5 implements the rival directly as
a third arm of the controlled experiment, running the standard
teacher-student self-training loop over the same features, model, split and
seeds, so that pseudo-labelling and programmatic labelling are compared
head to head on the humans' own held-out annotations. That is a comparison
the weak-supervision and semi-supervised literatures each motivate but
rarely run against one another.

## 2.5 Geometry-to-label pipelines (the lineage we build on)

A line of work *computes* spatial facts from perceived geometry instead of
predicting them from learned relational patterns. The idea predates its
current instantiations: CLEVR (Johnson et al., 2017) emitted the spatial
relations of hundreds of thousands of scenes *from the renderer*, exact by
construction, and became the standard diagnostic for compositional reasoning
precisely because programmatic labels carry no annotator noise to memorise.
What it sidesteps is the hard half: its geometry is known because the scenes
are synthetic, whereas an annotator for real photographs must first *recover*
geometry from pixels before any rule can fire. The works below take up that
half. Each is methodological ancestry here, and each outputs something other
than a scene-graph annotation for this dataset's seven predicates.

- **SpatialVLM** (Chen et al., 2024) is the foundational geometry-to-label
  method. It builds an *automatic* 3D spatial-VQA data-generation framework,
  lifting internet images to metric 3D via monocular depth and segmentation, then
  emitting up to ~2B spatial question–answer pairs from ~10M images. It
  establishes the central premise we adopt: spatial relations can be *derived from
  measured geometry without human relational labels*. But its output is
  **free-form VQA text**, its domain is **internet images**, and it targets no
  fixed predicate set. It is not an annotator for a robot scene-graph dataset.

- **SpatialRGPT** (Cheng et al., 2024) extends this with a data-curation
  pipeline that learns regional representations from **3D scene graphs** and a
  **depth "plugin"** that injects relative depth (via Depth Anything) into a VLM's
  visual encoder. It is the cleanest published RGB→depth→relation recipe and a key
  reference for our depth use, but the artefact is a **region-reasoning VLM**, not
  a deterministic labeller producing VG-format triplets.

  This raises the obvious alternative to the present project: rather than
  computing relations geometrically, ask a capable vision-language model to
  name them. Section 4.13 tests that instead of arguing it away, putting two
  current models through the same battery on the same images with the same
  boxes and definitions. Both recover under half the human triplets the
  pipeline does and lose F1 on every predicate, and neither is simply worse,
  since both are *more precise* where they speak. What settles the question
  is the shape of the output: silence on most pairs, and a symmetric relation
  asserted in one direction only about a third of the time, which are the
  behaviours §4.5 measures in the *human* annotation. A vision-language model
  asked to annotate reproduces the failure mode this project set out to
  replace, which is why pointing a larger model at it does not fill the gap.

- **VQASynth** (Remyx AI, 2024) is an open reproduction of the SpatialVLM
  pipeline: a chain of expert models (SAM2, monocular depth, grounded
  captioning) inferring spatial relationships to **create a spatial-VQA
  dataset**. It is the most reusable code reference, and its output stage
  produces **QA pairs** where we need a scene-graph-triplet writer. Its depth
  backend has shifted over time (DepthPro → VGGT), so we pin our own.

- **Open3D-VQA** (Zhang et al., 2025) is the source of our **error-correction**
  idea. It is an embodied spatial-reasoning benchmark for *open/aerial* space
  (89k QA pairs over seven spatial tasks, visual + point cloud) whose generation
  pipeline "extracts 3D spatial relationships from a single RGB image" with a
  **multi-modal correction flow to ensure quality**. We adapt that correction
  principle, rejecting geometrically impossible labels, into our correction step
  (spec §8), in a different domain (indoor tabletop, not aerial) and for a
  different output (triplets, not QA).

- **RoboSpatial** (Song et al., 2025) is the closest robotics-domain
  match, cited by the source paper. It teaches spatial understanding to 2D/3D VLMs
  from real indoor/tabletop scans (≈1M images, ≈3M annotated spatial relations).
  Critically for our design, it formalises **reference frames**: ego-centric,
  world-centric and object-centric readings of the same spatial phrase. The
  ambiguity is a documented property of spatial language, not an
  engineering nuisance, Landau and Jackendoff (1993) having shown that
  language encodes location through a small set of frame-dependent
  primitives, so "the cup is left of the box" is true in one frame and false
  in another and an annotator must pin the frame before any label is well
  defined. That justifies expressing *left/right* in the **camera (ego)
  frame** (predicate spec §0), which is also the frame the dataset's
  annotators saw on screen. Its output is spatial QA over three frames, not a
  fixed seven-predicate annotation.

One adjacent family needs explicit separation, because from a robotics
standpoint it looks closest. Online 3D scene-graph *mapping* systems build a
spatial-semantic graph as a robot moves: Hydra optimises a multi-layer graph
in real time from depth-equipped SLAM (Hughes, Chang and Carlone, 2022), and
ConceptGraphs opens the node vocabulary by fusing foundation-model features
into an RGB-D map (Gu et al., 2024). They *consume* depth sensors and
*produce* maps for a live robot, emit no dataset-format annotation for
existing monocular images, and none is validated against human annotators,
which is this project's deliverable and test. They strengthen the case for
automatic annotation rather than weakening it: the training data their
learned components need is what an annotator supplies.

A question the lineage rarely puts to itself is how anyone knows the computed
labels are right, and it matters here because this project's central claim is
a validation claim, not a generation one. Two kinds of evidence are
offered and neither is the kind RQ1 requires. The first is **downstream
benefit**: SpatialVLM (Chen et al., 2024) establishes its generated
supervision by fine-tuning a vision-language model on it and showing better
answers to spatial questions, and SpatialRGPT (Cheng et al., 2024) likewise
judges its curated region representations through the model they produce. The
second is **internal consistency**: Open3D-VQA (Zhang et al., 2025) discards
configurations its own geometry rules declare impossible, a principle this
project adapts in §3.6.

Both share one blind spot. A model trained on computed labels and tested on
questions from the same computation scores well on any convention applied
consistently, including a wrong one, and an internal consistency check is
satisfied by any coherent convention, since impossibility is judged by the
rules that produced the labels. Neither can detect systematic disagreement
with how humans use the words, the failure mode Chapter 4 measures in the
tool and in the annotators alike. Only comparison against an independently
produced human record can, which none of these works performs and which is
what RQ1 is. It also explains why Chapter 6's less favourable result does not
contradict Chapter 5's: judged against *human* annotation, not labels
from the same source, downstream benefit becomes genuinely adversarial, and
the lineage does not run that test.

**Synthesis.** Every pipeline in this lineage either (a) targets a *different
output* (VQA text or a live map, not VG triplets), (b) operates in a
*different domain or sensor suite* (internet images, scans, RGB-D, not this
Spot dataset's monocular captures), or (c) is a *reference recipe* rather
than a deployable annotator. The geometry-to-label *idea* is established; its
application as a **fully-automatic seven-predicate annotator validated against
this dataset's human labels** is not.

## 2.6 Perception components and the open-vocabulary family

Our pipeline is geometry-first, but it stands on off-the-shelf perception, and
each component was chosen against alternatives from the recent literature. The
relevant design space, which is also the basis of the **detector-swap
ablation**, is:

- **Detection.** *Closed-set:* **YOLOv10m** (Wang, A. et al., 2024) is the
  dataset's own baseline over its six classes, at 0.93 mAP@50, and is what a
  replication with the authors' weights would use. *Open-vocabulary:*
  **Grounding DINO** (Liu et al., 2024) trades per-class sharpness for
  arbitrary text-named objects, which is what carries the pipeline beyond the
  six classes. Chapter 4's deployment mode uses it at a 0.25 box threshold
  precisely because it is the worst reasonable detector, so the end-to-end
  bound it produces (0.38 triplet recall, against 0.85 conditional on both
  endpoints being found) errs low.
- **Segmentation.** **SAM2** (Ravi et al., 2024), box-prompted. The
  silhouette is load-bearing twice over: depth is sampled by median over
  object pixels, not over the whole box, and the support rule's contact
  test needs the object's bottom boundary pixel by pixel (§3.5).
- **Monocular depth.** **Depth Anything v2** (Yang, L. et al., 2024) emits a
  *relative* map, ordering pixels without a unit, and that single property
  fixes the rule design: every depth comparison is ordinal and within-image,
  and no rule may consume an absolute distance.
- **Adjacent, not used.** **CLIP** (Radford et al., 2021) and **SCLIP**
  (Wang, F. et al., 2024), which reaches 38.2% zero-shot mIoU by making CLIP
  dense through a training-free *Correlative Self-Attention*, bear on
  open-vocabulary scaling and not on relation logic; **PrimitiveAnything** (Ye et al., 2025) decomposes 3D shapes
  into primitives and assumes clean 3D input this project does not have. All
  three are future directions.

The division of labour is deliberate. The neural components only *measure*,
and every relationship decision is made by an explicit rule over those
measurements, which is what makes the annotator auditable; §3.4 gives each
choice with the alternative it displaced, and Chapter 4's box-only ablation
quantifies what each actually contributes.

## 2.7 Learned scene-graph generation (the consumer of our output)

Scene-graph generation (SGG) models **predict** relationships from learned
visual patterns. The task predates the scene-graph framing (Lu et al., 2016,
whose language priors already leaned on label statistics, not
geometry), but the field's shape was set by **Visual Genome** (Krishna et
al., 2017), 108k images of crowdsourced triplets whose JSON format this
dataset and this project's writer inherit. Chang et al. (2023) survey the
lineage benchmarked on it and name annotation cost and label bias as its two
persistent constraints, which is this project's premise from the consumer's
side. **Neural Motifs** (Zellers et al., 2018) showed that context and label
statistics dominate: their frequency baseline, predicting the commonest
predicate for a pair while ignoring the image, proved hard to beat, a warning
that relation "accuracy" can be memorised co-occurrence, not
understood geometry. **VCTree** (Tang et al., 2019) composes dynamic tree
structures over objects and is the best model in the source paper's own
benchmark (mR@100 = 0.49). **Unbiased SGG** (Tang et al., 2020) showed
formally that such models absorb the *annotation distribution*, its long tail
and its biases, and proposed counterfactual debiasing. The field's corrective
direction is telling: panoptic scene graph generation (Yang, J. et al., 2022)
replaced box grounding with pixel-accurate masks after showing boxes
systematically mislocalise the objects whose relations are being learned,
the reasoning that puts SAM2 masks at the centre of this project's support
rule.

The lineage also fixes the evaluation vocabulary used in Chapters 4 and 7.
SGG models are scored by recall of the annotated triplets among their top K
ranked predictions: **R@K** pools all predicates (so frequent predicates
dominate), **mR@K** averages recall per predicate (so rare predicates count
equally), and **zero-shot recall (zR@K)** scores only subject–predicate–object
combinations never seen in training, isolating compositional generalisation
from memorisation (Tang et al., 2020). Two settings matter: **PredCls**
supplies ground-truth boxes and classes and asks only for the relations,
isolating the relation model, while **SGDet** requires detection, labelling
and relations end to end. This project adopts both conventions so its numbers
can be read against the source paper's tables.

Two lessons transfer directly. First, the models consuming this dataset's
labels are known bias-absorbers, so whatever the annotation carries (here,
measured inverted conventions and selective `near` usage; Chapter 4) becomes
the training signal. Cleaning the supply attacks the cause; debiasing the
model treats the symptom. Second, Motifs' frequency-baseline lesson dictates
this project's baseline discipline: every fidelity number in Chapter 4 is read
against trivial random/majority baselines for exactly this reason.

**REACT++** (Neau and Falomir, 2026) is a real-time SGG model with a YOLO
backbone, reportedly ~20% faster and ~10% more accurate on relation
prediction than its predecessor, small enough to run onboard a robot; it
ships in the open **SGG-Benchmark** framework. The essential point for
positioning this project: such models **require labelled training data and
therefore sit downstream of an annotator**. This project *computes* labels
from geometry and runs *before* any SGG model. It is the **supplier**, and
REACT++ is a natural **consumer**. Using REACT++/SGG-Benchmark to train on
our auto-labels versus the human labels is the heavyweight version of RQ2
(the lightweight classifier is the controlled main experiment); that test is
executed and reported in Chapter 6.

## 2.8 How the field measures success, and what those measures miss

The metrics introduced in §2.7 are not neutral instruments: each was adopted
to fix a defect in the one before it and carries a defect of its own.
Because Chapters 4 and 6 report results *in* these metrics and then argue
about what the results mean, the arguments belong here, established from the
literature and not improvised alongside the numbers.

**Recall without precision is a consequence of incomplete annotation, not a
choice.** The convention descends from visual relationship detection on
crowdsourced graphs (Lu et al., 2016; Krishna et al., 2017), where annotators
record a handful of the relations present in a scene and leave the rest
unmarked. An unannotated pair is not a negative but an unexamined one, so a
predicted relation absent from the gold cannot be scored wrong and precision
computed against such a gold is not precision at all. The field's response
was to drop it and rank by recall at K. The cost is a metric that cannot
distinguish a system predicting carefully from one predicting abundantly, and
the omission becomes acute exactly when the object of study is the annotation
itself: a method labelling the pairs the humans skipped is penalised for its
coverage under a precision reading and rewarded for it under a recall
reading, with neither number settling whether the extra labels are true.
Chang et al. (2023) name annotation cost and label bias as the field's
persistent constraints without resolving this measurement consequence of
them.

**A metric a context-free prior can saturate is measuring the prior.**
Zellers et al.'s (2018) frequency baseline predicts the commonest predicate
for an object pair with no access to the image and proved extremely hard to
beat on R@K, which is a statement about the metric at least as much as about
the models. **mR@K**, proposed by VCTree (Tang et al., 2019) and adopted as
standard after Tang et al. (2020), is the corrective: averaging recall per
predicate stops head classes from carrying the score. It introduces its
own sensitivity, however, because a predicate with few test instances now
weighs as much as one with thousands, so the aggregate can move on a handful
of triplets and, in a dataset annotated by different people in different
blocks, on which annotator happened to supply them. Per-predicate and
per-annotator decomposition is therefore not optional garnish; without it a
mean recall is uninterpretable, which is why Chapters 4 and 6 report both.

**Zero-shot recall measures something relative, and what it is relative to
must be stated.** zR@K scores only subject–predicate–object combinations
absent from *training*, isolating composition from memorisation. It comes
from visual relationship detection (Lu et al., 2016) and was first reported on
Visual Genome by Tang et al. (2020). The definition is well posed when one model is compared against
itself. It becomes ambiguous the moment two differently-supervised models are
compared in a single column, because the exclusion set is then drawn from one
shared reference, not from each model's own training data; what the
column reports is the extent to which each label source covers combinations
the reference omits. That is a real property, and for an annotation study
arguably the more interesting one, but it is not compositional
generalisation. Section 6.4 shows that this distinction is not hypothetical
for the present experiment, and reports the quantity under its accurate name.

**Setting and gold determine what a number means.** PredCls supplies
ground-truth boxes and classes, measuring the relation model in isolation and
producing systematically higher figures than SGDet, where detection errors
propagate; the two are frequently quoted side by side and are not comparable.
Underneath both sits the assumption that the gold is correct, which Northcutt,
Athalye and Mueller (2021) showed is false often enough to reorder published
rankings (§2.3). For spatial relations the annotation may also be
*consistently* wrong in the sense that matters here: where a group of
annotators applied a different reference frame, a system agreeing with them
scores well and a system that is right scores badly, and no recall metric can
tell the two apart.

Three commitments follow for this dissertation, each traceable to a defect
above. Recall against the human triplets is reported alongside an audited
estimate of true precision on pairs the gold never covers (§4.4), because
recall alone cannot separate coverage from correctness. Every aggregate is
decomposed per predicate and per annotator group, because means over
heterogeneous annotation hide exactly the effects being studied. And the
decisive test of the labels is deliberately moved *off* these metrics
altogether, to whether a downstream consumer trained on them performs the
task (Chapters 5 and 6), because a metric that rewards agreement with the
annotation cannot adjudicate a dispute about the annotation.

## 2.9 The case against a rule-based annotator

An honest review states the strongest version of the opposing argument, not
a version chosen because it is answerable. Four objections stand against the
method this project uses and a fifth against its premise, and they are set out
here so that later chapters can be read as attempts on them, not as a defence
assembled after the fact.

**Rules do not scale with the vocabulary.** Each predicate here is an
explicitly authored geometric test with fitted thresholds. Seven are
tractable; the literature routinely works with fifty (Krishna et al., 2017),
and the direction of travel Chang et al. (2023) identify is open-vocabulary
relations, including non-spatial ones such as *holding* for which no geometric
criterion exists. A learned predictor improves by being shown more data, a
rule set only by being extended by hand, so whatever this project demonstrates
about seven spatial predicates transfers to functional relations not at
all.

**Systematic error is worse for training than random error.** The appeal of
computed labels is consistency, but consistency guarantees only that mistakes
recur, and a rule's mistakes correlate with scene geometry and do not scatter at random. Tang et al. (2020) established how thoroughly SGG models
absorb the distribution of their supervision, a result that cuts both ways: a
model trained on rule output can learn the rule's blind spot as a property of
the world, and no amount of extra data averages it out. Independent human
noise is, in this narrow respect, the safer failure mode.

**Validating one's own labelling functions is circular.** Snorkel (Ratner et
al., 2017) anticipated this, treating labelling functions as noisy and
*estimating* their accuracies from the agreement structure among several
independent sources, precisely because an author's confidence in a rule is not
evidence about it. A single rule set verdicted by its own author has neither
multiple sources nor a generative model over them, and an audit by the person
who wrote the rules inherits their assumptions about what counts as correct.
Chapter 4 concedes this and reports the mitigations and their limits.

**The reference frame is a decision, not a fact.** *In front of* has no
answer independent of the frame in which it is asked: Landau and Jackendoff
(1993) set out the distinction between viewer-centred, object-centred and
environment-centred description, and RoboSpatial (Song et al., 2025)
maintains three frames explicitly instead of choosing one. A rule set must
commit to a convention, and where an annotator used a different one the two
will disagree systematically. Calling the rule correct in that situation is
an assertion about which convention should govern, not a measurement, and
the dissertation is obliged to argue for it, not assume it (§4.5).

A fifth objection is directed at the premise, not the method. If the
existing annotation is inconsistent, the direct remedy is better collection,
not cheaper labels; SpatialSense (Yang, K., Russakovsky and Deng, 2019) and
Rel3D (Goyal et al., 2020) both responded to defective relation annotation
by rebuilding the collection process. Automation makes labels cheap, which
is orthogonal to making the definitions right, and this project inherits the
dataset's definitions instead of improving them.

The dissertation answers the second and fourth objections empirically, in
Chapters 5 and 6 for systematic error and in Chapters 4 and 7 for the
reference frame, and it concedes the third while reporting what mitigation
was possible. The first, vocabulary scale, it does not answer at all and
records as a limitation (§9.3). The fifth is a different project.

## 2.10 Critical comparison and the research gap

Every neighbour below either targets a different output, is a reference
recipe and not an annotator, operates outside this dataset, or *predicts*
rather than *computes* relations. **None provides a fully-automatic annotator
for this dataset's seven predicates, validated against its human labels.**

| Work | Year/venue | Relations: compute vs. predict | Output | Auto-annotator? | This dataset's 7 predicates? | Validated vs. these human labels? |
|---|---|---|---|---|---|---|
| Wang et al. (source) | 2025 ACM MM | human-labelled | VG JSON / YOLO / h5 triplets | manual (SGDET-Annotate) | defines them | is the human label |
| SpatialVLM | 2024 CVPR | compute (geometry) | spatial VQA text | partial (QA gen) | no | no |
| VQASynth | 2024 | compute (geometry) | spatial VQA text | partial (QA gen) | no | no |
| SpatialRGPT | 2024 NeurIPS | compute (geometry+depth) | region-reasoning VLM | reference recipe | no | no |
| Open3D-VQA | 2025 ACM MM | compute (+correction) | 3D/aerial VQA | partial | no | no |
| RoboSpatial | 2025 CVPR | compute (geometry) | spatial QA (3 frames) | partial | no | no |
| REACT++ / SGG-Benchmark | 2026 | **predict** (learned) | scene-graph triplets | no (needs labels) | no | n/a |
| **This work** | 2026 | **compute (geometry)** | **VG JSON / YOLO / h5 triplets** | **yes, fully automatic** | **yes** | **yes (RQ1)** |

Two columns isolate the contribution. The *"this dataset's 7 predicates"*
column shows only Wang et al. and this work address them, and Wang et al. do
so manually. The *"validated vs. these human labels"* column shows only this
work quantifies agreement with the human consensus on the same images. The
geometry-to-label *method* is borrowed and well-precedented; its
instantiation as a validated automatic annotator for this dataset is new.

## 2.11 Summary and positioning

The literature establishes that spatial relations are computable from
geometry, that learned SGG absorbs the biases of the triplets it consumes, and
that dense rule-based supervision substitutes for scarce human labels when
validated carefully. It leaves a precise gap: no automatic, geometry-based
annotator emits this dataset's seven predicates in its native formats and is
validated against its human labels, even though its authors ask for exactly
such fixes.

Two things follow from the critical sections. Because the field's metrics are
recall-shaped by its own incomplete annotation (§2.8), no number from them can
settle a dispute *about* that annotation, so the protocol of Chapters 4 to 6
is built around that limitation, not inside it; and because the
strongest case against the approach is stated in advance (§2.9), the results
can be read as an attempt on it, with §7.7 the reckoning. Chapter 3 develops
the design the gap calls for: explicit geometric rules over measured
perception, and not a learned predictor that would re-import the very
labelling bias in question.
