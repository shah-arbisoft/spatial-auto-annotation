# Chapter 2: Literature Review

> Facts attributed to the source paper are verified against its arXiv full
> text (2506.12525); all other cited works are verified against their published
> versions. Full bibliography entries: [references.md](references.md).

This chapter reviews the work this project builds on and the work it must be
distinguished from. Its organising question is **label quality**, because
that is what the project's evidence ultimately turns on: not whether spatial
relations can be computed, which the literature already settles, but whether
computed labels are better or worse than the human labels they replace, and
how anyone would know.

The chapter therefore moves from the representation (scene graphs, §2.1) to
the dataset whose annotation bottleneck motivates the work (§2.2); then to
the literature on what makes labels good or bad, covering weak supervision,
annotator disagreement, and the statistics used to measure agreement (§2.3);
then to the rival remedies that stretch scarce labels rather than replacing
them, semi-supervised and active learning, with the argument for why they do
not fit this dataset (§2.4). Only then does it turn to the approach this
project takes and its lineage (§2.5), the perception components it stands on
(§2.6), and the learned models that consume its output (§2.7). Section 2.8
states the research gap as a critical comparison, and §2.9 positions the
work against it.

## 2.1 Scene graphs and spatial relationships in robotics

A robot acting in a human environment must represent not only *what* objects are
present but *how they are spatially arranged*. The structured representation for
this is the **scene graph**: objects as nodes and pairwise relationships as
labelled, directed edges (subject → predicate → object). Spatial predicates
(*on*, *under*, *left/right of*, *in front of / behind*, *near*) are the edges
that matter for manipulation, navigation and instruction following, because they
encode the geometry an agent must respect to act. A planner told only "cube,
book, table" cannot decide what to move first; a planner told "the cube is on
the book" can. This review concentrates on how such spatial edges are
**produced**: by hand, by learned prediction, or, as this project argues, by
computation from measured geometry.

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
**early saturation**: "all predictors reached their peak mR@100 well before the
final epoch," which they attribute to the fact that "the dataset's limited
diversity exhausts relational learning capacity early." Second, the **`near`
predicate is unreliable**: they report "inconsistencies, particularly with the
'near' predicate," and it "remains challenging for all models (0.2247–0.2494)."
Third, their future-work prescription is explicit: "augment under-represented
relations while enforcing clear annotation guidelines (e.g., spatial thresholds
for 'near')."

The common cause is the **manual annotation bottleneck**. SGDET-Annotate only
*accelerates* human labelling (a human still decides every edge), so the dataset
cannot grow cheaply, diversity stays low, and inter-annotator disagreement
(worst on `near`) is baked in. **No automatic annotator exists for this dataset.**
This is the gap the project fills, and the fitted `near` threshold is a direct,
data-driven realisation of the authors' own "spatial thresholds for near."

## 2.3 Label quality: weak supervision and annotator disagreement

The project's premise, replacing scarce human labels with dense computed ones,
has an established name: **weak supervision**. **Snorkel** (Ratner et al.,
2017) formalised *data programming*: instead of labelling examples, experts
write labelling functions (heuristics, rules, distant supervision) whose
noisy, overlapping votes are combined into training labels, trading per-label
human authority for coverage and consistency. Its deployments repeatedly
matched or beat hand-labelled baselines wherever the labelled set, not the
model, was the bottleneck. This project's geometric rules are labelling
functions in precisely that sense: deterministic, auditable and dense, with two
departures from the Snorkel setting. Measured geometry gives near-exact rather
than noisy votes for most predicates (the audit estimates true precision ≈ 1.0
for five of seven), so no probabilistic label aggregation is needed; and the
computed labels are *validated against* the human labels they replace (RQ1)
rather than assumed comparable.

The complementary literature dismantles the premise that human annotation is a
single reliable gold standard. **Uma et al.'s (2021) survey of learning from
disagreement** documents systematic annotator disagreement across vision and
language tasks, driven by ambiguous guidelines, subjective category
boundaries and annotator-specific conventions, and reviews methods that treat
disagreement as signal rather than noise. That frame fits this dataset
exactly: Chapter 4 measures three annotator behaviours (selective `near`
usage, an inverted front/behind convention in two groups, one-directional
support labelling) which make "agreement with the humans" a per-annotator
rather than a global quantity. This motivates two design decisions taken in
Chapters 3–4: evaluation is reported per annotator group, never only pooled;
and thresholds are calibrated only on annotators who actually used a label,
with other annotators held out.

Where human judgements must themselves be evaluated, the measurement
tradition supplies the instruments. Chance-corrected agreement between two
verdict sets is measured by **Cohen's kappa** (Cohen, 1960), and the
reliability of a pool of raters by chance-corrected coefficients such as
Krippendorff's alpha; Artstein and Poesio (2008) survey both and their
pitfalls, including the prevalence effect that depresses kappa when one
answer dominates. Chapter 4's independent validation of the automatic labels
(crowd judgements of sampled predictions, scored against the author's own
audit verdicts) applies exactly these instruments. It also sharpens RQ2 into
a question the weak-supervision literature predicts but rarely tests this
directly: can consistent computed labels *out-teach* inconsistent human ones
on the humans' own held-out annotations?

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

Applied here, the recipe would be: train a relation model on the ~10% of
pairs the annotators labelled, pseudo-label the remaining 90%, retrain. Three
properties of this dataset argue against it, each measured in this
dissertation rather than assumed. First, self-training *amplifies its seed*:
the pseudo-labels inherit whatever the seed model learned, and the seed here
is sparse and internally inconsistent (selective `near` usage, two inverted
front/behind conventions, one-directional support; §2.2, Chapter 4). A
teacher trained on contradictory conventions teaches its student the same
contradictions, with added confidence. Second, the seed is not merely small
but *selectively* small: annotators labelled the pairs they found salient,
so the labelled 10% is not an unbiased sample of the 90% to be filled in,
violating the assumption under which pseudo-labelling is well behaved.
Third, the empirical anchor: Chapter 5's human-trained classifier, which is
precisely the seed model such a loop would start from, collapses on the
sparsely-labelled predicates (recall 0.08–0.25) and is unstable across
seeds. A self-training loop built on that teacher has nothing reliable to
amplify. Active learning fails differently: it still buys *human* labels,
so it reduces the bottleneck's slope without removing it, and it cannot fix
inconsistency between annotators, only ration it.

The geometric route sidesteps all three failure modes because its labelling
function does not derive from the flawed seed at all: the rules are fitted to
a handful of thresholds (with the fit itself validated on held-out
annotators) and are exactly as consistent on the 90% as on the 10%. The
comparison is not merely argued: RQ2 trains the same model on each label
source and measures which teaches better, which is the head-to-head test the
semi-supervised literature rarely runs against a programmatic labeller.

## 2.5 Geometry-to-label pipelines (the lineage we build on)

A line of work *computes* spatial facts from perceived geometry rather than
predicting them from learned relational patterns. This is the methodological
ancestry of the present project, but each member outputs something other than a
scene-graph annotation for this dataset's seven predicates.

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

- **VQASynth** (Remyx AI, 2024) is an open reproduction of the SpatialVLM
  pipeline: a chain of expert models (SAM2 for localisation refinement, monocular
  depth, and grounded captioning) that infers spatial relationships to **create a
  spatial-VQA dataset**. It is the most reusable code reference; its output stage
  produces **QA pairs**, which we would replace with a scene-graph-triplet
  writer. (Its depth backend has shifted over time, e.g. DepthPro → VGGT, so we
  pin our own.)

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
  world-centric and object-centric readings of the same spatial phrase. The same
  sentence ("the cup is left of the box") is true in one frame and false in
  another, so an annotator must pin the frame before any label is well defined.
  This directly justifies our explicit choice to express *left/right* in the
  **camera (ego) frame** (predicate spec §0), which is also the frame the
  dataset's annotators saw on screen. Its output is spatial QA over three
  frames, not a fixed seven-predicate annotation.

**Synthesis.** Every pipeline in this lineage either (a) targets a *different
output* (VQA text, not VG triplets), (b) operates in a *different domain*
(internet/scan data, not this Spot dataset), or (c) is a *reference recipe* rather
than a deployable annotator. The geometry-to-label *idea* is established; its
application as a **fully-automatic seven-predicate annotator validated against
this dataset's human labels** is not.

## 2.6 Perception components and the open-vocabulary family

Our pipeline is geometry-first, but it stands on off-the-shelf perception, and
each component was chosen against alternatives from the recent literature. The
relevant design space, which is also the basis of the **detector-swap
ablation**, is:

- **Detection.** *Closed-set:* **YOLOv10m** (Wang, A. et al., 2024), the
  dataset's own baseline (six classes; 0.93 mAP@50). YOLOv10 removes the
  non-maximum-suppression stage of earlier YOLO versions with a consistent
  dual-assignment training scheme, which is why it can run in real time onboard
  a robot. *Open-vocabulary:* **Grounding DINO** (Liu et al., 2024) marries a
  DETR-style detector with grounded language pre-training, so it detects
  arbitrary text-named objects at some cost in per-class sharpness. This
  enables extension beyond the six classes; the deployment-mode experiment in
  Chapter 4 uses it precisely because it is the worst reasonable case, so the
  detection-quality bound it produces is conservative.
- **Segmentation.** **SAM2** (Ravi et al., 2024) is a promptable segmentation
  model: given a box prompt it returns the pixel mask of the object inside it,
  and it is trained to do so for arbitrary objects rather than a fixed class
  list. The pipeline uses it to turn each detector box into a silhouette, which
  matters twice: depth is sampled only from object pixels rather than the
  background inside the box, and the support rule's contact test (Chapter 3)
  needs to know where an object's bottom edge actually runs, pixel by pixel.
- **Monocular depth.** **Depth Anything v2** (Yang et al., 2024) is a monocular
  depth estimator distilled from a large teacher trained on synthetic data and
  pseudo-labelled real images. Its output is *relative*, not metric: it orders
  pixels by distance but assigns no unit. That property shapes the rule design
  in Chapter 3: depth comparisons are **ordinal and within-image**, and no rule
  may use an absolute distance. The Small variant runs in under a gigabyte of
  VRAM, which keeps the whole pipeline on a 6 GB consumer GPU; Chapter 4's
  ablation A8 tests whether the larger Base variant would change the results.
- **Open-vocabulary classification/segmentation (CLIP family).** **CLIP**
  (Radford et al., 2021) gives zero-shot image–text matching; **SCLIP**
  (Wang, F. et al., 2024) makes CLIP dense via a training-free *Correlative
  Self-Attention*, reaching 38.2% zero-shot mIoU for open-vocabulary
  segmentation. These matter for the **open-vocabulary scaling** direction
  (labelling object types beyond the six classes) but add nothing to the
  *relation* logic; for the closed-set core they are positioned as
  **alternatives and future work**, not components.
- **3D primitive abstraction (out of scope).** **PrimitiveAnything**
  (Ye et al., 2025) decomposes *3D shapes* into primitive assemblies
  (cuboids, cylinders, ellipsoids). It assumes clean 3D input we do not have
  (only monocular RGB + relative depth), so it is noted only as a speculative
  future representation, not used here.

The division of labour is deliberate: the neural components above only
*measure* (where an object is, which pixels belong to it, how far away it
looks), and every relationship decision is made by an explicit rule over those
measurements. Chapter 3 argues this split is what makes the annotator
auditable, and Chapter 4's box-only ablation quantifies what each perception
component actually contributes.

## 2.7 Learned scene-graph generation (the consumer of our output)

Scene-graph generation (SGG) models **predict** relationships from learned visual
patterns. The field's shape was set by **Visual Genome** (Krishna et al., 2017),
108k images with crowdsourced relationship triplets, whose JSON format this
dataset (and this project's output writer) inherits, and by the model lineage
benchmarked on it. **Neural Motifs** (Zellers et al., 2018) demonstrated
that global context and label statistics dominate relation prediction: their
frequency baseline (predict the most common predicate for a given object pair,
ignoring the image) proved notoriously hard to beat, a warning that relation
"accuracy" can be memorised co-occurrence rather than understood geometry.
**VCTree** (Tang et al., 2019) composes dynamic tree structures over
objects to capture context, and is the best-performing model in the source
paper's own benchmark (mR@100 = 0.49). **Unbiased SGG** (Tang et al., 2020)
then showed formally that models trained on crowdsourced scene graphs largely
absorb the *annotation distribution*, its long tail and its biases, and
proposed counterfactual debiasing to recover the visual signal.

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
backbone, reportedly ~20% faster and ~10% more accurate on relation prediction
than its predecessor, small enough to run onboard a robot; it ships in the open
**SGG-Benchmark** framework. The essential point for positioning this project:
such models **require labelled training data and therefore sit downstream of an
annotator**. This project *computes* labels from geometry and runs *before* any
SGG model. It is the **supplier**, and REACT++ is a natural **consumer**. Using
REACT++/SGG-Benchmark to train on our auto-labels versus the human labels is the
heavyweight version of RQ2 (the lightweight classifier is the controlled main
experiment); that test is executed and reported in Chapter 6.

## 2.8 Critical comparison and the research gap

The table is the analytical core: it shows every neighbour either targets a
different output, is a reference recipe rather than an annotator, operates outside
this dataset, or *predicts* rather than *computes* relations, and that **none
provides a fully-automatic annotator for this dataset's seven predicates,
validated against its human labels**.

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

Two columns isolate the contribution. The *"this dataset's 7 predicates"* column
shows only Wang et al. and this work address them, and Wang et al. do so
manually. The *"validated vs. these human labels"* column shows only this work
quantifies agreement with the human consensus on the same images. The
geometry-to-label *method* is borrowed and well-precedented; its instantiation as
a validated automatic annotator for this dataset is new.

## 2.9 Summary and positioning

The literature establishes (i) that spatial relations are **computable from
geometry** (SpatialVLM and its lineage), (ii) that **depth-grounded region
reasoning** works (SpatialRGPT, RoboSpatial), (iii) that **learned SGG**
consumes labelled triplets and absorbs their biases (Visual Genome lineage,
REACT++), (iv) that **dense rule-based supervision is a proven substitute
for scarce human labels** when validated carefully (Snorkel; the disagreement
literature), and (v) that the standard annotation-stretching remedies,
active and semi-supervised learning, presuppose a consistent labelled seed
this dataset does not provide (§2.4). It also leaves a precise gap: there is no
automatic, geometry-based annotator that emits this robot dataset's seven spatial
predicates in its native formats and is validated against its human labels, even
though the dataset's authors explicitly ask for automation-friendly fixes
("spatial thresholds for near," augmenting under-represented relations).

Because the seven predicates are spatial, the appropriate instrument is **explicit
geometric rules over measured perception**, not a learned relation predictor that
would merely re-import human labelling bias. This motivates the design developed
in Chapter 3.

Full citation details for every work discussed in this chapter are given in
[references.md](references.md).
