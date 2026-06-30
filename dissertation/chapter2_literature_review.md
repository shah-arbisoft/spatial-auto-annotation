# Chapter 2 — Literature Review

> Draft, Week 1 (expanded). The marking scheme weights this chapter at 20% and
> rewards a **critical comparison** that exposes a gap, not a catalogue. The
> structure and the comparison table in §2.6 are the analytical spine.
>
> **Citation status.** Facts attributed to the source paper are verified against
> its arXiv full text (2506.12525). SpatialVLM, SpatialRGPT, RoboSpatial,
> Open3D-VQA, VQASynth, SCLIP, PrimitiveAnything and REACT++ are verified against
> their papers/repos (links at the end). Page/section numbers and formal
> bibliography entries are still to be added in the writing weeks.

## 2.1 Scene graphs and spatial relationships in robotics

A robot acting in a human environment must represent not only *what* objects are
present but *how they are spatially arranged*. The structured representation for
this is the **scene graph**: objects as nodes and pairwise relationships as
labelled, directed edges (subject → predicate → object). Spatial predicates —
*on*, *under*, *left/right of*, *in front of / behind*, *near* — are the edges
that matter for manipulation, navigation and instruction following, because they
encode the geometry an agent must respect to act. This review concentrates on how
such spatial edges are **produced**: by hand, by learned prediction, or — as this
project argues — by computation from measured geometry.

## 2.2 The source dataset and its annotation bottleneck

Wang et al. introduce *A Spatial Relationship Aware Dataset for Robotics*
(ACM MM 2025), captured by a Boston Dynamics Spot robot. Verified specifics from
the paper:

- **Scale:** "nearly 1,000 robot-acquired indoor images" collected; after quality
  control "approximately 900" remain. (The released GitHub subset used here
  contains 884 images / **838 annotated** — see [DATASET_NOTES.md](../docs/DATASET_NOTES.md).)
- **Annotation:** "Nine trained annotators worked independently in batches of 100
  images," using the manual **SGDET-Annotate** tool (every box drawn, every class
  and relationship clicked), with a majority-vote cleaning pass. The released data
  is organised into nine groups of ~100, matching this exactly.
- **Vocabulary:** exactly seven predicates — *behind, in front of, on, to the left
  of, to the right of, under, near* — over six object classes (book, bottle, box,
  cube, human, remote).
- **Detector baseline:** a YOLOv10m backbone reaches ≈0.92 precision, 0.90 recall
  and **0.93 mAP@50** (mAP@50-95 ≈0.68).

Three limitations the authors themselves flag motivate this project. First,
**early saturation**: "all predictors reached their peak mR@100 well before the
final epoch," which they attribute to the fact that "the dataset's limited
diversity exhausts relational learning capacity early." Second, the **`near`
predicate is unreliable**: they report "inconsistencies, particularly with the
'near' predicate," and it "remains challenging for all models (0.2247–0.2494)."
Third, their future-work prescription is explicit: "augment under-represented
relations while enforcing clear annotation guidelines (e.g., spatial thresholds
for 'near')."

The common cause is the **manual annotation bottleneck**: SGDET-Annotate only
*accelerates* human labelling — a human still decides every edge — so the dataset
cannot grow cheaply, diversity stays low, and inter-annotator disagreement
(worst on `near`) is baked in. **No automatic annotator exists for this dataset.**
This is the gap the project fills, and the fitted `near` threshold is a direct,
data-driven realisation of the authors' own "spatial thresholds for near."

## 2.3 Geometry-to-label pipelines (the lineage we build on)

A line of work *computes* spatial facts from perceived geometry rather than
predicting them from learned relational patterns. This is the methodological
ancestry of the present project, but each member outputs something other than a
scene-graph annotation for this dataset's seven predicates.

- **SpatialVLM** (Chen et al., CVPR 2024) is the foundational geometry-to-label
  method. It builds an *automatic* 3D spatial-VQA data-generation framework —
  lifting internet images to metric 3D via monocular depth and segmentation, then
  emitting up to ~2B spatial question–answer pairs from ~10M images. It
  establishes the central premise we adopt: spatial relations can be *derived from
  measured geometry without human relational labels*. But its output is
  **free-form VQA text**, its domain is **internet images**, and it targets no
  fixed predicate set — it is not an annotator for a robot scene-graph dataset.

- **SpatialRGPT** (Cheng et al., NeurIPS 2024) extends this with a data-curation
  pipeline that learns regional representations from **3D scene graphs** and a
  **depth "plugin"** that injects relative depth (via Depth Anything) into a VLM's
  visual encoder. It is the cleanest published RGB→depth→relation recipe and a key
  reference for our depth use, but the artefact is a **region-reasoning VLM**, not
  a deterministic labeller producing VG-format triplets.

- **VQASynth** (remyxai) is an open reproduction of the SpatialVLM pipeline: a
  chain of expert models (SAM2 for localisation refinement, monocular depth, and
  grounded captioning) that infers spatial relationships to **create a spatial-VQA
  dataset**. It is the most reusable code reference; its output stage produces
  **QA pairs**, which we would replace with a scene-graph-triplet writer. (Its
  depth backend has shifted over time — e.g. DepthPro → VGGT — so we pin our own.)

- **Open3D-VQA** (Zhang et al., ACM MM 2025; arXiv 2503.11094) is the source of
  our **error-correction** idea. It is an embodied spatial-reasoning benchmark for
  *open/aerial* space (89k QA pairs over seven spatial tasks, visual + point
  cloud) whose generation pipeline "extracts 3D spatial relationships from a
  single RGB image" with a **multi-modal correction flow to ensure quality**. We
  adapt that correction principle — rejecting geometrically impossible labels —
  into our correction step (spec §8), in a different domain (indoor tabletop, not
  aerial) and for a different output (triplets, not QA).

- **RoboSpatial** (Song et al., CVPR 2025, oral) is the closest robotics-domain
  match, cited by the source paper. It teaches spatial understanding to 2D/3D VLMs
  from real indoor/tabletop scans (≈1M images, ≈3M annotated spatial relations).
  Critically for our design, it formalises **reference frames** — ego-centric,
  world-centric, object-centric — which directly justifies our explicit choice to
  express *left/right* in the **camera (ego) frame** (predicate spec §0). Its
  output is spatial QA over three frames, not a fixed seven-predicate annotation.

**Synthesis.** Every pipeline in this lineage either (a) targets a *different
output* (VQA text, not VG triplets), (b) operates in a *different domain*
(internet/scan data, not this Spot dataset), or (c) is a *reference recipe* rather
than a deployable annotator. The geometry-to-label *idea* is established; its
application as a **fully-automatic seven-predicate annotator validated against
this dataset's human labels** is not.

## 2.4 Perception components and the open-vocabulary family

Our pipeline is geometry-first, but it stands on off-the-shelf perception. The
relevant design space — and the basis of the **detector-swap ablation** — is:

- **Detection.** *Closed-set:* **YOLOv10m**, the dataset's own baseline (six
  classes; 0.93 mAP@50). *Open-vocabulary:* **GroundingDINO** detects arbitrary
  text-named objects, enabling extension beyond the six classes. The ablation
  contrasts these to quantify how detector choice propagates into relation
  fidelity.
- **Open-vocabulary classification/segmentation (CLIP family).** **CLIP** (Radford
  et al., 2021) gives zero-shot image–text matching; **SCLIP** (Wang et al.,
  ECCV 2024) makes CLIP dense via a training-free *Correlative Self-Attention*,
  reaching 38.2% zero-shot mIoU for open-vocabulary segmentation. These matter for
  the **open-vocabulary scaling** direction (labelling object types beyond the six
  classes) but add nothing to the *relation* logic; for the closed-set core they
  are positioned as **alternatives and future work**, not components.
- **Segmentation.** **SAM2** turns each detector box into a precise mask
  (box-prompted), which we use for centroids and masked depth sampling.
- **Monocular depth.** **Depth Anything v2 (Small)** yields a *relative* depth map
  on the 6 GB GPU; relativity restricts depth comparisons to be **ordinal and
  within-image**, a limitation carried into the critical chapter.
- **3D primitive abstraction (out of scope).** **PrimitiveAnything**
  (Ye et al., SIGGRAPH 2025) decomposes *3D shapes* into primitive assemblies
  (cuboids, cylinders, ellipsoids). It assumes clean 3D input we do not have
  (only monocular RGB + relative depth), so it is noted only as a speculative
  future representation, not used here.

## 2.5 Learned scene-graph generation (the consumer of our output)

Scene-graph generation (SGG) models **predict** relationships from learned visual
patterns. **REACT++** (Neau & Falomir, 2026) is a real-time SGG model with a YOLO
backbone, reportedly ~20% faster and ~10% more accurate on relation prediction
than its predecessor, small enough to run onboard a robot; it ships in the open
**SGG-Benchmark** framework. The essential point for positioning this project:
such models **require labelled training data and therefore sit downstream of an
annotator**. This project *computes* labels from geometry and runs *before* any
SGG model — it is the **supplier**, and REACT++ is a natural **consumer**. Using
REACT++/SGG-Benchmark to train on our auto-labels versus the human labels is the
*optional* heavyweight version of RQ2 (the lightweight classifier is the
controlled main experiment); it is deferred unless the core lands early.

## 2.6 Critical comparison and the gap

The table is the analytical core: it shows every neighbour either targets a
different output, is a reference recipe rather than an annotator, operates outside
this dataset, or *predicts* rather than *computes* relations — and that **none
provides a fully-automatic annotator for this dataset's seven predicates,
validated against its human labels**.

| Work | Year/venue | Relations: compute vs. predict | Output | Auto-annotator? | This dataset's 7 predicates? | Validated vs. these human labels? |
|---|---|---|---|---|---|---|
| Wang et al. (source) | 2025 ACM MM | human-labelled | VG JSON / YOLO / h5 triplets | manual (SGDET-Annotate) | defines them | is the human label |
| SpatialVLM | 2024 CVPR | compute (geometry) | spatial VQA text | partial (QA gen) | no | no |
| VQASynth | — | compute (geometry) | spatial VQA text | partial (QA gen) | no | no |
| SpatialRGPT | 2024 NeurIPS | compute (geometry+depth) | region-reasoning VLM | reference recipe | no | no |
| Open3D-VQA | 2025 ACM MM | compute (+correction) | 3D/aerial VQA | partial | no | no |
| RoboSpatial | 2025 CVPR | compute (geometry) | spatial QA (3 frames) | partial | no | no |
| REACT++ / SGG-Benchmark | 2026 | **predict** (learned) | scene-graph triplets | no (needs labels) | no | n/a |
| **This work** | 2026 | **compute (geometry)** | **VG JSON / YOLO / h5 triplets** | **yes, fully automatic** | **yes** | **yes (RQ1)** |

Two columns isolate the contribution. The *"this dataset's 7 predicates"* column
shows only Wang et al. and this work address them — and Wang et al. do so
manually. The *"validated vs. these human labels"* column shows only this work
quantifies agreement with the human consensus on the same images. The
geometry-to-label *method* is borrowed and well-precedented; its instantiation as
a validated automatic annotator for this dataset is new.

## 2.7 Summary and positioning

The literature establishes (i) that spatial relations are **computable from
geometry** (SpatialVLM and its lineage), (ii) that **depth-grounded region
reasoning** works (SpatialRGPT, RoboSpatial), and (iii) that **learned SGG**
consumes labelled triplets (REACT++). It also leaves a precise gap: there is no
automatic, geometry-based annotator that emits this robot dataset's seven spatial
predicates in its native formats and is validated against its human labels — even
though the dataset's authors explicitly ask for automation-friendly fixes
("spatial thresholds for near," augmenting under-represented relations).

Because the seven predicates are spatial, the appropriate instrument is **explicit
geometric rules over measured perception**, not a learned relation predictor that
would merely re-import human labelling bias. This motivates the design developed
in Chapter 3.

---

## Reference links (to formalise into the bibliography)
- Source paper (ACM MM 2025): https://doi.org/10.1145/3746027.3758293 · full text https://arxiv.org/abs/2506.12525
- SpatialVLM (CVPR 2024): https://arxiv.org/abs/2401.12168
- SpatialRGPT (NeurIPS 2024): https://arxiv.org/abs/2406.01584 · https://github.com/AnjieCheng/SpatialRGPT
- RoboSpatial (CVPR 2025): https://arxiv.org/abs/2411.16537 · https://github.com/NVlabs/RoboSpatial
- SCLIP (ECCV 2024): https://arxiv.org/abs/2312.01597 · https://github.com/wangf3014/SCLIP
- CLIP (2021): https://github.com/openai/CLIP
- GroundingDINO; SAM2; Depth Anything v2 — building blocks (add canonical citations)
- PrimitiveAnything (SIGGRAPH 2025): https://arxiv.org/abs/2505.04622
- REACT++ (2026): https://arxiv.org/abs/2603.06386 · SGG-Benchmark https://github.com/Maelic/SGG-Benchmark
- VQASynth (remyxai): https://github.com/remyxai/VQASynth
- Open3D-VQA (ACM MM 2025): https://arxiv.org/abs/2503.11094 · https://github.com/EmbodiedCity/Open3D-VQA.code
