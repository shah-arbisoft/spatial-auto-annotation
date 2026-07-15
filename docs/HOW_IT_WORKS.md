# How the annotator works, from pixels to relationships

This is the plain-language walkthrough of the whole pipeline: what happens to
an image from the moment it is loaded to the moment the tool writes
"bottle2 — in front of — book5". The formal definitions live in
`predicate_spec.md`; every rule described here is implemented in
`src/predicates.py` and covered by the unit tests.

---

## 1. The big picture

The tool is an assembly line with two halves:

```
            PERCEPTION (neural networks measure)          RULES (geometry decides)
image ──> boxes ──> masks ──> depth ──> one 3D point ──> seven yes/no questions ──> labels
```

The neural networks (SAM2, Depth Anything) only ever *measure* things — where
an object's pixels are, how far away each pixel looks. They never choose a
label. Every relationship is decided by an explicit geometric rule with a
threshold you can read in `configs/default.yaml`. That split is the whole
design: the measuring part is replaceable, the deciding part is auditable.

## 2. Step by step

### 2.1 Boxes — where are the objects?

Every relationship is between two objects, so the tool first needs a box and
a class name per object. Two sources:

- **Study mode**: the dataset's own hand-drawn boxes. Used for all accuracy
  results, so the evaluation tests the rules, not a detector.
- **Deployment mode**: a detector finds the boxes (GroundingDINO from text
  prompts, or a trained YOLO). Fully automatic, quality depends on detection.

One trap discovered early: the dataset's images are stored rotated 180°
(an EXIF flag viewers apply silently). The loader corrects this; without the
fix the pixels are upside down while the boxes are upright, and every
depth/mask measurement is nonsense.

### 2.2 Masks — which pixels belong to the object?

A box is a rectangle; the object isn't. SAM2 takes each box and returns the
actual pixel silhouette (the mask). Masks matter for two things: sampling
depth only from the object (not the background inside its box), and the
contact test for `on`/`under` (§3.3), which needs to know where an object's
bottom edge actually is, pixel by pixel.

### 2.3 Depth — how far is each pixel?

Depth Anything v2 turns the image into a depth map: every pixel gets a number
for how far away it looks. Two facts about this map shape several rules:

- It is **relative**, not metric. It says "this pixel is farther than that
  one", never "1.4 metres". So depths can be compared inside one image but a
  fixed distance threshold like "30 cm" is impossible.
- The raw model output is larger-is-nearer; the pipeline flips it so that
  **smaller = nearer**, checked by a test.

Each object's distance is the **median** depth over its mask. Median, not
mean, so a few background pixels leaking into the mask edge cannot drag the
estimate.

### 2.4 The 3D point

Each object is reduced to one point: the mask centroid (x, y, both scaled to
0–1 so thresholds work at any resolution) plus the median depth (z). All
seven questions below are asked about pairs of these points, plus the boxes
and masks where a rule needs them.

## 3. The seven questions

For every ordered pair of objects (A, B), the tool asks seven yes/no
questions. A worked example runs through this section: a bottle at centre
x = 0.35 with depth 0.42, and a book at centre x = 0.61 with depth 0.44.

### 3.1 "Is A to the left of B?" (and right of)

Compare the horizontal centres, in the camera frame — left means left *on the
screen*, because that is the view the human annotators worked in:

```
left_of(A, B)  =  B.x − A.x > 0.02
```

Bottle vs book: 0.61 − 0.35 = 0.26 > 0.02 → "bottle to the left of book"
(and automatically "book to the right of bottle"). If the centres are within
0.02 of each other — about 2% of the image width — the tool refuses to call
it and flags the pair instead. That band exists because when two centres
nearly coincide, humans themselves disagree about left/right.

### 3.2 "Is A in front of B?" (and behind) — a two-stage decision

**Stage 1 — depth.** Compare the two median depths:

```
in_front_of(A, B)  =  B.depth − A.depth > 0.03
```

Bottle vs book: 0.44 − 0.42 = 0.02. That is *inside* the 0.03 band — the
depth model cannot reliably separate two objects this close in distance, so
stage 1 abstains. This band is the honest response to relative depth's
noise; it was swept in the ablations and the trade-off is documented.

**Stage 2 — the ground-plane fallback.** Most abstentions are still
decidable, without depth at all. Two objects standing on the same floor obey
simple perspective: **the nearer object's box reaches lower in the image.**
Feet of a near person are lower on screen than feet of a far person. So:

```
if bottle.box_bottom − book.box_bottom > 0.005  →  bottle is in front
```

Say the bottle's box bottom is at y = 0.80 and the book's at y = 0.60: the
bottle's base sits 20% of the screen lower → the bottle is nearer → "bottle
in front of book". This cue is pixel-precise exactly where depth is noisy.

The fallback has a guard, because its assumption is "both objects stand on
the floor". If either object is **on top of something** — which the tool
knows from its own contact evidence (§3.3) — its box bottom marks where its
*support* is, not where it is, and the fallback stays silent. No masks → no
guard evidence → no fallback. If both stages abstain, the pair is flagged
`depth_ambiguous` and no front/behind label is written.

Why not just use a bigger depth network? It was measured (ablation A8): a 4×
larger model changes front/behind recall by +0.001/+0.002. The limit is
monocular ambiguity — one camera genuinely cannot rank two objects at almost
the same distance — which is why the fix that worked is geometric.

### 3.3 "Is A on B?" (and under) — three conditions at once

Being "on" something is physical, so the rule demands three physical facts:

1. **Above**: A's centroid is higher in the image than B's.
2. **Resting**: walk along the bottom edge of A's mask; at least 60% of those
   columns must have B's mask directly beneath (within 5 pixels). This is
   what "sitting on" looks like in pixels, and it separates a real stack from
   two objects merely printed near each other.
3. **Same distance**: |A.depth − B.depth| ≤ 0.06. This kills the classic
   illusion — on a floor, *farther* projects as *higher*, so a cube behind a
   remote looks exactly like a cube on a remote in 2D. Truly stacked objects
   share a camera distance; a behind-pair does not.

`under` is defined as the mirror of `on`, so the two can never contradict.
One class rule sits on top: support is never evaluated when either object is
a person. Measured reason: the annotators never once labelled person-support
(0 of 2,466 gold triplets), and pixel contact cannot tell a held remote from
a resting one.

### 3.4 "Is A near B?"

"Near" depends on scale: two sofas a metre apart are next to each other; two
coins a metre apart are not. So the metric is relative:

```
near(A, B)  =  (edge-to-edge gap between the boxes) / (average object size)  ≤  1.372
```

Size is the square root of box area; the gap is straight-line distance
between the closest box edges (zero if they touch or overlap). Two books with
a 20-pixel gap and ~150-pixel size: 20/150 = 0.13 ≤ 1.372 → near.

The threshold 1.372 was not invented — it was **fitted** to the human labels
(§4). One exclusion, also measured from the data: a pair already labelled
on/under is never additionally near — in the entire human annotation, `near`
co-occurs with a support label on 0 of 469 opportunities. The annotators used
near to mean "next to, but not touching-as-support", and the rule follows
their convention.

## 4. Where the thresholds come from

Every number above (0.02, 0.03, 0.005, 0.06, 60%, 1.372) lives in one config
file, and none was chosen by feel:

- The dataset's nine annotator groups were split: **groups 0–5 for fitting,
  6–8 held out** and never touched by any calibration.
- `near_T` was fitted by sweeping the threshold and taking the tightest value
  that recovers everything the (fitting) annotators called near — the knee of
  the recall curve. On the held-out annotator: recall 1.00.
- The support depth gate and contact fraction were chosen the same way, by
  their F1 on the fitting groups, then checked held-out (0.58 → 0.87).
- The ambiguity bands are recall/precision levers; each has a published sweep
  (ablations A2, A3, A7) so the chosen operating point is a visible decision.

## 5. Correction and flags

After the rules run, impossible combinations are cleaned: on+under on the
same pair demotes to a flag, front+behind cannot co-fire by construction, and
near is suppressed on support pairs. Whatever the rules could not decide is
recorded as a flag (`depth_ambiguous`, `lateral_ambiguous`,
`near_threshold_edge`) — the tool's uncertainty is a countable queue, about
8.5% of pairs of genuine review work for the entire dataset, and the review
is optional.

## 6. Video

Video is the same pipeline run per frame, plus two additions:

- **Tracking**: objects are matched frame-to-frame (same class + overlapping
  boxes) so "cup3" stays "cup3" for the whole clip.
- **Temporal smoothing**: each pair's relationship is decided by majority
  vote over a ±2-frame window, so a one-frame flicker is outvoted while a
  real change (a hand picking something up) survives.

Measured on two demo clips: where a pair of objects is stably tracked, its
relationship persists across 90–92% of frames before smoothing.

**"What do I pass as `--prompts`?"** The detector has to be told what object
types to look for. Three answers, by situation:

1. **Dataset-like footage** (the six classes): pass nothing — the defaults
   cover book/bottle/box/cube/person/remote.
2. **A new scene**: glance at the video and list what's in it —
   `--prompts "cup,laptop,plant"`. Ten seconds of human effort per video;
   the prompts name *object types*, not relationships or positions.
3. **Unknown footage**: use `--common-objects`, which searches a built-in
   list of ~60 everyday object types. Broader vocabulary costs some detection
   sharpness, so name the objects yourself when you can.

The relationship rules never change — prompts only decide which objects get
found. Anything detected is annotated with the same seven rules.

## 7. Honest limits

- Front/behind tops out where monocular depth genuinely cannot rank objects
  and the plane fallback's guard lacks evidence (objects on supports the
  detector has no box for). Measured: ~65% agreement pooled, ~84% once two
  annotator groups' inverted conventions are accounted for.
- In deployment mode, overall quality is capped by the detector: relations
  score the same with detected boxes as with perfect ones (0.85 conditional),
  but a missed object means every relationship involving it is missed too.
- The thresholds are calibrated to this dataset on purpose. A new domain
  keeps the rules and refits the constants on a small labelled slice — the
  procedure, not the numbers, is what transfers.
