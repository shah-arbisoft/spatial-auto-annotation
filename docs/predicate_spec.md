# Seven-Predicate Definition Document

This is the authoritative specification for how each of the seven spatial
predicates is **computed from geometry**. It is also, verifiably, the first
*operational* definition of these predicates for this dataset: the original
annotators' guidance (SGDET-Annotate repository, confirmed 2026-07) consists
of vocabulary lists with no definitions — see DATASET_NOTES. It is the reference that
[`src/predicates.py`](../src/predicates.py) implements, and a graded part of the
dissertation (problem analysis & design). Every rule is a small, readable
function with explicit, justified thresholds.

The seven predicates are exactly those of the source dataset (Wang et al.,
*A Spatial Relationship Aware Dataset for Robotics*, ACM MM 2025):

> `behind`, `in front of`, `on`, `under`, `left of`, `right of`, `near`.

> **Naming/IDs (verified against the real data — see
> [DATASET_NOTES.md](DATASET_NOTES.md)).** The dataset's exact predicate strings
> are **`to the left of`** and **`to the right of`** (not "left of"/"right of").
> The canonical names and their dataset IDs are fixed in
> `src/predicates.PREDICATE_IDS`: in front of=6, behind=2, on=10, under=17,
> to the left of=15, to the right of=16, near=9. The shorthand "left of/right of"
> is used below only for readability.

---

## 0. Notation and the per-object measurements

For each detected object we have, after detection → segmentation → depth:

| Symbol | Meaning | Source |
|---|---|---|
| `box = (x1, y1, x2, y2)` | axis-aligned bounding box, image pixels | detector |
| `M` | binary segmentation mask | SAM2 |
| `(cx, cy)` | mask centroid in image coords (pixels) | `geometry.py` |
| `d` | per-object relative depth (smaller = nearer the camera) | Depth Anything v2 sampled over `M` |
| `P = (X, Y, Z)` | lifted 3D position | `geometry.py` |

**Image coordinate convention.** `x` increases to the **right**, `y` increases
**downward** (standard image convention). "Left/right" are expressed in the
**camera frame** — i.e. as the camera sees the scene — matching how the human
annotators clicked subject→object on screen. This is stated explicitly because it
is a design choice that must be justified against the dataset's annotation tool.

**Depth convention.** Depth Anything v2 returns a *relative* (ordinal) depth map.
We treat **smaller `d` as nearer the camera**. Because depth is relative and
per-image, all depth comparisons are *ordinal between objects in the same image*,
never absolute across images. This is a deliberate limitation discussed in the
critical chapter.

**Normalisation.** Distances and gaps are normalised by image dimensions (x by
width, y by height) so a single threshold transfers across image sizes. The
lifted 3D position `P = (X, Y, Z)` (normalised image coordinates plus scaled
depth) is retained per object for the downstream classifier's features; the
`near` rule itself uses a size-relative 2D box gap (see §7).

All predicates are defined for an **ordered pair (A, B)** — A is the subject,
B is the object — to match the dataset's subject→object triplet format.

---

## 1. `on(A, B)` — A rests on B

A is `on` B when A sits directly above B, they (nearly) touch, and they overlap
horizontally. Four conditions, all required:

1. **Above:** A's mask bottom is above B's mask top region — A is the higher
   object. Concretely `cy_A < cy_B` and the bottom edge of A is near the top of B.
2. **Touching (small vertical gap):** the normalised vertical gap between the
   bottom of A and the top of B is `<= on_vertical_gap` (default 0.05). A gap that
   is small but non-negative captures resting contact while tolerating mask noise.
3. **Horizontal overlap:** the horizontal extents of A and B overlap by at least
   `on_horizontal_overlap` (default 0.20) of the **narrower** box's x-extent
   (`_x_extent_overlap`; this is a containment fraction, not an IoU, so a small
   object fully above a large one scores 1.0). An object resting on another
   must share a footprint.
4. **Depth co-location:** `|depth_A − depth_B| ≤ on_depth_eps` (calibrated 0.06
   on the train annotator groups). Rationale: on a floor plane, "farther"
   projects as "higher in the image", so an object *behind* another mimics the
   2D signature of one stacked *on* it; truly stacked objects share a camera
   distance. Measured effect: held-out support F1 0.58 → 0.71, ~half of the
   audited false support labels removed at ≤2 points of recall.

Justification: contact + support is the everyday meaning of "on"; encoding it as
*above + touching + horizontal overlap* avoids labelling a cup floating in front
of a shelf as "on" it.

**Mask-contact evidence (primary when masks are available).** The box
conditions above are the no-mask fallback. With SAM2 masks the rule instead
uses the physical support signature: `contact_below(A, B)` = the fraction of
A's mask-bottom columns with B's mask within 5 px below (`src/contact.py`);
`on` requires `contact ≥ on_contact_min` (calibrated 0.60 on train groups;
flat optimum 0.60–0.80) plus the depth gate and centroid order. This captures
the containment case the box test misses (nested boxes at shallow view angles
— formerly 79–88% of support misses) and rejects cluster neighbours whose
boxes touch but masks don't. Measured: held-out support F1 0.71 → 0.87;
re-audited extra-label precision 0.07/0.20 → 0.73/0.80 (on/under). Known
residual failure mode: an object *held* by a person satisfies contact
(holding ≠ resting); a class-aware guard is a documented refinement.

**Class guard.** Classes listed in `no_support_classes` (shipped value:
`["human"]`) are excluded from `on`/`under` in either role. The justification
is measured rather than assumed: the annotators never recorded a person as
supporting or being supported, on 0 of 2,466 gold support triplets, and mask
contact alone cannot distinguish a person *holding* an object from a surface
*supporting* one. The guard is a config entry, not a hard-coded name, so a
deployment with different classes revises it in one line.

## 2. `under(A, B)` — A is below and supports B

`under` is the strict inverse of `on`:

```
under(A, B)  ==  on(B, A)
```

It is computed by evaluating the `on` conditions with the arguments swapped. This
guarantees consistency: the pair can never be both `on(A,B)` and `under(A,B)`
(see the correction step, §8).

## 3. `left of(A, B)` — A's centre is left of B's

```
left_of(A, B)  ==  (cx_B - cx_A) > lateral_center_eps
                   in camera-frame image coordinates, cx normalised by width
```

with the magnitude `|cx_A - cx_B|` (normalised by width) used for confidence: if
it is below `lateral_center_eps` (default 0.02) the centres nearly coincide and
the case is **flagged ambiguous** rather than silently labelled.

## 4. `right of(A, B)` — A's centre is right of B's

```
right_of(A, B)  ==  (cx_A - cx_B) > lateral_center_eps
```

The strict mirror of `left of`. Exactly one of `left of` / `right of` holds away
from the ambiguity band; inside the band, neither is emitted and the pair is
flagged. `right_of(A,B) == left_of(B,A)`.

## 5. `in front of(A, B)` — A is nearer the camera than B

Two-stage cascade. Stage 1 — depth ordering:

```
in_front_of(A, B)  ==  (d_B - d_A) > depth_eps    (smaller depth = nearer)
```

The band is inside the rule: with `depth_eps` at 0.03 a pair whose depths
differ by less than that is not ordered here at all ⇒ stage 2.

Stage 2 — **ground-plane fallback** (only where stage 1 abstained): two
objects standing on the same floor are depth-ordered by projection — the
nearer object's box bottom sits lower in the image.

```
plane(A, B)  ==  (y2_A - y2_B) > plane_band     ⇒  in front of
             ==  (y2_A - y2_B) < -plane_band    ⇒  behind
```

with `plane_band` = 0.005 (normalised image height; calibrated on train
groups, ablation A7). Guard: the fallback fires **only when neither object is
elevated** — no mask-contact ≥ `on_contact_min` with any partner object (an
elevated object's box bottom locates its support, not itself) — and only when
mask evidence exists at all (box-only mode: fallback off). Pairs both stages
abstain on are flagged ambiguous, not labelled.

Worked example (encoded in `tests/test_predicates.py`): bottle with box
bottom 0.80 vs book with box bottom 0.60, depths 0.50/0.51 (inside
`depth_eps`), both floor-standing ⇒ bottle `in front of` book. Same pair with
either object elevated ⇒ flagged.

## 6. `behind(A, B)` — A is farther from the camera than B

```
behind(A, B)  ==  (d_A - d_B) > depth_eps
```

The inverse of `in front of`: `behind(A,B) == in_front_of(B,A)`. The same
cascade applies — `depth_eps` band, then the guarded ground-plane fallback,
then the flag.

## 7. `near(A, B)` — A and B are close, relative to their size

```
near(A, B)  ==  box_gap_rel(A, B) <= near_T   AND   not (on(A,B) or under(A,B))
```

where `box_gap_rel` is the **edge-to-edge gap between the two (normalised)
boxes, divided by the mean object size** (sqrt of box area): "near" scales with
the objects — a small gap between two books reads as near; the same absolute gap
between a person and a cube may not. The gap is 0 when boxes touch or overlap.

**Intended semantics (supervisor, by email):** `near` meant **"next to"** —
the annotation team merged an earlier separate "next to" label into it. An
adjacency reading supports the gap metric and the contact-boundary behaviour
below.

**Contact exclusion.** Measured on the human labels, `near` co-occurs with
`on`/`under` on **0 of 469** pairs (74% of near pairs carry *only* near): the
annotators used `near` as "close but no contact relation". The rule encodes
this: a pair already labelled on/under is not additionally near.

**Why not 3D centroid distance.** Monocular depth is normalised per image, so
centroid distances are not comparable across scenes: in a metric bake-off, every
3D-centroid variant transferred to held-out annotators at F1 ≤ 0.024, while the
size-relative gap metric transfers with recall 1.0 (see DATASET_NOTES).

**Fitting protocol (annotator-aware).** Only 3 of the 9 annotator groups ever
used `near` (group_0: 244, group_4: 129, group_8: 93 *unordered pairs*, 469
with group_2's 3; the rest 0–3). The fit works on pairs, so those are the
counts that matter here. Section 3.2 of the dissertation quotes the same
annotation as 461 / 160 / 93 plus 3, summing to 717, because it counts
*ordered triplets*, which is the unit the recall figures use: most `near`
pairs carry both directions, group_8's carry one each.
`near_T` is therefore fitted on the near-using groups inside the training split
(groups 0 and 4), on human-annotated non-contact pairs, and evaluated on the
held-out near-using annotator (group_8). Fitted **T = 1.372**; held-out
**recall = 1.000** (every pair that annotator called near is within T), with
precision differences across annotators (0.16–0.63) reflecting how exhaustively
each applied the label, not geometric disagreement. The deterministic threshold
applies one definition uniformly — precisely the "spatial thresholds for near"
the source paper's future work calls for.

`near` is symmetric: `near(A,B) == near(B,A)`. Cases within `flag_near_band` of
`near_T` (gap units) are **flagged** for optional human review.

---

## 8. Correction step — consistency by construction, plus active corrections

Geometric consistency is enforced at two levels, and it is worth being precise
about which is which:

**By construction.** With the shipped rules the three mutually exclusive
families can never co-occur on an ordered pair: `on(A,B)` and `under(A,B)`
require strictly opposite centroid orderings; `left/right` and `front/behind`
use strict comparisons separated by an ambiguity band. Inverses mirror exactly
across the two orderings (`on(A,B) ⇔ under(B,A)`, etc.). These invariants are
pinned by a randomised test over thousands of scenes
(`tests/test_invariants.py`); the runtime conflict check therefore acts as an
assertion/safety net for future rule variants rather than a filter that fires
in practice.

**Active corrections** (the Open3D-VQA-style error-correction idea, adapted):

- `near` is **suppressed on contact pairs** — measured, near co-occurs with
  on/under on 0 of 469 human pairs (§7).
- Ambiguous cases are **abstained and flagged** rather than guessed (§9): the
  rule emits nothing when the geometric evidence is inside an ambiguity band.

## 9. Confidence flags — what gets marked, and the honest cost

A pair is flagged (not dropped) when any of:

- `near` gap within `flag_near_band` of `near_T` (§7).
- `left/right` centres within `lateral_center_eps` (§3–4).
- `front/behind` depths within `depth_eps` AND the ground-plane fallback
  unable to decide (elevated object, no mask evidence, or bottom edges within
  `plane_band`) (§5–6).

Flags are written alongside the triplets. Measured on the full dataset with the
shipped rules, 31.5% of ordered pairs carry some flag: `depth_ambiguous` 19.3%,
`lateral_ambiguous` 10.0% and `near_threshold_edge` 8.5% (a pair may carry more
than one). The depth share was 29.5% before the ground-plane fallback shipped,
which resolved about a third of the depth abstentions. The flag types serve
different purposes and are reported separately: depth and lateral flags mark
*abstentions* (no wrong label was emitted; nothing to fix, and humans typically
do not label those pairs either), while `near_threshold_edge` is the genuine
*review queue* for the borderline-near cases. The evaluation reports per-type
flag rates rather than a single number, and the human-in-the-loop claim is
costed on the review-queue flags, not the abstentions.

---

## Summary table

| Predicate | Core test (ordered pair A,B) | Threshold(s) and shipped values | Symmetry |
|---|---|---|---|
| `on` | mask contact below + depth co-location + centroid order (box test is the no-mask fallback) | `on_contact_min` 0.60, `on_depth_eps` 0.06, `on_vertical_gap` 0.05, `on_horizontal_overlap` 0.20 | `on(A,B)=under(B,A)` |
| `under` | inverse of `on` | (same as `on`) | `under(A,B)=on(B,A)` |
| `left of` | `(cx_B - cx_A) > eps` | `lateral_center_eps` 0.02 | `left(A,B)=right(B,A)` |
| `right of` | `(cx_A - cx_B) > eps` | `lateral_center_eps` 0.02 | `right(A,B)=left(B,A)` |
| `in front of` | `(d_B - d_A) > eps`, then guarded ground-plane fallback | `depth_eps` 0.03, `plane_band` 0.005 | `front(A,B)=behind(B,A)` |
| `behind` | `(d_A - d_B) > eps`, then guarded ground-plane fallback | `depth_eps` 0.03, `plane_band` 0.005 | `behind(A,B)=front(B,A)` |
| `near` | `box_gap_rel <= near_T`, no contact | `near_T` 1.372 (fitted), `flag_near_band` 0.15 | symmetric |

All thresholds are declared in `configs/default.yaml`; `near_T` is fitted in
`eval/fit_near.py` (annotator-aware protocol above) and frozen there.
