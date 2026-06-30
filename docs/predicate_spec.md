# Seven-Predicate Definition Document

This is the authoritative specification for how each of the seven spatial
predicates is **computed from geometry**. It is the reference that
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
width, y by height) so a single threshold transfers across image sizes. The 3D
position used for `near` is `(X, Y, Z)` with `X, Y` normalised image coordinates
and `Z` a depth scaled to comparable units (see §7).

All predicates are defined for an **ordered pair (A, B)** — A is the subject,
B is the object — to match the dataset's subject→object triplet format.

---

## 1. `on(A, B)` — A rests on B

A is `on` B when A sits directly above B, they (nearly) touch, and they overlap
horizontally. Three conditions, all required:

1. **Above:** A's mask bottom is above B's mask top region — A is the higher
   object. Concretely `cy_A < cy_B` and the bottom edge of A is near the top of B.
2. **Touching (small vertical gap):** the normalised vertical gap between the
   bottom of A and the top of B is `<= on_vertical_gap` (default 0.05). A gap that
   is small but non-negative captures resting contact while tolerating mask noise.
3. **Horizontal overlap:** the horizontal extents of A and B overlap by at least
   `on_horizontal_overlap` (default 0.20), measured as overlap-of-x-extent. An
   object resting on another must share a footprint.

Justification: contact + support is the everyday meaning of "on"; encoding it as
*above + touching + horizontal overlap* avoids labelling a cup floating in front
of a shelf as "on" it.

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
left_of(A, B)  ==  (cx_A < cx_B)   in camera-frame image coordinates
```

with the magnitude `|cx_A - cx_B|` (normalised by width) used for confidence: if
it is below `lateral_center_eps` (default 0.02) the centres nearly coincide and
the case is **flagged ambiguous** rather than silently labelled.

## 4. `right of(A, B)` — A's centre is right of B's

```
right_of(A, B)  ==  (cx_A > cx_B)
```

The strict mirror of `left of`. Exactly one of `left of` / `right of` holds away
from the ambiguity band; inside the band, neither is emitted and the pair is
flagged. `right_of(A,B) == left_of(B,A)`.

## 5. `in front of(A, B)` — A is nearer the camera than B

```
in_front_of(A, B)  ==  (d_A < d_B)    (smaller depth = nearer)
```

`|d_A - d_B|` below `depth_eps` (default 0.03) ⇒ depths nearly equal ⇒ flagged
ambiguous, not labelled.

## 6. `behind(A, B)` — A is farther from the camera than B

```
behind(A, B)  ==  (d_A > d_B)
```

The inverse of `in front of`: `behind(A,B) == in_front_of(B,A)`. The same
`depth_eps` ambiguity band applies.

## 7. `near(A, B)` — A and B are close in 3D

```
near(A, B)  ==  dist3D(P_A, P_B) <= near_T
```

where `dist3D` is Euclidean distance between the two lifted 3D centroids
`P = (X, Y, Z)`, and `near_T` is **fitted to the human labels**, not guessed:

- Sweep candidate `T` over a range; for each, compute agreement (F1) of the
  resulting `near` labels against the human `near` annotations on the 900 images.
- Choose the `T` that maximises agreement; **freeze and report** it in
  `configs/default.yaml` and the dissertation.

This directly addresses the paper's flagged problem that `near` was inconsistent
between annotators and that future work should use a "spatial threshold for near."
A single fitted threshold is, by construction, more self-consistent than nine
people. `near` is symmetric: `near(A,B) == near(B,A)`.

Cases where `dist3D` falls within `flag_near_band` of `near_T` are **flagged**
for optional human review (human-in-the-loop accelerator, not brittle full-auto).

---

## 8. Correction step — reject geometrically impossible labels

After computing all predicates for a pair, enforce mutual exclusivity:

- Not both `on(A,B)` and `under(A,B)`.
- Not both `left of(A,B)` and `right of(A,B)`.
- Not both `in front of(A,B)` and `behind(A,B)`.
- `on`/`under` implies vertical adjacency; if depth simultaneously says the
  objects are far in `Z` while geometry says `on`, prefer the contact evidence
  and flag the conflict.

If a contradiction survives the per-rule thresholds (e.g. due to mask/depth
noise), the pair is **demoted to a flag**, never emitted as a contradictory
triplet. This is the Open3D-VQA-style error-correction idea applied to our rules.

## 9. Confidence flags — what gets marked for review

A pair is flagged (not dropped) when any of:

- `near` distance within `flag_near_band` of `near_T` (§7).
- `left/right` centres within `lateral_center_eps` (§3–4).
- `front/behind` depths within `depth_eps` (§5–6).
- the correction step found and resolved a contradiction (§8).

Flags are written alongside the triplets so a human can review only the
genuinely ambiguous minority.

---

## Summary table

| Predicate | Core test (ordered pair A,B) | Threshold(s) | Symmetry |
|---|---|---|---|
| `on` | above + touching + horiz. overlap | `on_vertical_gap`, `on_horizontal_overlap` | `on(A,B)=under(B,A)` |
| `under` | inverse of `on` | (same as `on`) | `under(A,B)=on(B,A)` |
| `left of` | `cx_A < cx_B` | `lateral_center_eps` | `left(A,B)=right(B,A)` |
| `right of` | `cx_A > cx_B` | `lateral_center_eps` | `right(A,B)=left(B,A)` |
| `in front of` | `d_A < d_B` | `depth_eps` | `front(A,B)=behind(B,A)` |
| `behind` | `d_A > d_B` | `depth_eps` | `behind(A,B)=front(B,A)` |
| `near` | `dist3D <= near_T` | `near_T` (fitted) | symmetric |

All thresholds are declared in `configs/default.yaml`; `near_T` is fitted in
`eval/fidelity.py` and frozen there.
