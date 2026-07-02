# Dataset Notes — SpatialAwareRobotDataset (verified from the real files)

These facts were measured directly from the supervisor-supplied dataset
(`SpatialAwareRobotDataset-main/`), not assumed from the brief. They settle the
"things to verify" in the handoff (§15) and feed the dissertation's data section.

> Source: Wang, Pham, Guo, Zhou, *A Spatial Relationship Aware Dataset for
> Robotics*, ACM MM 2025. Annotation tool: SGDET-Annotate. Licence: CC-BY 4.0.

## Counts (measured)

| Quantity | Value |
|---|---|
| Image groups (annotator batches) | **9** (`group_0` … `group_8`) — matches "~nine annotators" |
| Images on disk | **884** |
| Annotated images (non-empty) | **838** (839 annotation JSONs; one frame has no objects) |
| `group_8` is partial | 84 images, 37 annotations |
| Object instances | 8,030 (mean **9.6 / image**, range 0–17) |
| Relationship triplets (all predicates) | 9,313 (mean **11.1 / image**, range 0–48) |
| Relationship triplets (the 7 targets) | **8,928 (95.9%)** |

So the working ground-truth size for RQ1/RQ2 is **838 images / 8,928 target
triplets**, not "~900 labelled images" as loosely stated in the brief.

## Object vocabulary (`labels.json`, identical across all groups)

`book(1), bottle(2), box(3), cube(4), human(5), remote(6)` — 6 classes.
Two object instances carry an out-of-vocabulary `id=7` (no name in any map);
treated as annotation noise.

| class | instances |
|---|---|
| book | 2,273 |
| cube | 2,195 |
| box | 1,209 |
| bottle | 1,193 |
| human | 593 |
| remote | 565 |

## Predicate vocabulary (`relationships.json`, identical across all groups)

The tool ships the 19-predicate Visual Genome relationship list, but annotators
only ever used **14**, and the **seven spatial targets dominate (95.9%)**.

**Target predicates** (what we compute) and their dataset IDs:

| predicate | ID | human triplets |
|---|---|---|
| in front of | 6 | 2,013 |
| behind | 2 | 1,586 |
| on | 10 | 1,465 |
| to the right of | 16 | 1,174 |
| under | 17 | 1,001 |
| to the left of | 15 | 972 |
| near | 9 | 717 |

Note: the dataset uses **"to the left of" / "to the right of"** (not "left of"/
"right of"). `near` is the rarest target → class weighting matters for RQ2.

**Non-target tail (4.1%, out of scope for the 7-predicate study):**
`above`(151), `using`(68), `far away from`(67), `mounted on`(59),
`looking at`(31), `over`(6), `holding`(3). Of these, `above` and `far away from`
are themselves geometric and are noted as trivial future extensions; the rest
are semantic/contact relations outside the spatial-from-geometry scope.

## File schema (per image, e.g. `group_0/000000.json`)

| key | meaning |
|---|---|
| `image-name` | annotator's local path to the source image |
| `width`, `height` | original image size (e.g. 640×480) |
| `boxes_canvas` | `[x1,y1,x2,y2]` in the annotation canvas (oversized; **not** image px) |
| `boxes_1024`, `boxes_512` | `[x_center, y_center, w, h]` in a 1024 / 512 longest-side resize |
| `attribute` | N×10 attribute IDs (0 = none) — we do not predict attributes |
| `labels` | object label IDs (index-aligned with the box arrays) |
| `relationships` | `[[subj_idx, obj_idx], …]` indices into the object arrays |
| `predicates` | predicate ID per relationship, 1-to-1 with `relationships` |

`.h5` mirrors this: `image-name/width/height` as attributes; the arrays as
int32 datasets.

### Box coordinate handling (verified)

`boxes_1024 × 0.625 == boxes_512 × 1.25 == original 640×480 pixels`. To get
normalised `(x1,y1,x2,y2)`, divide a `boxes_1024` entry by the resized dims
(`scale = 1024/max(W,H)`). Round-trip through `src/dataset.py` →
`src/writers.py` reproduces the original `boxes_1024` with **0-pixel error**.
`boxes_canvas` is in a larger on-screen canvas space (values exceed image dims)
and is **not** used. ~1.8% of boxes overflow [0,1] by ≤0.0007 (sub-pixel resize
rounding); clamped in the loader.

## The `near` predicate — measured inter-annotator inconsistency

The paper reports "inconsistencies, particularly with the 'near' predicate".
Measured on the released labels, this is quantifiable and severe:

- **Only 3 of the 9 annotator groups ever used `near`**: group_0 (244 labels),
  group_4 (129), group_8 (93); group_2 has 3; the other five groups have 0.
- `near` **never co-occurs with `on`/`under`** (0 of 469 pairs), and 74% of near
  pairs carry *only* near — annotators treated it as "close but no contact
  relation".
- At a single size-relative gap threshold (T = 1.372, fitted on groups 0+4),
  **recall is 1.000 for all three near-using annotators** — everything anyone
  called near lies within one common threshold — but precision per annotator is
  0.41 / 0.63 / 0.16: each labelled only a fraction of the equally-close pairs.
  The human labels are directionally consistent but **non-exhaustive**, and the
  degree of exhaustiveness varies ~4× between annotators.
- Consequence for metric choice: per-image relative depth makes 3D-centroid
  distances incomparable across scenes — every centroid-based variant transfers
  to held-out annotators at F1 ≤ 0.024, so the rule uses the size-relative 2D
  box gap instead.

Implication: `near` cannot be validated as simple agreement with "the human
consensus" — for this predicate there isn't one. The fitted threshold is instead
presented as the **consistent operational definition** the paper's future work
asks for ("spatial thresholds for near"), with per-annotator agreement reported
against it, and the manual audit (RQ1) checking whether threshold-only pairs the
humans skipped are genuinely near.

## Implications for the project

1. **Ground truth scale:** 838 images / 8,928 target triplets.
2. **Naming:** canonical predicate strings are the dataset's exact ones; IDs are
   fixed in `src/predicates.PREDICATE_IDS`.
3. **`group_4` has no mapping files** — it safely reuses the shared vocabulary.
4. **Output compatibility:** `src/writers.py` emits this exact schema (validated).
5. **RQ1 scope:** evaluate the 7 targets; report the 4.1% tail as out-of-scope.
6. **RQ2 imbalance:** weight by inverse predicate frequency (near/left are rare).
