# RQ1 fidelity tables

## T1 — Per-predicate recall of human triplets

| Predicate | Gold | Ours | Ours (held-out) | Random | Majority (in front of) | Box-only |
|---|---|---|---|---|---|---|
| on | 1465 | 0.88 | 0.92 | 0.13 | 0.00 | 0.84 |
| under | 1001 | 0.81 | 0.92 | 0.16 | 0.00 | 0.77 |
| to the left of | 972 | 0.97 | 0.95 | 0.13 | 0.00 | 0.97 |
| to the right of | 1174 | 0.98 | 0.99 | 0.13 | 0.00 | 0.99 |
| in front of | 2013 | 0.64 | 0.20 | 0.13 | 1.00 | 0.00 |
| behind | 1584 | 0.66 | 0.35 | 0.16 | 0.00 | 0.00 |
| near | 717 | 1.00 | 1.00 | 0.16 | 0.00 | 0.87 |
| **mean** |  | **0.85** | **0.76** | 0.14 | 0.14 | 0.63 |

## T2 — P/R/F1 restricted to human-annotated pairs (n=8790)

| Predicate | P | R | F1 | support |
|---|---|---|---|---|
| on | 0.88 | 0.88 | 0.88 | 1465 |
| under | 0.84 | 0.81 | 0.83 | 1001 |
| to the left of | 0.35 | 0.96 | 0.51 | 972 |
| to the right of | 0.42 | 0.98 | 0.59 | 1174 |
| in front of | 0.43 | 0.64 | 0.51 | 2013 |
| behind | 0.36 | 0.66 | 0.46 | 1584 |
| near | 0.12 | 1.00 | 0.21 | 717 |

## T3 — Per-annotator-group recall (tenth-annotator view)

| Group | Gold triplets | Recall |
|---|---|---|
| group_0 | 1949 | 0.87 |
| group_1 | 1489 | 0.93 |
| group_2 | 1192 | 0.86 |
| group_3 | 435 | 0.72 |
| group_4 | 356 | 0.87 |
| group_5 | 687 | 0.93 |
| group_6 | 970 | 0.58 |
| group_7 | 796 | 0.91 |
| group_8 | 1052 | 0.57 |

## T5 — Front/behind decomposition per annotator group

| Group | Gold | Emit rate | Agreement when committed | Convention | Raw recall | Aligned recall |
|---|---|---|---|---|---|---|
| group_0 | 724 | 0.92 | 0.95 | same | 0.88 | 0.88 |
| group_1 | 639 | 0.98 | 1.00 | same | 0.98 | 0.98 |
| group_2 | 351 | 0.58 | 1.00 | same | 0.58 | 0.58 |
| group_3 | 258 | 0.54 | 0.98 | same | 0.53 | 0.53 |
| group_4 | 65 | 1.00 | 0.57 | same | 0.57 | 0.57 |
| group_5 | 371 | 1.00 | 0.99 | same | 0.98 | 0.98 |
| group_6 | 415 | 0.98 | 0.05 | inverted | 0.05 | 0.93 |
| group_7 | 330 | 0.90 | 1.00 | same | 0.90 | 0.90 |
| group_8 | 444 | 0.73 | 0.02 | inverted | 0.01 | 0.71 |
| **overall** | 3597 |  |  |  | **0.65** | **0.84** |

## T4 — Flag rates

pairs flagged: 26716 (31.5%)

| Flag | Count | Rate |
|---|---|---|
| depth_ambiguous | 16412 | 19.3% |
| lateral_ambiguous | 8454 | 10.0% |
| near_threshold_edge | 7200 | 8.5% |