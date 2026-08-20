# RQ1 fidelity tables

## T1 — Per-predicate recall of human triplets

| Predicate | Gold | Ours | Ours (held-out) | Random | Majority (in front of) | Box-only |
|---|---|---|---|---|---|---|
| on | 1465 | 0.81 | 0.85 | 0.13 | 0.00 | 0.84 |
| under | 1001 | 0.75 | 0.82 | 0.16 | 0.00 | 0.77 |
| to the left of | 972 | 0.97 | 0.95 | 0.13 | 0.00 | 0.97 |
| to the right of | 1174 | 0.98 | 0.99 | 0.13 | 0.00 | 0.99 |
| in front of | 2013 | 0.70 | 0.20 | 0.13 | 1.00 | 0.00 |
| behind | 1584 | 0.71 | 0.37 | 0.16 | 0.00 | 0.00 |
| near | 717 | 1.00 | 1.00 | 0.16 | 0.00 | 0.87 |
| **mean** |  | **0.84** | **0.74** | 0.14 | 0.14 | 0.63 |

## T2 — P/R/F1 restricted to human-annotated pairs (n=8790)

| Predicate | P | R | F1 | support |
|---|---|---|---|---|
| on | 0.95 | 0.81 | 0.88 | 1465 |
| under | 0.92 | 0.75 | 0.83 | 1001 |
| to the left of | 0.35 | 0.96 | 0.51 | 972 |
| to the right of | 0.42 | 0.98 | 0.59 | 1174 |
| in front of | 0.43 | 0.70 | 0.53 | 2013 |
| behind | 0.35 | 0.71 | 0.47 | 1584 |
| near | 0.11 | 1.00 | 0.20 | 717 |

## T3 — Per-annotator-group recall (tenth-annotator view)

| Group | Gold triplets | Recall |
|---|---|---|
| group_0 | 1949 | 0.85 |
| group_1 | 1489 | 0.90 |
| group_2 | 1192 | 0.92 |
| group_3 | 435 | 0.88 |
| group_4 | 356 | 0.86 |
| group_5 | 687 | 0.93 |
| group_6 | 970 | 0.56 |
| group_7 | 796 | 0.90 |
| group_8 | 1052 | 0.57 |

## T5 — Front/behind decomposition per annotator group

| Group | Gold | Emit rate | Agreement when committed | Convention | Raw recall | Aligned recall |
|---|---|---|---|---|---|---|
| group_0 | 724 | 0.94 | 0.95 | same | 0.89 | 0.89 |
| group_1 | 639 | 1.00 | 1.00 | same | 1.00 | 1.00 |
| group_2 | 351 | 0.85 | 1.00 | same | 0.85 | 0.85 |
| group_3 | 258 | 0.82 | 0.99 | same | 0.81 | 0.81 |
| group_4 | 65 | 1.00 | 0.57 | same | 0.57 | 0.57 |
| group_5 | 371 | 1.00 | 0.99 | same | 0.98 | 0.98 |
| group_6 | 415 | 0.99 | 0.05 | inverted | 0.05 | 0.94 |
| group_7 | 330 | 0.94 | 1.00 | same | 0.94 | 0.94 |
| group_8 | 444 | 0.84 | 0.02 | inverted | 0.02 | 0.82 |
| **overall** | 3597 |  |  |  | **0.70** | **0.91** |

## T4 — Flag rates

pairs flagged: 24712 (29.1%)

| Flag | Count | Rate |
|---|---|---|
| depth_ambiguous | 13258 | 15.6% |
| lateral_ambiguous | 8454 | 10.0% |
| near_threshold_edge | 7200 | 8.5% |