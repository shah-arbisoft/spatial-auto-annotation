# SGDet (detector-in-the-loop) vs PredCls — 836 images

## Detection (GroundingDINO zero-shot, IoU 0.5, class-matched)

| class | recall | precision |
|---|---|---|
| book | 0.66 | 0.61 |
| bottle | 0.94 | 0.65 |
| box | 0.58 | 0.26 |
| cube | 0.40 | 0.19 |
| human | 0.95 | 0.63 |
| remote | 0.52 | 0.17 |

## Triplet recall of human labels (SGDet vs the PredCls headline)

| predicate | SGDet recall | given both endpoints detected | PredCls recall |
|---|---|---|---|
| on | 0.31 | 0.83 (544) | 0.88 |
| under | 0.33 | 0.77 (428) | 0.81 |
| to the left of | 0.48 | 0.96 (487) | 0.97 |
| to the right of | 0.33 | 0.98 (400) | 0.98 |
| in front of | 0.27 | 0.69 (797) | 0.52 |
| behind | 0.28 | 0.70 (636) | 0.55 |
| near | 0.63 | 1.00 (452) | 1.00 |
| **mean** | **0.38** | **0.85** | **0.81** |

The conditional column isolates the relation layer under detected (noisier) boxes: where both endpoints are found, the rules perform close to their PredCls levels — the SGDet gap is detection, not relations.