# Two vision-language models on the same battery

Both runs: 30 images, the same numbered boxes, the same written predicate definitions, the same 374 human-judged pairs. Only the model differs. The pipeline column is one measurement, identical in both files by construction.

## Recall of the human triplets

| predicate | gold | gemini-flash-latest | gemini-3.1-pro-preview | pipeline |
|---|---|---|---|---|
| on | 57 | 0.667 | 0.737 | 0.860 |
| under | 47 | 0.660 | 0.766 | 0.809 |
| to the left of | 49 | 0.408 | 0.469 | 0.918 |
| to the right of | 49 | 0.327 | 0.388 | 0.939 |
| in front of | 80 | 0.188 | 0.237 | 0.650 |
| behind | 65 | 0.169 | 0.138 | 0.662 |
| near | 34 | 0.382 | 0.382 | 1.000 |
| **mean** | 381 | **0.400** | **0.445** | **0.834** |

## On the judged pairs, where precision is defined

| metric | gemini-flash-latest | gemini-3.1-pro-preview | pipeline |
|---|---|---|---|
| precision (micro) | 0.419 | 0.389 | 0.347 |
| recall (micro) | 0.378 | 0.423 | 0.806 |
| F1 (micro) | 0.397 | 0.405 | 0.485 |
| assertions | 344 | 414 | 885 |

## Self-consistency: one direction of a symmetric pair without the other

| predicate | gemini-flash-latest | gemini-3.1-pro-preview |
|---|---|---|
| to the left of | 131/374 (0.35) | 67/427 (0.16) |
| to the right of | 1/244 (0.00) | 0/360 (0.00) |
| in front of | 1/117 (0.01) | 3/251 (0.01) |
| behind | 1/117 (0.01) | 4/252 (0.02) |

Direct contradictions (a pair given a predicate and its opposite): gemini-flash-latest 0, gemini-3.1-pro-preview 0.

Malformed records dropped: gemini-flash-latest 0, gemini-3.1-pro-preview 0.
