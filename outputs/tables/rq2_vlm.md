# RQ2 - downstream classifier: human vs automatic vs self-trained labels

Identical features, model, oversampling and split; averaged over seeds [42, 43, 44]; evaluated against held-out human gold (groups 6-8). Only the label source differs. The pseudo-labelled arm self-trains on the human labels (teacher, confident pseudo-labels, student).

| predicate | human-trained | pseudo-labelled | auto-trained | gold (held-out) |
|---|---|---|---|---|
| on | 0.84 (0.83-0.86) | 0.90 (0.89-0.92) | 0.92 (0.92-0.93) | 348 |
| under | 0.44 (0.36-0.53) | 0.59 (0.59-0.60) | 0.92 (0.92-0.93) | 192 |
| to the left of | 0.22 (0.18-0.29) | 0.31 (0.27-0.34) | 0.95 (0.95-0.95) | 446 |
| to the right of | 0.25 (0.22-0.31) | 0.38 (0.37-0.39) | 0.99 (0.99-0.99) | 550 |
| in front of | 0.09 (0.08-0.10) | 0.12 (0.12-0.13) | 0.19 (0.19-0.19) | 609 |
| behind | 0.15 (0.12-0.17) | 0.22 (0.17-0.28) | 0.33 (0.32-0.34) | 580 |
| near | 0.08 (0.00-0.19) | 0.03 (0.00-0.06) | 1.00 (1.00-1.00) | 93 |
| **mean** | **0.30** | **0.36** | **0.76** | |