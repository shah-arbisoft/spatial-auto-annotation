# RQ2 - downstream classifier: human vs automatic vs self-trained labels

Identical features, model, oversampling and split; averaged over seeds [42, 43, 44]; evaluated against held-out human gold (groups 6-8). Only the label source differs. The pseudo-labelled arm self-trains on the human labels (teacher, confident pseudo-labels, student).

| predicate | human-trained | pseudo-labelled | auto-trained | gold (held-out) |
|---|---|---|---|---|
| on | 0.90 (0.89-0.91) | 0.91 (0.91-0.92) | 0.92 (0.92-0.92) | 348 |
| under | 0.54 (0.50-0.57) | 0.84 (0.74-0.94) | 0.93 (0.92-0.93) | 192 |
| to the left of | 0.16 (0.12-0.21) | 0.22 (0.20-0.26) | 0.95 (0.95-0.95) | 446 |
| to the right of | 0.29 (0.28-0.29) | 0.35 (0.33-0.40) | 0.99 (0.99-0.99) | 550 |
| in front of | 0.09 (0.07-0.11) | 0.13 (0.10-0.15) | 0.19 (0.18-0.19) | 609 |
| behind | 0.18 (0.14-0.21) | 0.25 (0.23-0.27) | 0.33 (0.32-0.33) | 580 |
| near | 0.04 (0.00-0.10) | 0.03 (0.01-0.06) | 1.00 (1.00-1.00) | 93 |
| **mean** | **0.31** | **0.39** | **0.76** | |