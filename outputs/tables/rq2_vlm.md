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

## Beyond recall: precision, F1 and ranking quality

Precision against gold that annotates about a tenth of the ordered pairs is a floor, not an error rate: it charges an arm for every true relation the annotators never recorded, the artefact measured in SS4.3. Average precision is threshold-free and so removes the density confound outright. Macro weights the seven predicates equally; micro pools the counts.

| arm | macro R | macro P | macro F1 | macro AP | micro R | micro P | micro F1 |
|---|---|---|---|---|---|---|---|
| human-trained | 0.297 | 0.252 | 0.267 | 0.230 | 0.272 | 0.254 | 0.262 |
| pseudo-labelled | 0.365 | 0.243 | 0.289 | 0.215 | 0.348 | 0.225 | 0.273 |
| vlm-trained | 0.380 | 0.197 | 0.253 | 0.219 | 0.346 | 0.163 | 0.221 |
| auto-trained | 0.758 | 0.136 | 0.194 | 0.164 | 0.662 | 0.035 | 0.066 |

Per predicate, average precision (threshold-free):

| predicate | human-trained | pseudo-labelled | vlm-trained | auto-trained |
|---|---|---|---|---|
| on | 0.790 | 0.689 | 0.677 | 0.664 |
| under | 0.246 | 0.247 | 0.308 | 0.352 |
| to the left of | 0.156 | 0.144 | 0.230 | 0.040 |
| to the right of | 0.254 | 0.261 | 0.154 | 0.046 |
| in front of | 0.052 | 0.052 | 0.043 | 0.019 |
| behind | 0.085 | 0.095 | 0.114 | 0.021 |
| near | 0.029 | 0.019 | 0.007 | 0.009 |
| **mean** | **0.30** | **0.36** | **0.76** | |