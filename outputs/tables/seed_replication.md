# Benchmark replication across seeds

Seeds [42, 43, 44]. Identical data, split and frozen detector in every run, so the spread is the relation model's own training variance. Source: re-evaluation (fixed zero-shot reference, per-group slices); vlm from raw folders, which the re-evaluation predates.

## full

| metric | human-trained | auto-trained | vlm-trained | ranges overlap |
|---|---|---|---|---|
| R@100 | 0.328 (0.313-0.348, n=3) | 0.254 (0.250-0.260, n=3) | 0.310 (0.296-0.327, n=3) | no |
| mR@100 | 0.326 (0.304-0.346, n=3) | 0.278 (0.268-0.288, n=3) | 0.329 (0.314-0.348, n=3) | no |
| F1@100 | 0.327 (0.309-0.347, n=3) | 0.265 (0.259-0.268, n=3) | 0.319 (0.305-0.337, n=3) | no |
| zR@100 | 0.003 (0.000-0.004, n=3) | 0.172 (0.157-0.196, n=3) | - | no |

## group_6

| metric | human-trained | auto-trained | vlm-trained | ranges overlap |
|---|---|---|---|---|
| R@100 | 0.395 (0.376-0.412, n=3) | 0.281 (0.271-0.290, n=3) | - | no |
| mR@100 | 0.366 (0.343-0.382, n=3) | 0.286 (0.261-0.304, n=3) | - | no |
| F1@100 | 0.380 (0.359-0.396, n=3) | 0.283 (0.266-0.293, n=3) | - | no |
| zR@100 | 0.000 (0.000-0.000, n=3) | 0.300 (0.152-0.379, n=3) | - | no |

## group_7

| metric | human-trained | auto-trained | vlm-trained | ranges overlap |
|---|---|---|---|---|
| R@100 | 0.313 (0.299-0.338, n=3) | 0.307 (0.292-0.337, n=3) | - | yes |
| mR@100 | 0.308 (0.298-0.323, n=3) | 0.307 (0.289-0.334, n=3) | - | yes |
| F1@100 | 0.310 (0.299-0.330, n=3) | 0.307 (0.291-0.335, n=3) | - | yes |
| zR@100 | 0.005 (0.000-0.008, n=3) | 0.165 (0.094-0.221, n=3) | - | no |

## group_8

| metric | human-trained | auto-trained | vlm-trained | ranges overlap |
|---|---|---|---|---|
| R@100 | 0.178 (0.165-0.199, n=3) | 0.076 (0.062-0.085, n=3) | - | no |
| mR@100 | 0.174 (0.150-0.196, n=3) | 0.111 (0.088-0.127, n=3) | - | no |
| F1@100 | 0.176 (0.159-0.198, n=3) | 0.090 (0.073-0.102, n=3) | - | no |
| zR@100 | 0.000 (0.000-0.000, n=3) | 0.036 (0.018-0.061, n=3) | - | no |

## What the replication settles

- Pooled mR@100: human 0.326 vs auto 0.278; per-seed ranges do not overlap, so the human arm's headline advantage is larger than run-to-run variation at n=3 seeds per arm.
- Group 7 (the one test annotator with no measured convention defect): human 0.308 vs auto 0.307. The human arm leads on the mean and the per-seed ranges overlap, so the margin is not separable from seed variance.