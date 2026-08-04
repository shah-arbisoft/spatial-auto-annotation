# Benchmark replication across seeds

Seeds [42, 43, 44]. Identical data, split and frozen detector in every run, so the spread is the relation model's own training variance. Source: re-evaluation (fixed zero-shot reference, per-group slices).

## full

| metric | human-trained | auto-trained | vlm-trained | ranges overlap |
|---|---|---|---|---|
| R@100 | 0.328 (0.312-0.349, n=3) | 0.254 (0.251-0.260, n=3) | 0.310 (0.296-0.327, n=3) | no |
| mR@100 | 0.326 (0.303-0.347, n=3) | 0.278 (0.268-0.289, n=3) | 0.329 (0.316-0.347, n=3) | no |
| F1@100 | 0.327 (0.308-0.348, n=3) | 0.265 (0.259-0.269, n=3) | 0.319 (0.306-0.337, n=3) | no |
| zR@100 | 0.003 (0.000-0.004, n=3) | 0.172 (0.157-0.196, n=3) | 0.337 (0.320-0.365, n=3) | no |

## group_6

| metric | human-trained | auto-trained | vlm-trained | ranges overlap |
|---|---|---|---|---|
| R@100 | 0.395 (0.376-0.412, n=3) | 0.281 (0.271-0.290, n=3) | 0.322 (0.309-0.345, n=3) | no |
| mR@100 | 0.366 (0.343-0.382, n=3) | 0.286 (0.261-0.304, n=3) | 0.336 (0.317-0.369, n=3) | no |
| F1@100 | 0.380 (0.359-0.396, n=3) | 0.283 (0.266-0.293, n=3) | 0.329 (0.315-0.356, n=3) | no |
| zR@100 | 0.000 (0.000-0.000, n=3) | 0.300 (0.152-0.379, n=3) | 0.288 (0.222-0.404, n=3) | no |

## group_7

| metric | human-trained | auto-trained | vlm-trained | ranges overlap |
|---|---|---|---|---|
| R@100 | 0.313 (0.299-0.338, n=3) | 0.307 (0.292-0.337, n=3) | 0.387 (0.365-0.399, n=3) | yes |
| mR@100 | 0.308 (0.298-0.323, n=3) | 0.307 (0.289-0.334, n=3) | 0.381 (0.365-0.395, n=3) | yes |
| F1@100 | 0.310 (0.299-0.330, n=3) | 0.307 (0.291-0.335, n=3) | 0.384 (0.365-0.395, n=3) | yes |
| zR@100 | 0.005 (0.000-0.008, n=3) | 0.165 (0.094-0.221, n=3) | 0.491 (0.486-0.496, n=3) | no |

## group_8

| metric | human-trained | auto-trained | vlm-trained | ranges overlap |
|---|---|---|---|---|
| R@100 | 0.175 (0.160-0.200, n=3) | 0.074 (0.061-0.084, n=3) | 0.122 (0.109-0.134, n=3) | no |
| mR@100 | 0.171 (0.142-0.197, n=3) | 0.109 (0.087-0.125, n=3) | 0.148 (0.134-0.159, n=3) | no |
| F1@100 | 0.173 (0.150-0.199, n=3) | 0.088 (0.071-0.101, n=3) | 0.134 (0.121-0.145, n=3) | no |
| zR@100 | 0.000 (0.000-0.000, n=3) | 0.036 (0.018-0.061, n=3) | 0.045 (0.034-0.055, n=3) | no |

## What the replication settles

- Pooled mR@100: human 0.326 vs auto 0.278; per-seed ranges do not overlap, so the human arm's headline advantage is larger than run-to-run variation at n=3 seeds per arm.
- Group 7 (the one test annotator with no measured convention defect): human 0.308 vs auto 0.307. The human arm leads on the mean and the per-seed ranges overlap, so the margin is not separable from seed variance.