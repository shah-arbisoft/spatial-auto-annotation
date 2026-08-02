# Ablation sweeps (offline re-evaluation from cached geometry)

## A1 — Support depth-co-location gate (`on_depth_eps`)

| eps | on recall | on P(restr.) | under recall | support F1 train | support F1 held-out | on emitted |
|---|---|---|---|---|---|---|
| off | 0.840 | 0.575 | 0.773 | 0.642 | 0.579 | 5521 |
| 0.15 | 0.838 | 0.622 | 0.771 | 0.683 | 0.619 | 4303 |
| 0.12 | 0.837 | 0.642 | 0.768 | 0.696 | 0.633 | 4051 |
| 0.10 | 0.835 | 0.670 | 0.766 | 0.716 | 0.652 | 3796 |
| 0.08 | 0.831 | 0.702 | 0.759 | 0.728 | 0.689 | 3454 |
| 0.06 | 0.818 | 0.726 | 0.744 | 0.732 | 0.707 | 3170 |
| 0.05 | 0.795 | 0.731 | 0.720 | 0.726 | 0.703 | 2985 |
| 0.04 | 0.772 | 0.745 | 0.687 | 0.713 | 0.721 | 2793 |
| 0.03 | 0.726 | 0.773 | 0.638 | 0.699 | 0.728 | 2458 |

calibrated on_depth_eps = **0.06** — selected by support F1 on TRAIN groups only; the held-out column is reported, never optimised.

## A2 — Front/behind abstention band (`depth_eps`)

| eps | front recall | front P(restr.) | behind recall | behind P(restr.) |
|---|---|---|---|---|
| 0.00 | 0.714 | 0.355 | 0.722 | 0.263 |
| 0.01 | 0.656 | 0.392 | 0.653 | 0.296 |
| 0.02 | 0.567 | 0.436 | 0.590 | 0.366 |
| 0.03 | 0.517 | 0.467 | 0.547 | 0.413 |
| 0.05 | 0.449 | 0.527 | 0.464 | 0.492 |
| 0.08 | 0.374 | 0.594 | 0.380 | 0.572 |

## A3 — Lateral abstention band (`lateral_center_eps`)

| eps | left recall | left P(restr.) | right recall | right P(restr.) |
|---|---|---|---|---|
| 0.000 | 0.973 | 0.217 | 0.986 | 0.261 |
| 0.005 | 0.973 | 0.259 | 0.986 | 0.311 |
| 0.010 | 0.972 | 0.295 | 0.986 | 0.351 |
| 0.020 | 0.965 | 0.350 | 0.985 | 0.417 |
| 0.040 | 0.832 | 0.445 | 0.955 | 0.539 |

## A4 — Proximity threshold (`near_T`)

| T | near recall | near recall (held-out) | near P(restr.) | emitted |
|---|---|---|---|---|
| 0.600 | 0.911 | 0.624 | 0.125 | 22536 |
| 0.800 | 0.944 | 0.710 | 0.122 | 27972 |
| 1.000 | 0.967 | 0.817 | 0.119 | 33166 |
| 1.200 | 0.990 | 0.978 | 0.117 | 38154 |
| 1.372 | 0.997 | 1.000 | 0.116 | 42432 |
| 1.600 | 0.997 | 1.000 | 0.114 | 47556 |
| 1.900 | 0.997 | 1.000 | 0.114 | 53350 |
| 2.200 | 0.997 | 1.000 | 0.113 | 58354 |

## A5 - Mask-contact support rule (`on_contact_min`)

| contact_min | on recall | on P(restr.) | under recall | support F1 train | support F1 held-out | on emitted |
|---|---|---|---|---|---|---|
| 0.10 | 0.939 | 0.710 | 0.894 | 0.784 | 0.710 | 3349 |
| 0.20 | 0.930 | 0.739 | 0.883 | 0.796 | 0.740 | 3097 |
| 0.30 | 0.924 | 0.765 | 0.874 | 0.809 | 0.773 | 2863 |
| 0.40 | 0.914 | 0.803 | 0.864 | 0.840 | 0.800 | 2533 |
| 0.50 | 0.896 | 0.840 | 0.840 | 0.851 | 0.836 | 2272 |
| 0.60 | 0.879 | 0.879 | 0.813 | 0.856 | 0.868 | 2042 |
| 0.70 | 0.859 | 0.902 | 0.790 | 0.852 | 0.888 | 1866 |
| 0.80 | 0.832 | 0.943 | 0.766 | 0.854 | 0.894 | 1646 |

calibrated on_contact_min = **0.6** (train-group selection; compare the box-rule A1 row at the shipped on_depth_eps).


## A6 - Correction step: near contact-exclusion on/off

| setting | near recall | near P(restr.) | near emitted |
|---|---|---|---|
| on (shipped) | 0.997 | 0.116 | 42432 |
| off | 1.000 | 0.083 | 46516 |

The exclusion costs 2 recalled triplets and prevents 4,084 near labels on contact pairs - labels that would contradict the measured human convention (near co-occurs with on/under on 0 of 469 pairs).


## A7 - Ground-plane depth fallback (`plane_band`)

| band | front recall | front (held-out) | front P(restr.) | behind recall | behind (held-out) | behind P(restr.) |
|---|---|---|---|---|---|---|
| off | 0.517 | 0.167 | 0.467 | 0.547 | 0.264 | 0.413 |
| 0.000 | 0.650 | 0.197 | 0.416 | 0.666 | 0.369 | 0.345 |
| 0.005 | 0.640 | 0.197 | 0.429 | 0.655 | 0.347 | 0.356 |
| 0.010 | 0.623 | 0.197 | 0.445 | 0.643 | 0.329 | 0.370 |
| 0.020 | 0.580 | 0.187 | 0.462 | 0.607 | 0.303 | 0.392 |
| 0.050 | 0.533 | 0.171 | 0.472 | 0.562 | 0.271 | 0.414 |

shipped plane_band = **0.005** - selected on train groups (added commits at 0.91 direction agreement; held-out group 7: all added commits correct). The fallback fires only when the depth rule abstained and neither object rests on another by the tool's own contact evidence; without masks it is off (A2 is the pure-depth trade for reference).
