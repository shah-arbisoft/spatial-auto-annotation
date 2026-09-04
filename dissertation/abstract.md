# Abstract

Scene-graph datasets for robotics are built by hand, a human deciding every
spatial relationship between every pair of objects, which keeps them small and
their labels inconsistent, as the authors of one robot-acquired dataset report.
This dissertation asks whether the human can be removed, for the seven
predicates that dataset defines (on, under, left/right of, in front of/behind,
near).

The pipeline computes each relationship instead of predicting it: off-the-shelf
models supply segmentation masks and monocular depth, objects are lifted to 3D
positions, and geometric rules decide every label, save one semantic
exception on support that ablation A10 shows geometry cannot remove.
Thresholds are fitted on
six of the nine annotator groups and validated on the other three; outputs
use the dataset's native formats. All 836 images are annotated in five minutes
on a consumer GPU, at 20 times the manual pass's density.

The automatic labels recover 7,276 of the 8,926 human relationships, 81%
weighting every triplet equally and 0.85 as the unweighted per-predicate mean
the field reports; on the held-out annotators the mean is 0.74 and five of the
seven predicates sit at 0.82 or better. The nine annotators labelled disjoint batches, so how
well two would have agreed cannot be measured and no human ceiling is claimed.
A blinded, decoy-controlled audit of 191 claims puts precision at 0.79–1.00 for
the lateral, depth and proximity predicates, on 24 samples each, so with wide
intervals, but **0.54 [0.42, 0.65]** on 71 for support, against 0.83 from
an independent vision-language judge. Both supersede an unblinded 0.77; support
is the second of two audits, the first measuring 0.40, after which a threshold
was refitted and a fresh sample drawn. Diagnosing every disagreement puts the
residual gap on the annotation itself, including two groups using opposite
conventions, and not on tool error (~7%). The images being consecutive frames
of one capture, the labels can also be checked without ground truth:
front/behind reproduces across viewpoints 0.96 of the time, so a predicate
recovering 0.70 of the human labels applies an unshared criterion rather than
guessing.

A classifier trained on the automatic labels reaches 0.75 mean recall against
held-out human annotations, where the human labels give 0.30 and self-training
0.36, on geometric features close kin to the rules that wrote the labels.
Repeated in a current scene-graph framework with a shared frozen
detector and three seeds per arm, that advantage disappears without reversing:
the arms rank level (mR@100 0.292 against 0.293), while the automatic model
recovers five times more of the relation types its annotation omits (zR@100
0.268 against 0.052) and reproduces across seeds more than eight times more
tightly. What difference remains sits on the two test annotators with a
measured labelling defect and reverses on the one without, a pattern
consistent with the metric rewarding annotation habits as well as spatial
correctness.

One link further down the chain the relations decide the outcome. Asked to
plan a grasp on 25 held-out scenes where an object rests on the target, and
scored only on whether the plan lifts that object off first, a planner clears
it in 0 of 25 given objects alone, 25 of 25 given human
relations, 19 of 25 given automatic ones and 25 of 25 given automatic and
vision-language sources together, identically on two planners of very different
capability. Automatic labels are at least the equal of human ones where the
criterion is annotation practice and better where it is geometric consistency,
and robot planning needs the second. On this dataset the bottleneck was
limiting not just its size but what it could teach a model.
