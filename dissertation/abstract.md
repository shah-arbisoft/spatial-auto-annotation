# Abstract

Scene-graph datasets for robotics are built by hand, a human deciding every
spatial relationship between every pair of objects. The bottleneck keeps
such datasets small and their labels inconsistent, as the authors of one
robot-acquired dataset report. This dissertation asks whether that human can
be removed, for the seven predicates the dataset defines (on, under,
left/right of, in front of/behind, near).

The pipeline computes each relationship instead of predicting it:
off-the-shelf models measure the scene (segmentation masks and monocular
depth), each object is lifted to a 3D position, and geometric rules decide
every label. Thresholds are fitted on six of the nine annotator groups and
validated on the held-out three; outputs use the dataset's native formats.
All 836 images are annotated in five minutes on a consumer GPU, at 20 times
the manual pass's label density.

Against 8,926 human relationships the automatic labels recover five of the
seven predicates at 0.82 or better (0.85 mean recall, 0.74 on held-out
annotators); the nine annotators labelled disjoint batches, so how well two
of them would have agreed cannot be measured here and no human ceiling is
claimed. A blinded, decoy-controlled audit of 191 sampled claims
puts precision at 0.79–1.00 for the lateral, depth and proximity predicates,
on 24 samples each and so with wide intervals, but at **0.54 [0.42, 0.65]**
on 71 samples for support, against 0.83 from an independent vision-language
judge. Both supersede an earlier unblinded estimate of 0.77, and the support
figure is the second of two audits: the first measured 0.40, a threshold was
refitted in response, and the second was drawn afresh from the corrected
labels. The hardest pair, in front of/behind, uses a cascade of relative
depth and a ground-plane cue; diagnosing every disagreement puts the
residual gap on the annotation itself, including two groups that used
opposite conventions, and not on tool error (~7%). Because the images are
consecutive frames of one capture, the labels can be checked against
themselves without ground truth: the pipeline reproduces its front/behind
verdict across viewpoints 0.96 of the time, so a predicate recovering only
0.70 of the human labels is applying an unshared criterion and not guessing.

A classifier trained on the automatic labels reaches 0.75 mean recall
against held-out human annotations, against 0.30 for human labels and 0.36
when those are stretched by self-training.

Repeated in the source paper's benchmark framework, with a shared frozen
detector and three seeds per arm, that advantage disappears without
reversing: the two arms rank level against human test annotation (mR@100
0.292 against 0.293), while the automatic model recovers five times more of
the relation types its own annotation omits (zR@100 0.268 against 0.052) and
reproduces itself across seeds more than eight times more tightly. What
difference remains sits on the two test annotators carrying a measured
labelling defect and reverses on the one without, so the metric rewards
annotation habits as well as spatial correctness. One link down the chain it
is the relations that decide the outcome: asked for a
safe grasp plan on 25 held-out scenes where an object rests on the target, a
planner clears it in 0 of 25 given objects alone, 25 of 25 given human
relationships, 19 of 25 given automatic ones and 25 of 25 given the union of
the automatic and vision-language sources, identically on two planners
of very different capability. Automatic labels are at least the equal of human
ones where the criterion is annotation practice and better where it is
geometric consistency, and robot planning needs the second. The bottleneck removed was
limiting not only the dataset's size but what it could teach.
