# Abstract

Scene-graph datasets for robotics are built by hand: every spatial
relationship between every pair of objects is decided by a human annotator.
This bottleneck keeps such datasets small, and the manual labels it produces
are inconsistent: problems the authors of a recent robot-acquired
spatial-relationship dataset report themselves. This dissertation asks whether
the human can be removed from the labelling loop entirely, for the seven
spatial predicates the dataset defines (on, under, left/right of, in front
of/behind, near).

The proposed pipeline computes, rather than predicts, each relationship:
off-the-shelf perception models measure the scene (segmentation masks and
monocular depth), each object is lifted to a 3D position, and explicit
geometric rules decide every label. Rule thresholds are fitted only on six of
the dataset's nine annotator groups and validated on the held-out three, and
outputs are written in the dataset's native formats. The full 836-image
dataset is annotated in about five minutes on a consumer GPU, with 20 times
the label density of the manual pass.

Validated against 8,926 human-annotated relationships, the automatic labels
match or exceed the human process on five of seven predicates (0.85 mean
recall, 0.76 on held-out annotators; manually audited precision ≈ 1.0 for the
lateral and proximity predicates and ≈ 0.9 for support). The hardest pair,
in front of/behind, is decided by a two-stage cascade (relative depth, then
a ground-plane projection cue where depth cannot separate the objects), and
diagnosing every remaining disagreement attributes the residual gap to
measured properties of the human annotation itself, including two annotator
groups that labelled the pair with opposite conventions, rather than to tool
error (~7% of misses). In a controlled downstream experiment, a classifier
trained on the automatic labels reaches 0.76 mean recall against held-out
human annotations, versus 0.30 when trained on the human labels and 0.36 when
those labels are stretched by self-training, the standard semi-supervised
remedy: at this dataset's annotation scale, dense and consistent computed
labels are better training material than the sparse human labels they
replace, and better than any attempt to extrapolate from them. The annotation bottleneck
this pipeline removes was not only limiting dataset size; it was limiting what
the dataset could teach.
