# Chapter 4: Fidelity against the Human Annotations (RQ1)

This chapter answers RQ1, from the protocol (§4.1) through headline recall
(§4.2), audited precision (§4.3–§4.4), the hardest predicate pair per
annotator (§4.5), the residual human cost (§4.7) and every remaining miss
diagnosed (§4.10), to full automation with a detector in the loop (§4.11).

Every one of those measures is bounded by the labels it is scored against,
so the last two sections leave the gold behind: whether the labels survive
the camera moving (§4.12), and whether a vision-language model would have
done the job instead (§4.13). Three further studies carry no gold at all and
are reported in Appendix E: video from outside the calibration domain (E.4),
1,766 unlabelled robot frames (E.6), and the study now re-estimating
precision with disinterested judges (E.3).

## 4.1 Protocol

Four counts recur below and each names a different set, so they are fixed
here. The released subset holds **884** frames; **838** carry a non-empty
annotation; the pipeline runs on **836**, the two dropped being annotation
files with no matching image (§4.3); and **802** of those yield at least one
ordered pair, the remaining 34 holding a single object. Two properties of
the gold then shape the protocol. It is **sparse**: 8,790 of 84,880 ordered
pairs (~10%) carry a human label, so a label absent from the gold is
unexamined, not wrong, and raw precision undercounts. And annotator
behaviour is **uneven** (Chapter 3 for `near`, §4.5 for a second case), so
agreement is reported per annotator group as well as pooled.

The evaluation therefore uses per-predicate **recall of the human triplets**
as the primary metric, following the source paper's convention; **restricted
precision, recall and F1** on annotated pairs; a **manual audit** of a
stratified sample of the extra predictions, for an unbiased true-precision
estimate; per-type **flag rates**, costing the human-in-the-loop claim; and
three baselines, random, majority and **box-only geometry** (box centres, no
masks, no depth). Boxes and classes are ground truth throughout (PredCls), so
box IoU does not apply; detector-in-the-loop results sit with the ablations.
Thresholds were fitted on annotator groups 0–5 only, with 6–8 held out. Each
group is also a contiguous block of the capture, so the split holds out an
unseen arrangement as well as an unseen annotator (§4.12).

{{fig:qualitative-examples}} shows what the tool emits on two frames, one
from a calibration group and one from a held-out group, before any of the
numbers below abstract away from it.

## 4.2 Headline: recall of the human triplets

{{fig:rq1-recall}} plots the per-predicate result; the table below carries
the same figures with the baselines.

| Predicate | Gold | Ours | Ours (held-out) | Random | Majority | Box-only |
|---|---|---|---|---|---|---|
| on | 1465 | 0.81 | 0.85 | 0.13 | 0.00 | 0.84 |
| under | 1001 | 0.75 | 0.82 | 0.16 | 0.00 | 0.77 |
| to the left of | 972 | 0.97 | 0.95 | 0.13 | 0.00 | 0.97 |
| to the right of | 1174 | 0.98 | 0.99 | 0.13 | 0.00 | 0.99 |
| in front of | 2013 | 0.70 | 0.20 | 0.13 | 1.00* | 0.00 |
| behind | 1584 | 0.71 | 0.37 | 0.16 | 0.00 | 0.00 |
| near | 717 | 1.00† | 1.00 | 0.16 | 0.00 | 0.87 |
| **mean** | 8,926 | **0.85** | **0.74** | 0.14 | 0.14 | 0.63 |

The held-out front/behind cells (0.20/0.37) are dominated by two held-out
annotator groups that labelled the pair with the *opposite direction
convention*; §4.5 decomposes this, and convention-aligned depth recall is
0.84.

These are population values, not sample estimates: every human triplet is
scored. Whether another batch would give the same numbers is answered by a
cluster bootstrap over images (`eval/uncertainty.py`, 2,000 resamples, whole
images resampled so triplets sharing a scene, a depth map and an annotator
stay together). The 95% intervals are narrow where the tool is strong and
materially wider on the depth pair, which is the signature of a predicate
decided by scene composition, which no stable rule governs; the bootstrap
ran on the labels of §4.9 and its widths, not its centres, are what the
argument uses. Held-out intervals appear in §4.6.

Three observations. **(i)** The tool recovers 81% of all human triplets
(7,276 of 8,926; mean 0.845, and 0.74 on annotators no threshold ever saw)
against 14% for random and 23% for majority on the same triplet-weighted
basis. The mean row above is per-predicate and puts majority at 0.14, which
is the right number for a per-predicate question and the wrong one for this
comparison: guessing `in front of` everywhere recovers 2,013 of 8,926
triplets because that predicate is a quarter of the gold. Both figures are
reported so neither reading can flatter the tool by itself. **(ii)**
Box-only matches the full pipeline on every box-computable predicate, so
masks contribute almost nothing to on/under/left/right/near recall and the
pipeline's advantage is confined to the depth pair (0.70/0.71 against 0.00).
Even there the ground-plane fallback, a pure box cue, needs masks to fire,
because its elevation guard is mask-contact evidence (§4.9). **(iii)**
Held-out beats pooled on on/under/near and falls far below it on
front/behind. Both are annotator signatures: convention inversion for the
depth pair (§4.5), and direction-usage asymmetry for support, where several
groups label one direction only (group_2 records 188 *on* and no *under*;
group_8 only *under*) while the held-out groups' support labels happen to be
canonical stackings the rules recover at 0.95–1.00. Gold totals 8,926, not
8,928, because two stray annotation files without matching images are
excluded (Chapter 3).

## 4.3 Precision on the annotated pairs

Restricted to the 8,790 annotated pairs, precision runs 0.95 and 0.92 on the
support pair, 0.35 and 0.42 on the laterals, 0.43 and 0.35 on the depth pair
and 0.11 on `near` (full table in Appendix F.6). Every one is bounded below
by construction: the human typically recorded one or two relations where
several hold at once, a pair being near, left-of and in-front-of
simultaneously. `near` is the extreme case, emitted wherever the fitted gap
holds while only 3 of 9 annotator groups ever used the label. That is why
the protocol includes the audit (§4.4) instead of reading these columns at
face value.


## 4.4 Manual audit of extra predictions (true-precision estimate)

105 extra predictions (15 per predicate, seeded stratified sample) were
rendered with subject/object boxes and manually verdicted, using a
conservative rule under which any case not clearly true was marked wrong
(`outputs/audit/audit_sheet.csv`; verdicts to be independently
spot-checked). This audit was run against the *pre-gate box rule* and is
reported as found because it motivated the support-rule repairs; §4.9
re-audits the shipped rules and §4.14 audits them again under blinding, so
the support rows below describe a fixed failure, not the final tool. How far
it was fixed is the subject of those two sections, and they disagree: pooled
support precision moves 0.13 → 0.77 on an unblinded re-audit, back to 0.40
when the same rules are judged blind against decoys, and to 0.54 once the
threshold that caused the over-emission is refitted and audited again.

| Predicate | Correct / n | Precision est. | Wilson 95% CI |
|---|---|---|---|
| on | 1 / 15 | 0.07 | [0.01, 0.30] |
| under | 3 / 15 | 0.20 | [0.07, 0.45] |
| to the left of | 15 / 15 | 1.00 | [0.80, 1.00] |
| to the right of | 15 / 15 | 1.00 | [0.80, 1.00] |
| in front of | 15 / 15 | 1.00 | [0.80, 1.00] |
| behind | 15 / 15 | 1.00 | [0.80, 1.00] |
| near | 15 / 15 | 1.00 | [0.80, 1.00] |

The audit splits the predicates in two. For lateral, depth and proximity,
**every sampled extra prediction was correct**: §4.3's low restricted
precision is the sparse-gold artefact the protocol anticipated, and the dense
labels on unannotated pairs are trustworthy within the sample's bounds. For
support the picture inverts, extra `on`/`under` labels being mostly **false
fires**: box adjacency triggers on objects that merely project next to each
other (a book *behind* a bottle, a cube on the floor with a remote in front).
With §4.2's containment misses, both error directions point to one repair, a
mask-contact test instead of box adjacency, which the ablations evaluate. Two
caveats: 15 per predicate gives wide per-predicate intervals, and depth
verdicts on near-coincident objects were occasionally marginal (noted in the
sheet).

## 4.5 The front/behind decomposition: a second measured annotation defect

| Group | Gold | Emit rate | Agreement when committed | Convention | Raw recall | Aligned recall |
|---|---|---|---|---|---|---|
| group_0 | 724 | 0.94 | 0.95 | same | 0.89 | 0.89 |
| group_1 | 639 | 1.00 | 1.00 | same | 1.00 | 1.00 |
| group_2 | 351 | 0.85 | 1.00 | same | 0.85 | 0.85 |
| group_3 | 258 | 0.82 | 0.99 | same | 0.81 | 0.81 |
| group_4 | 65 | 1.00 | 0.57 | same | 0.57 | 0.57 |
| group_5 | 371 | 1.00 | 0.99 | same | 0.98 | 0.98 |
| group_6 | 415 | 0.99 | **0.05** | **inverted** | 0.05 | 0.94 |
| group_7 | 330 | 0.94 | 1.00 | same | 0.94 | 0.94 |
| group_8 | 444 | 0.84 | **0.02** | **inverted** | 0.02 | 0.82 |
| **overall** | 3597 | | | | **0.70** | **0.91** |

{{fig:front-behind-decomposition}} plots the same decomposition, where the
two inverted groups are the pair sitting alone near zero. (The table reports
the shipped cascade, depth ordering plus the ground-plane fallback of §4.9,
which lifted the emit rates of the abstention-heavy groups 2 and 3 by about
half.) The pooled 0.70 decomposes into three distinct causes:

1. **Direction agreement is near-perfect where the tool commits**, 0.95–1.00
   for six of the eight groups with meaningful counts. Genuine depth-ordering
   errors are rare.
2. **Two annotator groups used the inverted convention.** Groups 6 and 8
   agree with the committed direction 2–5% of the time; flipping their labels
   recovers 0.94/0.82 and aligned overall recall is 0.91 (alignment uses one
   disclosed bit per group, the majority direction of its own labels). After
   `near`, this is a second measured annotation defect, and it is the
   reference-frame ambiguity RoboSpatial formalises (§2.5) observed in the
   wild, with no frame declared in the guidance and two teams resolving it
   oppositely.
3. **The remaining gap is abstention, not error.** Groups 2 and 3 agree almost
   perfectly when the tool commits, but their pairs sit inside the `depth_eps`
   band *and* beyond the ground-plane fallback's reach, so the tool abstains:
   emit rates 0.58 and 0.54, roughly doubled by the fallback. The `depth_eps`
   and `plane_band` sweeps trade this against precision.

## 4.6 The tenth annotator, and what the annotators would score against each other

Per-group recall of each annotator's triplets (all predicates):
group_0 0.85 · group_1 0.90 · group_2 0.92 · group_3 0.88 · group_4 0.86 ·
group_5 0.93 · group_6 0.56 · group_7 0.90 · group_8 0.57. The dispersion is
almost entirely the two convention-inverted groups (6, 8): the tool agrees
with the *consistent* annotators at 0.85–0.93, a band the support refit of
§4.14 tightened by lifting group 3 from 0.72.

On the held-out annotators the cluster-bootstrap intervals of §4.2 put five of
the seven predicates at 0.82 or better and the depth pair far below, at
0.199 (0.148–0.257) and 0.369 (0.300–0.443); Appendix F.8 gives all seven with
their intervals. The gap between the two groups is
far wider than the sampling uncertainty, which is what makes the convention
explanation below a claim about the labels, not about noise.

Reading those numbers as a quality verdict requires a yardstick the dataset
does not supply: how well would two *human* annotators agree with each
other? The groups labelled disjoint image batches, so the quantity cannot be
measured directly, and that absence is itself a finding about the dataset's
construction. Two things can nevertheless be recovered by treating the
automatic annotator as a fixed common reference
(`eval/annotator_agreement.py`).

Annotator heterogeneity, measured without assumptions. The tool is
deterministic: it is literally the same labeller for every group, applying
one definition. Any variation in its agreement across annotators is
therefore variation in the *annotators*. Across the seven consistent groups
that variation spans 0.851 to 0.933 about a mean of 0.892, a spread of 0.082
(sd 0.028). That is a modest number and it is reported as one: on the
shipped labels the consistent annotators are much closer to interchangeable
than the pre-refit spread of 0.216 suggested, and the case that they differ
rests on the two inverted groups and the three measured defects of §4.5 and
§4.9 rather than on this band.

The yardstick this dataset cannot supply. The natural way to get one is to
bound it: with the tool as a fixed common reference, the Fréchet
inequalities turn each pair of annotator agreement rates into an interval
for those annotators' agreement with each other. No such interval is quoted
here, because deriving it exposed an assumption this dataset violates. The
inequalities hold for any joint distribution, but carrying one annotator's
rate onto another's batch presumes the two batches are comparable, and these
are not: mean objects per frame runs from 4.47 in group 4 to 14.30 in group
1, a coefficient of variation of 0.31 across the nine, and pair count grows
with the square of that. A sparse batch is an easier batch, so the
presumption is false here and not merely unverified. Appendix F.7 gives the
derivation, the interval it produces and the density figures, so the
objection can be checked rather than taken on trust.

Because the nine groups labelled disjoint batches of unequal difficulty,
this dataset cannot yield an inter-annotator agreement figure even by
bounding, and a replication that wants one has to collect overlapping
assignments. RQ1's comparability claim therefore rests on the trivial
baselines and the per-predicate audit (§4.8). Knowing which yardsticks a
dataset cannot support is part of describing it.

## 4.7 Flags: what review actually costs

31.5% of ordered pairs carry a flag: depth-ambiguous 19.3% (down from 29.5%
before the ground-plane fallback, which resolved a third of the depth
abstentions) and lateral-ambiguous 10.0% are *abstentions* (no label
emitted; nothing to review), while the borderline-near band, 8.5% of pairs,
is the genuine review queue. At a conservative 3 seconds per queued pair
this is ≈6 hours of review for the full dataset against the original
nine-annotator manual pass, and it is optional for the fidelity reported
above. The guidance the annotators worked from is vocabulary lists only, so
Chapter 3's specification is the first operational definition of these
predicates, which §7.3 draws out.


## 4.8 Answer to RQ1

The two axes divide the seven predicates differently, and saying so plainly
matters more than a single headline. On **recall**, five reach
human-comparable levels, 0.75 to 1.00 as Table 4.1 reports them, mean 0.85
and 0.74 on annotators no threshold ever saw; the exception is the depth
pair, at 0.70/0.71 pooled and 0.91 once the two inverted groups are aligned.
On **precision**, a different five audit blind at 0.79–1.00, the two
laterals, the two depth predicates and `near`, each on 24 samples, so the
claim they support is comparability; the exception there is support, at
0.535 [0.42, 0.65] on 71 samples of the shipped rule, up from 0.404 before
§4.14 traced the shortfall to a threshold fitted on a metric that could not
see the error it controls.

So no predicate is weak on both axes and none is strong on both except the
laterals and `near`. Support recalls well and cannot be trusted where it
adds; the depth pair is trustworthy where it commits and commits less often
than the annotators did. Section 9.1 answers RQ1 against the criteria of
§1.2.2 on that basis.

`near` appears in both fives and deserves its own note. It is recovered
completely once its inconsistent usage is accounted for (0.997 pooled, 1.00
held-out), which answers the predicate the source paper reports as failing
for every model it benchmarks (§2.2). That is a claim about
recall, and on the precision side `near` is the weakest of the five: 0.792
audited, the widest disagreement between the two judges, and two of its four
decoys accepted by the author (§4.14). A rule that fires on sixty times more
pairs than the annotators labelled will recover their labels almost by
construction, so this is the predicate where the recall figure most
overstates what is known.

Section 4.5 decomposes the depth pair's shortfall into calibrated abstention
and a convention the annotators did not share, in measured proportions, and
neither component is depth error. The residual human cost is an 8.5% review
queue (§4.7), against labels 20× denser than the human set.

## 4.9 Shipped from the ablations

Ten ablations were run. Seven sweep a shipped parameter over the cached
geometry and are re-runnable offline in about twenty seconds
(`eval/ablations.py`); two test whether a heavier perception stack would do
better and one whether geometry can replace the class guard, and all three
say no. Every parameter was selected on the training
annotator groups alone, with the held-out column reported and never
optimised against.

| # | What it tests | Setting | Verdict |
|---|---|---|---|
| A1 | support depth co-location gate | `on_depth_eps` 0.06 | **shipped**; held-out support F1 0.58 → 0.71 |
| A2 | front/behind abstention band | `depth_eps` 0.03 | **shipped**; bounds the trade (recall 0.71 at ε=0 for 0.26–0.36 precision) |
| A3 | lateral abstention band | `lateral_center_eps` 0.02 | **shipped**; recall flat to 0.02 while precision rises |
| A4 | proximity threshold | `near_T` 1.372 | **shipped**; the knee of the recall plateau, held-out recall 1.00 |
| A5 | mask-contact support rule | `on_contact_min` 0.60, re-fitted to 0.85 (§4.14) | **shipped**; held-out support F1 0.71 → 0.87, both error directions at once |
| A6 | `near` contact exclusion | on | **shipped**; costs 2 recalled triplets, prevents 4,084 labels contradicting the measured convention |
| A7 | ground-plane depth fallback | `plane_band` 0.005 | **shipped**; front/behind 0.52/0.55 → 0.70/0.71, mean recall 0.79 → 0.85 |
| A8 | larger depth model (Base, 4× parameters) | n/a | **declined**; +0.001/−0.002 front/behind, mean recall marginally lower |
| A9 | multi-frame depth (two-view triangulation) | n/a | **declined**; 0.706 against the monocular cascade's 0.902, on 9% of pairs |
| A10 | geometric drop fraction in place of the class guard | n/a | **declined**; the resting and held populations overlap, no threshold separates them (D.8) |

Three changed the headline table materially, and their order is this
section's argument. The audit localised a support precision failure; a
geometric insight (stacked objects share a camera distance) fixed half; a
perception upgrade (mask-bottom contact) fixed most of the rest while
*raising* recall, the rare change that improves both error directions at
once; and the ground-plane fallback then recovered most of the front/behind
abstention band without depth at all.

The two declined perception ablations bound where engineering can help: neither a
four-times-larger depth model nor two-view triangulation over the raw
capture improves the depth pair, and the second is 0.20 *worse* where it
answers at all. The limit is monocular ambiguity in the scenes, not model
capacity, and multi-view geometry inherits it instead of removing it; §7.2
takes up what follows.

Full derivations, calibration evidence, audit samples and the failure
structure of each refinement are given in **Appendix D**.

## 4.10 Failure gallery: every miss diagnosed

Each missed human triplet is diagnosed automatically by re-checking the rule's
individual conditions against the cached geometry and mask-contact maps
(`scripts/make_failure_gallery.py`; rendered examples in
`outputs/failure_gallery/`). Appendix D.7 breaks the 1,689 misses of the
shipped rule set down by predicate and cause.

Genuine depth-ordering errors remain 1–5% of front/behind misses, the
support misses are threshold trades on real contact evidence, which box
geometry does not explain, and `near` misses have all but vanished. The
convention-inverted *share* grew to 38–42% not because those misses
increased but because the ground-plane fallback shrank the abstention share
around them, total misses falling from 2,107 to 1,689. Misses attributable
to avoidable tool error across all seven predicates: ~7%.


## 4.11 Detector-in-the-loop: full automation, attributed

The deployment mode replaces ground-truth boxes with Grounding DINO
(Liu et al., 2024) zero-shot detection (short-noun text prompts for the six
classes; threshold 0.25, tuned in one disclosed iteration on a 20-image
trial), then runs the identical SAM2 → depth → rules stack
(`scripts/run_sgdet.py`, scored by `eval/sgdet_eval.py` with class-matched
greedy IoU ≥ 0.5).

End-to-end triplet recall over 836 images is **0.38** against the PredCls
headline, and the decomposition attributes the gap exactly. Zero-shot
detection recall spans 0.40 (cube) to 0.95 (human), and a triplet needs
*both* endpoints. **Conditioned on both being detected the relation layer
scores 0.85 mean, matching PredCls** (lateral 0.96/0.98, near 1.00, support
0.83/0.77; front/behind 0.69/0.70, computed before the ground-plane fallback
shipped, so those are a floor, and detectable pairs skew towards
well-separated objects). Detection accounts for the gap, and the rules
themselves are detector-agnostic.

Two caveats. Zero-shot open-vocabulary detection is the worst-case
detector, the trade §2.6 identifies, and it was used because the authors'
trained YOLOv10m weights were not available; with those the end-to-end gap
would largely close. And the 20-image trial over-estimated detection quality,
its scenes coming from one annotator batch. The same attribution holds on two
out-of-domain clips, where every visible failure is again a detection failure
(Appendix E.4).



## 4.12 Temporal redundancy and stability under viewpoint change

The 884 released images are not independent photographs. Pixel-matching them
against the 2,650-frame raw capture the supervising group later supplied
identifies them exactly: they are frames 000000–000883 of one continuous
walk, and each annotator group is a contiguous 100-frame block
(`group_0` = frames 0–99, and so on). That has one consequence for the
results already reported and enables one measurement they could not make.

**The qualification first.** A group is simultaneously an annotator identity
*and* a temporal block holding one arrangement, so the held-out split of §4.1
is held out by scene as well as by annotator. The annotator reading survives,
since an inverted front/behind convention (§4.5) and a `near` label used by
three groups in nine (§3.2) are labelling behaviours no arrangement of
furniture can produce, and the confound runs favourably: 0.76 on held-out
groups is generalisation to an unseen annotator *and* an unseen
arrangement.

**The measurement.** Consecutive frames show a scene from different
viewpoints, so the pipeline's verdicts can be checked against themselves with
no human labels: a relation fixed by geometry should survive the camera
moving, and one decided by a coin toss at a threshold should not. Frames were
segmented by content drift (§3.10), which compresses the released frames 2.7×
at τ = 10 and locates all eight known layout changes within five frames, and
each segment's predicates were propagated from its keyframe to the rest
(`eval/keyframe_propagation.py`). The propagation runs over the 802 frames that yield at least one ordered pair (§4.1), not the 884 released ones, which segment differently because a subset sits further apart in the original capture (E.2); at τ = 10 those 802
give 568 keyframes, leaving 234 propagated frames and 11,352 comparable object
pairs. Appendix E.2 reports the sweeps, the
object-matching rule and the coverage figures.

| Predicate | Stability | Recall (propagated) | Recall (per frame) |
|---|---|---|---|
| on | 0.878 | 0.798 | 0.798 |
| under | 0.878 | 0.760 | 0.740 |
| to the left of | 0.989 | 0.959 | 0.966 |
| to the right of | 0.989 | 0.989 | 0.989 |
| in front of | 0.958 | 0.682 | 0.648 |
| behind | 0.958 | 0.636 | 0.659 |
| near | 0.972 | 1.000 | 1.000 |
| **Mean** | **0.946** | **0.832** | **0.829** |

Skipping frames costs nothing measurable: mean recall under propagation is
0.832 against 0.829 per frame, and the ordering holds at every threshold
tested. The small advantage is likely an artefact of the selection rule, the
segment representative being the frame nearest the segment mean.

The first column holds the finding, and it is not the expected one.
Front/behind was the predicted loser, on the assumption its errors are depth
noise near the boundary, which is exactly what a viewpoint change perturbs.
It does not behave that way: it agrees with itself 0.958 of the time, above
`on` and `under` at 0.878, and still 0.911 at 89× compression, where segment
members are substantially different views. A predicate recalling 0.648 of the
human labels on these frames while agreeing with itself at 0.958 is not making
random errors;
it is making the same call repeatedly and disagreeing systematically. That
converges with A8, where a four-times-larger depth model changed nothing, and
with §4.5, where two groups labelled the pair oppositely. §7.2 develops what
follows.

**Limits.** The figures are an upper bound: only pairs matched between
keyframe and frame contribute, and those are the ones whose objects moved
least. Coverage also thins as segments grow, so aggressive compression leaves pairs uncovered instead of mislabelled, which Appendix E.2 traces to box
drift and not absent annotation, making a tracker the straightforward
remedy.


## 4.13 Would a vision-language model do this instead?

The three baselines of §4.2 are deliberately weak, and a reader is entitled
to ask about the strong one: if a vision-language model annotates this dataset
as well as the geometric pipeline does, the pipeline is unnecessary. Thirty
images, stratified across all nine annotator groups, go to the model with the
ground-truth boxes drawn on and numbered, and it answers by index
(`scripts/run_vlm_pilot.py`). That is the PredCls setting the pipeline is
evaluated in, so neither is scored on detection, and the prompt carries
Chapter 3's definitions verbatim, without which the run would measure §2.5's
reference-frame ambiguity, not accuracy. Two models were run, because
the objection to one is that a larger model would close the gap:
`gemini-flash-latest` is small and non-reasoning, `gemini-3.1-pro-preview` a
reasoning model an order of magnitude larger.

| Predicate | Gold | Flash | Pro | Pipeline |
|---|---|---|---|---|
| on | 57 | 0.667 | 0.737 | 0.860 |
| under | 47 | 0.660 | 0.766 | 0.809 |
| to the left of | 49 | 0.408 | 0.469 | 0.918 |
| to the right of | 49 | 0.327 | 0.388 | 0.939 |
| in front of | 80 | 0.188 | 0.237 | 0.650 |
| behind | 65 | 0.169 | 0.138 | 0.662 |
| near | 34 | 0.382 | 0.382 | 1.000 |
| **Mean** | **381** | **0.400** | **0.445** | **0.834** |

{{fig:rq1-with-vlm}} plots all three against the human triplets. On recall
both lose everywhere, and the size of the improvement is the point:
mean recall moves from 0.400 to 0.445, real but barely half the pipeline's
0.834, and on the depth pair 0.24 against 0.65 puts the model below the
geometric method's known weak point. Scaling the model does not scale the
ability being measured.

Recall alone would be an unfair verdict, because it rewards whoever asserts
more: on the 374 judged pairs the pipeline makes 885 assertions against 344
and 414. Restricted to those pairs, where precision is defined, the column
reverses, and **both models are more precise than the pipeline**, 0.419 and
0.389 against 0.347. They buy it with silence, and the price is steep enough
that both lose F1 on every predicate, 0.397 and 0.405 against 0.485 pooled
(Appendix E.1).

Most of that recall gap is silence, and the comparison has to say so. The
model was asked for every ordered pair, with the dataset's own definitions
and the same instruction to omit what it was unsure of that the rules
implement as abstention, so the two are answering the same question. It
nonetheless never addressed **171 of the 381 gold triplets at all**, 44.9%.
Scored only on the pairs it did judge, its recall is **0.686** rather than
0.378, and on the predicates where the tool's advantage looks largest the
gap closes entirely: `to the left of` 0.909 against 0.918, and `on` 0.864
against 0.860, where the model is marginally the better of the two on the
pairs it chose to judge. A headline recall of 0.40 therefore measures two
things at once, how often the model is wrong and how often it declines, and
only the first is a claim about spatial competence. What it is bad at is
*exhaustiveness*; where it speaks it is roughly as good as the geometry.

That is not a defence of the model as an annotator, since exhaustiveness is
the property this project exists to supply and a source skipping 45% of the
pairs cannot deliver it. But the fair statement is that the pipeline beats
it on coverage and matches it on judgement, which is the version this
dissertation should be held to. The asymmetry matters again in §4.14, which
asks the same family of model to judge claims it is handed rather than to
find them, and so asks only for the half measured here as sound.

What decides it is the shape of the failure, since neither model contradicts
itself or inverts the front/behind convention; what they do is fall silent,
and supply one direction of a symmetric pair without the other in a third of
cases, which are the two defects §4.5 measures in the *human* annotation.
**Asked to annotate, a capable vision-language model reproduces the
characteristic failure of the human process and not a geometric one**,
and its conservatism is both why its precision beats the pipeline's and why
most relations go unrecorded. That it is more precise where it speaks still
begins a case for it as an adjudicator on the depth pair, which §7.6 takes up.
Appendix E.1 gives the per-predicate figures, the diagnostics and the limits
of a thirty-image pilot.


## 4.14 Auditing the audit: blinding, decoys, and a second judge

Every precision figure so far rests on the author verdicting the author's
tool, the objection §2.9 raises and §7.4 concedes. More verdicts of the same
kind would only narrow an interval around a possibly biased centre, so this
section re-runs the audit with the three defects of §4.4 and §4.9 removed at
once.

**Design.** 242 items: 214 claims the tool emitted and **28 decoys**,
relations it did *not* emit and no annotator labelled, mixed in unmarked. The
decoys are the instrument: every item in the earlier sheets was a tool
assertion, so an auditor who simply agreed scored 100% and looked calibrated,
where here that auditor scores zero. Sampling, the class guard and the
separation of sheet from key are in Appendix E.7. The same 242 images, with the same definitions and the same instruction to
answer wrong when unsure, were put to `gemini-3.6-flash` as a second judge
independent of the author (`scripts/judge_audit_vlm.py`).

Why a model may judge what §4.13 shows it cannot annotate. One objection
arrives immediately: §4.13 spends a section establishing that a
vision-language model makes a poor annotator, and this section then gives
one a vote. The two tasks differ in the half that failed. What §4.13
measures is *coverage* — the model never addressed 171 of 381 gold triplets,
44.9%, and its headline recall is mostly that silence — while on the pairs
it did judge it was the *more precise* of the two, 0.419 against the
pipeline's 0.347. Judging a claim that is handed to it asks only for the
half that measured sound, since the item is supplied and nothing has to be
enumerated.

That would still be only an argument if the audit did not test it, and the
decoys test it. The model rejected 26 of 28 relations the tool never
emitted, against the author's 27 of 28, so on this pack it is a strict judge
and not one that agrees with whatever it is shown; the two reach Cohen's κ
0.601 over all items and 0.425 over the claims alone, moderate agreement
rather than an echo. What the model is not is a human. Two judges who
disagree at κ 0.601 are better evidence than one, and neither is the
independent human estimate Appendix E.3 was built to supply and has not yet
returned.

| Predicate | Author | Model |
|---|---|---|
| on | 16/43 0.372 [0.24, 0.52] | 25/43 0.581 [0.43, 0.72] |
| under | 22/51 0.431 [0.31, 0.57] | 35/51 0.686 [0.55, 0.80] |
| to the left of | 22/24 0.917 [0.74, 0.98] | 23/24 0.958 [0.80, 0.99] |
| to the right of | 23/24 0.958 [0.80, 0.99] | 22/24 0.917 [0.74, 0.98] |
| in front of | 23/24 0.958 [0.80, 0.99] | 22/24 0.917 [0.74, 0.98] |
| behind | 22/24 0.917 [0.74, 0.98] | 20/24 0.833 [0.64, 0.93] |
| near | 24/24 1.000 [0.86, 1.00] | 15/24 0.625 [0.43, 0.79] |
| **support pooled** | **38/94 0.404 [0.31, 0.51]** | **60/94 0.638 [0.54, 0.73]** |
| decoys rejected | 19/28 0.679 [0.49, 0.82] | 24/28 0.857 [0.69, 0.94] |

**The two precision measurements point in opposite directions, and which way
is diagnostic.** Section 4.3 measured precision on the pairs a human labelled;
this section measures it on the pairs a human did not. For five predicates the
first badly understates the second (`near` 0.11 against 1.000, the laterals
0.35 and 0.42 against 0.917 and 0.958), which is the sparse-gold artefact §4.1
anticipated. For support the relation inverts, and sharply: 0.95 and 0.93 on
annotated pairs against 0.372 and 0.431 off them under the pre-refit labels
these audits used.

The direction of that gap says what the human record *is*. A lateral relation
holds for nearly every ordered pair and the annotators wrote down a handful,
so their labels are a small sample of a large truth and the tool's extras are
mostly further instances of it. Support is rare and salient: a thing resting
on a thing is worth recording and was recorded, so the human labels are close
to the complete set of easy cases, the tool agrees with them, and what it adds
beyond them is mostly not there. Restricted precision therefore understates a
predicate whose gold is a sample and overstates one whose gold is nearly
exhaustive, and no single reading of §4.3 is correct for both. This is why the
protocol pairs it with an audit instead of reporting either alone.

The lateral, depth and proximity claims survive; support does not. At
0.404 the support figure is less than half what §4.9 reported and outside any
interval this dissertation previously stated. The two judges disagree on its
level, 0.404 against 0.638, and agree emphatically on its direction: both are
far below 0.9. Raw agreement is 0.814 and Cohen's κ 0.576.

**`near` is the least supported number in the table, and carries the most
labels.** The two judges agree closely on the laterals and the depth pair, to
within 0.042 and 0.083, and diverge on `near` by **0.375**: 1.000 against
0.625 on the same 24 images. The author also accepted two of the four `near`
decoys, against the model's one. That combination, the widest disagreement
and a permissive decoy score, sits underneath the predicate the tool emits
most freely: 43,388 ordered pairs against 717 in the human record. The
threshold generalised to a held-out annotator at recall 1.00 (§3.8), so the
*notion* is calibrated; what 24 samples cannot establish is that a rule firing
sixty times more often than the annotators did is right every time it fires.
The most the 1.000 supports is that no counter-example appeared in 24
draws, not that none exists.

The decoys establish this is not an auditor being harsh. Both judges
rejected **all eight** support decoys, so on support neither is disposed to
agree with the tool for the sake of it. The author is more generous
elsewhere and consistently so: three of four `behind` decoys accepted
against the model's one, and two of four for `in front of` and for `near`.
That is a measured author bias, confined to the family §4.5 shows the
annotators used inconsistently, reported and not corrected because the same
instruction governed both judges.

Most of the drop is the blinding, not the sample. The same rules on the
same data scored 0.77 unblinded (§4.9) and 0.404 blind, because every row of
the earlier sheet was known to be a tool emission and that knowledge is a
prior the decoys remove. The rest is the independence rule.

The cause is a threshold fitted where its error was invisible. Sorted by the
contact fraction the rule fires on, audited claims below 0.85 are correct 1
time in 11 (4/44) and above it 2 times in 3 (34/50). The value shipped at
that point, `on_contact_min` 0.60, came from Appendix D.2's fit on train F1
against the human annotation, which covers ~10% of ordered pairs: a false
positive on the other 90% is not in the gold and cost the fit nothing. The
plateau D.2 calls "uncritical" from 0.60 to 0.80 is flat because the metric
could not see the error the parameter controls. A second, independent signal
is the supporting object's size, since `on(A, B)` requires B to be able to
hold A up and a 20-pixel cube is not a surface.

The repair was fitted, shipped, and then re-audited from scratch. The
obvious response, raising the threshold until precision recovers, cannot be
evaluated on the sample that suggested it: a cut-off chosen by inspecting
these 94 verdicts and then scored against them would be optimistic by an
unknown amount, which is the error that produced the 0.9. The cut-off was
therefore fitted the way every threshold in Chapter 3 is fitted, on the 63
audited claims from annotator groups 0–5, where precision rises steeply to
0.686 at 0.85 and flattens after; on the 31 claims from groups 6–8 that no
part of the fit saw, it predicted **0.367 → 0.667**.

`on_contact_min` was then set to 0.85 and every experiment in this
dissertation re-run against the new labels. That prediction is not what this
section reports, because a projection from held-out items selected under the
*old* labels is still an extrapolation. A second pack was drawn from the new
emissions instead — 219 items, 191 claims and 28 decoys, same construction,
same blinding, same two judges — and audited independently. In the table
below **v3** is the pre-refit pack of §4.14, drawn at `on_contact_min`
0.60, and **v4** this fresh one, drawn at the shipped 0.85.

| Predicate | v3 author | v4 author (shipped) | v4 model |
|---|---|---|---|
| on | 0.372 | **14/31 0.452 [0.29, 0.62]** | 24/31 0.774 [0.60, 0.89] |
| under | 0.431 | **24/40 0.600 [0.45, 0.74]** | 35/40 0.875 [0.74, 0.95] |
| to the left of | 0.917 | **23/24 0.958 [0.80, 0.99]** | 24/24 1.000 [0.86, 1.00] |
| to the right of | 0.958 | **24/24 1.000 [0.86, 1.00]** | 24/24 1.000 [0.86, 1.00] |
| in front of | 0.958 | **21/24 0.875 [0.69, 0.96]** | 23/24 0.958 [0.80, 0.99] |
| behind | 0.917 | **23/24 0.958 [0.80, 0.99]** | 22/24 0.917 [0.74, 0.98] |
| near | 1.000 | **19/24 0.792 [0.60, 0.91]** | 16/24 0.667 [0.47, 0.82] |
| **support pooled** | 0.404 | **38/71 0.535 [0.42, 0.65]** | 59/71 0.831 [0.73, 0.90] |
| decoys rejected | 19/28 0.679 | **27/28 0.964 [0.82, 0.99]** | 26/28 0.929 [0.77, 0.98] |

The middle column is the shipped tool, and the 0.79–1.00 range quoted for the
non-support predicates elsewhere is `near` at its floor and `to the right of`
at its ceiling.

The change is confirmed and the projection was optimistic. Both judges
record a large improvement on an independent draw, and both record it below
the 0.667 the held-out fit predicted, held-out support recall falling from
0.92 to 0.843 in exchange. Fitting on one audit and validating on the next is
as far as this design can go towards an unbiased estimate, and the gap between
the two numbers is the price of the extrapolation the earlier draft quoted.

The auditor also improved between the two packs, which Appendix E.7 reads in
full. Neither that nor the model's parallel movement rescues the support
figure: at 0.535 the labels the tool adds beyond the human record on this
predicate are right about half the time.

## 4.15 A disinterested check, against a human baseline

Both judges in §4.14 carry an objection. One built the tool; the other is a
system §4.13 shows annotates poorly. Appendix E.3 specifies the arm that
answers the first, putting sampled claims to volunteers who did not build
the tool and are not shown what it predicted. It closed with 1,415 usable
judgements from 20 raters over 832 of the 1,000 claims in the pool, 83.2%, no
rater supplying more than 15% of them.

**The shipped support rule has still not been judged by anyone outside this
project.** The tool claims were drawn on 17 July, a month before
`on_contact_min` was re-fitted from 0.60 to 0.85, so every `on` and `under`
claim below comes from the superseded rule and belongs against §4.14's first
pack, not the shipped second one. For the five predicates the support
threshold does not touch, the labels are identical in both generations and
the distinction does not arise.

**What makes this arm readable is that it scores the annotators too.** Half
the pool is drawn from the human annotations rather than from the tool,
rendered through the identical pipeline, interleaved so a rater cannot tell
the two apart, and judged under the same instruction to answer WRONG when
unsure. Without it a low score on the tool would be uninterpretable: the
same raters and the same conservative rule might score any claim low. They
do not. On the human-written claims they answer TRUE 0.940 of the time
(395/420), against 0.726 on the tool's (299/412). The raters are not
uniformly severe, so what the tool scores is about the tool.

| Predicate | Volunteers | Author | Model |
|---|---|---|---|
| on | 21/63 0.333 | 16/43 0.372 | 25/43 0.581 |
| under | 31/63 0.492 | 22/51 0.431 | 35/51 0.686 |
| to the left of | 47/57 0.825 | 22/24 0.917 | 23/24 0.958 |
| to the right of | 44/49 0.898 | 23/24 0.958 | 22/24 0.917 |
| in front of | 54/62 0.871 | 23/24 0.958 | 22/24 0.917 |
| behind | 54/66 0.818 | 22/24 0.917 | 20/24 0.833 |
| near | 48/52 0.923 | 24/24 1.000 | 15/24 0.625 |
| **support pooled** | **52/126 0.413** | **38/94 0.404** | **60/94 0.638** |

**On support the author's verdicts survive the check exactly.** Pooled `on`
and `under` precision on the pre-refit labels is 0.413 for the volunteers
against the author's 0.404 — a difference of 0.009 between the person who
built the tool, blinded and working against decoys, and strangers with no
stake in the outcome. This is the objection §2.9 raises and §7.4 concedes,
answered by measurement. The model sits apart at 0.638, and `near` shows the
same shape from the other side: 0.923 and 1.000 from the two human judges
against the model's 0.625. Across all seven predicates the volunteers rank
the tool almost exactly as the author does, Spearman 0.96 against the
model's 0.34, and land 0.074 away on average. Two independent human judges
agreeing to that degree is the strongest evidence available that the audits
were not theatre.

**Against the human baseline, five predicates hold and support does not.**
On the five the support threshold does not touch, the tool scores 0.864
against the annotators' 0.926, a gap of 0.063. On `in front of` it is
marginally ahead. On support it scores 0.413 against 0.975, a gap of 0.563
whose intervals do not come close to overlapping. That is the same weakness
§4.14 found and the refit responded to, now measured against what human
annotation scores on the identical instrument, and it is the sharpest
statement of it in this dissertation.

One asymmetry limits how far the comparison can be pushed. The tool's claims
are its *extra* predictions, on pairs the annotators passed over; the
control claims are pairs they chose to record. Annotators record what is
clear, so some of every gap above is the difficulty of the claim rather than
the quality of the label. For a 0.063 gap that reservation may account for
most of it; for 0.563 it cannot.

**The author-bias check.** On the 147 claims carrying both a volunteer and
an author verdict the two agree 0.871 of the time, Cohen's κ 0.683.
Crowd-internal reliability is Krippendorff's α 0.703 across the same claims. Both are substantial on the conventional reading, and both roughly doubled as the sample grew, which is what a real signal does and noise does not. Neither approaches 1.0, and §2.3's account of spatial language predicts that better than rater carelessness does: some of the residual disagreement is over what the words mean, not over what the photograph shows.

**What this arm settles and what it leaves open.** It settles that the
author's audit did not run in the author's favour: on the labels both
judged, two independent human verdicts agree to 0.009 on the measurement
most exposed to that bias. It settles that the raters are not uniformly
harsh, because they score human annotation at 0.940 on the same instrument.
It does not settle the shipped tool, whose support rule postdates the
sample, and it cannot fully separate label quality from claim difficulty,
because the two arms are drawn from populations the annotators themselves
divided. §7.6 and §9.3 carry what survives; re-running the arm on post-refit
support labels is the one outstanding item that would close it.
