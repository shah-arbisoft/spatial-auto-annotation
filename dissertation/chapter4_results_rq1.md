# Chapter 4: Fidelity against the Human Annotations (RQ1)

This chapter answers RQ1: the protocol (§4.1), headline recall (§4.2),
audited precision (§4.3–§4.4), the hardest predicate pair per annotator
(§4.5), the residual human cost (§4.7), every remaining miss diagnosed
(§4.10), and full automation with a detector in the loop (§4.11). Because
each of those measures is bounded by the labels it is scored against, §4.12
and §4.13 leave the gold behind: whether the labels survive the camera
moving, and whether a vision-language model would have done the job instead.
Three further studies carry no gold at all and sit in Appendix E: video from
outside the calibration domain (E.4), 1,766 unlabelled robot frames (E.6),
and the re-estimate of precision by disinterested judges (E.3, reported in
§4.15).

## 4.1 Protocol

Four counts recur below. The released subset holds **884** frames; **838**
carry a non-empty annotation; the pipeline runs on **836**, two annotation
files having no matching image (§4.3); and **802** of those yield at least
one ordered pair, the remaining 34 holding a single object. Two properties
of the gold shape the protocol. It is **sparse** — 8,790 of 84,880 ordered
pairs (~10%) carry a human label, so a label absent from the gold is
unexamined, not wrong, and raw precision undercounts — and annotator
behaviour is **uneven** (Chapter 3 for `near`, §4.5 for a second case), so
agreement is reported per annotator group as well as pooled.

The metrics: per-predicate **recall of the human triplets**, the primary
one, following the source paper's convention; **restricted precision,
recall and F1** on annotated pairs; a **manual audit** of a stratified
sample of the extra predictions, for an unbiased true-precision estimate;
per-type **flag rates**, costing the human-in-the-loop claim; and three
baselines — random, majority, and **box-only geometry** (box centres, no
masks, no depth). Boxes and classes are ground truth throughout (PredCls),
so box IoU does not apply; detector-in-the-loop results sit with the
ablations. Thresholds were fitted on annotator groups 0–5 only, with 6–8
held out; each group is also a contiguous block of the capture, so the split
holds out an unseen arrangement as well as an unseen annotator (§4.12).

{{fig:qualitative-examples}} shows what the tool emits on two frames, one
from a calibration group and one held out.

## 4.2 Headline: recall of the human triplets

{{fig:rq1-recall}} plots the per-predicate result; the table adds the
baselines.

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
groups that labelled the pair under the *opposite direction convention*;
§4.5 decomposes this, and convention-aligned depth recall is 0.84. These are
population values, every human triplet being scored; whether another batch
would agree is answered by a cluster bootstrap over images
(`eval/uncertainty.py`, 2,000 resamples, whole images resampled so triplets
sharing a scene, a depth map and an annotator stay together). Its 95%
intervals are narrow where the tool is strong and materially wider on the
depth pair, the signature of a predicate decided by scene composition; the
bootstrap ran on the labels of §4.9, its widths rather than its centres
carry the argument, and held-out intervals appear in §4.6.

Three observations. **(i)** The tool recovers 81% of all human triplets
(7,276 of 8,926; mean 0.845, and 0.74 on annotators no threshold ever saw)
against 14% for random and 23% for majority on the same triplet-weighted
basis. The mean row puts majority at 0.14 — right for a per-predicate
question, wrong for this comparison, since guessing `in front of` everywhere
recovers 2,013 of 8,926 triplets; both figures are reported so neither
reading can flatter the tool. **(ii)** On recall alone box-only is level
with the pipeline on the laterals and slightly ahead on support (0.84 and
0.77 against 0.81 and 0.75) while falling 0.13 behind on `near`; but the
mask rule was adopted because it lifts support *precision* and recall
together (held-out support F1 0.71 to 0.87; A5, §4.9), and a looser box rule
buys its extra recall with the false fires §4.4 audits. The pipeline's
unshared advantage is the depth pair (0.70/0.71 against 0.00), and even
there the ground-plane fallback, a pure box cue, needs masks to fire, its
elevation guard being mask-contact evidence (§4.9). **(iii)** Held-out beats
pooled on on/under/near and collapses on front/behind — both annotator
signatures: convention inversion for the depth pair (§4.5), and
direction-usage asymmetry for support (group_2 records 188 *on* and no
*under*; group_8 only *under*), the held-out groups' support labels being
canonical stackings the rules recover at 0.95–1.00. Gold totals 8,926, not
8,928: two stray annotation files without matching images are excluded
(Chapter 3).

## 4.3 Precision on the annotated pairs

Restricted to the 8,790 annotated pairs, precision runs 0.95 and 0.92 on the
support pair, 0.35 and 0.42 on the laterals, 0.43 and 0.35 on the depth pair
and 0.11 on `near` (full table in Appendix F.6). Every one is bounded below
by construction: the human typically recorded one or two relations where
several hold at once. `near` is the extreme case, emitted wherever the
fitted gap holds while only 3 of 9 annotator groups ever used the label.
That is why the protocol includes the audit (§4.4) instead of reading these
columns at face value.

## 4.4 Manual audit of extra predictions (true-precision estimate)

105 extra predictions (15 per predicate, seeded stratified sample) were
rendered with subject/object boxes and manually verdicted, any case not
clearly true marked wrong (`outputs/audit/audit_sheet.csv`). The audit ran
against the *pre-gate box rule* and is reported as found because it
motivated the support-rule repairs; §4.9 re-audits the shipped rules and
§4.14 audits them again under blinding, and they disagree: pooled support
precision moves 0.13 → 0.77 on an unblinded re-audit, back to 0.40 when the
same rules are judged blind against decoys, and to 0.54 once the threshold
that caused the over-emission is refitted and audited again. The support
figures below therefore describe a fixed failure, not the final tool.

The audit splits the predicates in two. For lateral, depth and proximity,
**every sampled extra prediction was correct** — 15/15 each, Wilson 95%
interval [0.80, 1.00] — so §4.3's low restricted precision is the
sparse-gold artefact the protocol anticipated, and the dense labels on
unannotated pairs are trustworthy within the sample's bounds. For support
the picture inverts: `on` 1/15 (0.07, Wilson [0.01, 0.30]) and `under` 3/15
(0.20, [0.07, 0.45]), the extras being mostly **false fires** where box
adjacency triggers on objects that merely project next to each other (a book
*behind* a bottle, a cube on the floor with a remote in front). With §4.2's
containment misses, both error directions point to one repair, a
mask-contact test instead of box adjacency, which the ablations evaluate.
Two caveats: 15 per predicate gives wide intervals, and depth verdicts on
near-coincident objects were occasionally marginal (noted in the sheet).

## 4.5 The front/behind decomposition: a second measured annotation defect

{{fig:front-behind-decomposition}} decomposes front/behind by annotator
group — agreement where the tool commits, deliberate abstention, and the two
convention-inverted groups sitting alone near zero. The per-group table is
in Appendix F.9; it reports the shipped cascade of §4.9, whose ground-plane
fallback roughly doubled the emit rates of the abstention-heavy groups 2 and
3. The pooled 0.70 decomposes into three causes:

1. **Direction agreement is near-perfect where the tool commits**, 0.95–1.00
   for six of the eight groups with meaningful counts. Genuine depth-ordering
   errors are rare.
2. **Two annotator groups used the inverted convention.** Groups 6 and 8
   agree with the committed direction 2–5% of the time; flipping their labels
   recovers 0.94/0.82, and aligned overall recall is 0.91 (alignment uses one
   disclosed bit per group, the majority direction of its own labels). After
   `near`, this is a second measured annotation defect — the reference-frame
   ambiguity RoboSpatial formalises (§2.5) observed in the wild, no frame
   declared in the guidance and two teams resolving it oppositely. The shape
   of the distribution rules out simple subjectivity: a genuinely contested
   judgement would scatter agreement around chance, whereas these two groups
   sit at 0.02–0.05 and the other six at 0.95–1.00. Agreement that far
   *below* chance is a sign flip, not a difference of opinion, and neither
   convention is the wrong one.
3. **The remaining gap is abstention, not error.** Groups 2 and 3 agree
   almost perfectly when the tool commits, but their pairs sit inside the
   `depth_eps` band and beyond the fallback's reach, so the tool abstains:
   emit rates 0.58 and 0.54. The `depth_eps` and `plane_band` sweeps trade
   this against precision.

## 4.6 The tenth annotator, and what the annotators would score against each other

Per-group recall of each annotator's triplets (all predicates):
group_0 0.85 · group_1 0.90 · group_2 0.92 · group_3 0.88 · group_4 0.86 ·
group_5 0.93 · group_6 0.56 · group_7 0.90 · group_8 0.57. The dispersion is
almost entirely the two convention-inverted groups (6, 8): the tool agrees
with the *consistent* annotators at 0.85–0.93, a band the support refit of
§4.14 tightened by lifting group 3 from 0.72. On the held-out annotators the
cluster-bootstrap intervals of §4.2 put five of the seven predicates at 0.82
or better and the depth pair far below, at 0.199 (0.148–0.257) and 0.369
(0.300–0.443); Appendix F.8 gives all seven. That gap is far wider than the
sampling uncertainty, which is what makes §4.5's convention explanation a
claim about the labels, not about noise.

Reading these numbers as a quality verdict needs a yardstick the dataset
does not supply: how well two *human* annotators would agree. The groups
labelled disjoint image batches, so the quantity cannot be measured directly
— itself a finding about the dataset's construction. Treating the
deterministic tool as a fixed common reference (`eval/annotator_agreement.py`)
recovers a bound: none of the variation in its per-group agreement comes
from the tool, but the nine batches differ threefold in object density
(mean objects per frame 4.47 to 14.30, coefficient of variation 0.31, pair
count growing with the square), so the variation carries batch difficulty as
well as annotator behaviour and is an **upper bound on annotator
heterogeneity, not a measurement of it**. Across the seven consistent groups
it spans 0.851 to 0.933 about a mean of 0.892 (spread 0.082, sd 0.028): on
the shipped labels the consistent annotators are much closer to
interchangeable than the pre-refit spread of 0.216 suggested, and the case
that they differ rests on the two inverted groups and the measured defects
of §4.5 and §4.9, not on this band.

The same non-exchangeability defeats the natural yardstick. With the tool as
a common reference, the Fréchet inequalities would turn each pair of
agreement rates into an interval for those annotators' agreement with each
other; but carrying one annotator's rate onto another's batch presumes
comparable batches, and a sparse batch is an easier batch, so the
presumption is false here, not merely unverified. Appendix F.7 gives the
derivation, the interval it produces and the density figures. This dataset
cannot yield an inter-annotator agreement figure even by bounding — a
replication that wants one has to collect overlapping assignments — so
RQ1's comparability claim rests on the trivial baselines and the
per-predicate audit (§4.8).

## 4.7 Flags: what review actually costs

31.5% of ordered pairs carry at least one flag; the kinds overlap, so they
sum past that (Appendix C.9). Depth-ambiguous 19.3% (down from 29.5% before
the ground-plane fallback resolved a third of the depth abstentions) and
lateral-ambiguous 10.0% are *abstentions* — no label emitted, nothing to
review — while the borderline-near band, 8.5% of pairs, is the genuine
review queue. At a conservative 3 seconds per queued pair this is ≈6 hours
of review for the full dataset against the original nine-annotator manual
pass, and it is optional for the fidelity reported above. The guidance that
survives in the released materials is vocabulary lists only, so Chapter 3's
specification is the first *executable* definition of these predicates,
which §7.3 draws out.

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
for every model it benchmarks (§2.2). That is a claim about recall; on the
precision side `near` is the weakest of the five, 0.792 audited, the widest
disagreement between the two judges, and two of its four decoys accepted by
the author (§4.14). A rule that fires on sixty times more pairs than the
annotators labelled will recover their labels almost by construction, so
this is the predicate where the recall figure most overstates what is known.

Section 4.5 decomposes the depth pair's shortfall into calibrated abstention
and a convention the annotators did not share, in measured proportions.
Genuine depth error is the remainder rather than absent — Appendix D.7
measures it at 1% of `in front of` misses and 6% of `behind` — so the
dominant terms are not depth error, a weaker claim than none of it being.
The residual human cost is an 8.5% review queue (§4.7), against labels 20×
denser than the human set.

## 4.9 Shipped from the ablations

Ten ablations were run: seven sweep a shipped parameter over the cached
geometry and re-run offline in about twenty seconds (`eval/ablations.py`);
two test whether a heavier perception stack would do better and one whether
geometry can replace the class guard, and all three say no. Every parameter
was selected on the training annotator groups alone, the held-out column
reported and never optimised against.

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

Three changed the headline table materially, in this order: the audit
localised a support precision failure; a geometric insight (stacked objects
share a camera distance) fixed half; a perception upgrade (mask-bottom
contact) fixed most of the rest while *raising* recall, the rare change
improving both error directions at once; and the ground-plane fallback then
recovered most of the front/behind abstention band without depth at all.
The declined perception ablations bound where engineering can help: neither
a four-times-larger depth model nor two-view triangulation improves the
depth pair, the second 0.20 *worse* where it answers at all — the limit is
monocular ambiguity in the scenes, not model capacity. What is declined is a
lightweight uncalibrated estimator (two views, an assumed focal length), not
multi-view geometry in general; a calibrated stereo pair is the open route
§9.3 keeps, and §7.2 takes up what follows. Full derivations, calibration
evidence, audit samples and the failure structure of each refinement are in
**Appendix D**.

## 4.10 Failure gallery: every miss diagnosed

Each missed human triplet is diagnosed automatically by re-checking the
rule's individual conditions against the cached geometry and mask-contact
maps (`scripts/make_failure_gallery.py`; rendered examples in
`outputs/failure_gallery/`). Appendix D.7 breaks the 1,650 misses of the
shipped rule set down by predicate and cause. Genuine depth-ordering errors
remain 1–6% of front/behind misses, the support misses are threshold trades
on real contact evidence, and `near` misses have all but vanished. The
convention-inverted *share* grew to 45–51% not because those misses
increased but because the ground-plane fallback shrank the abstention share
around them, total misses falling from 2,107 to 1,650. Misses attributable
to avoidable tool error across all seven predicates: ~7%.

## 4.11 Detector-in-the-loop: full automation, attributed

The deployment mode replaces ground-truth boxes with Grounding DINO
(Liu et al., 2024) zero-shot detection (short-noun text prompts for the six
classes; threshold 0.25, tuned in one disclosed iteration on a 20-image
trial), then runs the identical SAM2 → depth → rules stack
(`scripts/run_sgdet.py`, scored by `eval/sgdet_eval.py` with class-matched
greedy IoU ≥ 0.5).

End-to-end triplet recall over 836 images is **0.38** against the PredCls
headline, and the decomposition attributes most of that gap. Zero-shot
detection recall spans 0.40 (cube) to 0.95 (human), and a triplet needs
*both* endpoints; **conditioned on both being detected the relation layer
scores 0.85 mean, matching PredCls** (lateral 0.96/0.98, near 1.00, support
0.83/0.77; front/behind 0.69/0.70, computed before the ground-plane fallback
shipped, so a floor, and detectable pairs skew towards well-separated
objects). Detection is the dominant term rather than a complete accounting,
since missed objects also change which pairs are presented and what the
masks look like, and the rules themselves are detector-agnostic.

Two caveats. Zero-shot open-vocabulary detection is the worst-case detector,
the trade §2.6 identifies, used because the authors' trained YOLOv10m
weights were not available; with those the end-to-end gap would largely
close. And the 20-image trial over-estimated detection quality, its scenes
coming from one annotator batch. The same attribution holds on two
out-of-domain clips, where every visible failure is again a detection
failure (Appendix E.4).

## 4.12 Temporal redundancy and stability under viewpoint change

The 884 released images are one continuous walk: pixel-matching them against
the 2,650-frame raw capture the supervising group later supplied identifies
them as frames 000000–000883, each annotator group a contiguous 100-frame
block (`group_0` = frames 0–99, and so on). The qualification first: a group
is simultaneously an annotator identity *and* a temporal block holding one
arrangement, so the held-out split of §4.1 is held out by scene as well as
by annotator. The annotator reading survives — an inverted front/behind
convention (§4.5) and a `near` label used by three groups in nine (§3.2) are
labelling behaviours no arrangement of furniture can produce — and the
confound runs favourably: 0.74 on held-out groups is generalisation to an
unseen annotator *and* an unseen arrangement.

The sequence also lets the pipeline's verdicts be checked against themselves
with no human labels: a relation fixed by geometry should survive the camera
moving, and one decided by a coin toss at a threshold should not. Frames
were segmented by content drift (§3.10), which compresses the released
frames 2.7× at τ = 10 and locates all eight known layout changes within five
frames, and each segment's predicates were propagated from its keyframe to
the rest (`eval/keyframe_propagation.py`): over the 802 pair-bearing frames
of §4.1, 568 keyframes, 234 propagated frames, 11,352 comparable pairs.
Appendix E.2 holds the per-predicate table, the sweeps, the matching rule
and the coverage figures.

Two results. Skipping frames costs nothing measurable: mean recall under
propagation is 0.832 against 0.829 per frame, at every threshold tested. And
the stability finding is not the expected one. Front/behind was the
predicted loser, on the assumption its errors are depth noise near the
boundary, exactly what a viewpoint change perturbs; instead it agrees with
itself 0.958 of the time, above `on`/`under` at 0.878 (mean 0.946), and
still 0.911 at 89× compression, where segment members are substantially
different views. A predicate recalling 0.648 of the human labels on these
frames while agreeing with itself at 0.958 is not making random errors: it
is making the same call repeatedly and disagreeing systematically, which
converges with A8 (a four-times-larger depth model changed nothing) and §4.5
(two groups labelled the pair oppositely). §7.2 develops what follows.

**Limits.** Consistency is not correctness — a systematically wrong rule is
perfectly stable too. What 0.958 rules out is only the depth-noise
explanation this section was written to test; the case that the criterion is
unshared rather than wrong rests on §4.5's measured inversion. The figures
are also an upper bound, only pairs matched between keyframe and frame
contributing, the ones whose objects moved least; and coverage thins as
segments grow, so aggressive compression leaves pairs uncovered rather than
mislabelled, which E.2 traces to box drift and not absent annotation, a
tracker being the straightforward remedy.

## 4.13 Would a vision-language model do this instead?

The three baselines of §4.2 are deliberately weak; the strong one is a
vision-language model, and if it annotates this dataset as well as the
geometric pipeline does, the pipeline is unnecessary. Thirty images,
stratified across all nine annotator groups, go to the model with the
ground-truth boxes drawn on and numbered, and it answers by index
(`scripts/run_vlm_pilot.py`) — the PredCls setting the pipeline is evaluated
in, so neither is scored on detection, with Chapter 3's definitions in the
prompt verbatim, without which the run would measure §2.5's reference-frame
ambiguity rather than accuracy. Two models were run, because the objection
to one is that a larger model would close the gap: `gemini-flash-latest`,
small and non-reasoning, and `gemini-3.1-pro-preview`, a reasoning model an
order of magnitude larger.

{{fig:rq1-with-vlm}} plots both against the pipeline on the same human
triplets; the per-predicate tables are in Appendix E.1. On recall both lose
everywhere: scaling moves the mean from 0.400 to 0.445, real but barely half
the pipeline's 0.834, and on the depth pair 0.24 against 0.65 puts the model
below the geometric method's known weak point — scaling the model does not
scale the ability being measured. Recall alone would be an unfair verdict,
because it rewards whoever asserts more: the pipeline makes 885 assertions
on the 374 judged pairs against 344 and 414. Restricted to those pairs,
where precision is defined, the column reverses and **both models are more
precise than the pipeline**, 0.419 and 0.389 against 0.347; they buy it with
silence, at a price steep enough that both lose F1 on every predicate, 0.397
and 0.405 against 0.485 pooled (Appendix E.1).

Most of the recall gap is that silence. Asked for every ordered pair, with
the dataset's own definitions and the same instruction to omit what it is
unsure of that the rules implement as abstention, the model never addressed
**171 of the 381 gold triplets**, 44.9%. Scored only on the pairs it did
judge, its recall is **0.686** rather than 0.378, and where the tool's
advantage looks largest the gap closes entirely: `to the left of` 0.909
against 0.918, `on` 0.864 against 0.860. What it is bad at is
*exhaustiveness*; where it speaks it is roughly as good as the geometry.
That is no defence of the model as an annotator — exhaustiveness is the
property this project exists to supply, and a source skipping 45% of the
pairs cannot deliver it — but the fair statement, the one this dissertation
should be held to, is that **the pipeline beats it on coverage and matches
it on judgement**.

The shape of the failure decides the reading. Neither model contradicts
itself or inverts the front/behind convention; they fall silent, and supply
one direction of a symmetric pair without the other in a third of cases —
the two defects §4.5 measures in the *human* annotation. Asked to annotate,
a capable vision-language model reproduces the characteristic failure of the
human process, not a geometric one. Its precision where it speaks still
begins a case for it as an adjudicator on the depth pair, which §7.6 takes
up; §4.14 asks the same family of model to judge claims it is handed rather
than to find them, only the half measured sound here. Appendix E.1 gives the
diagnostics and the limits of a thirty-image pilot.

## 4.14 Auditing the audit: blinding, decoys, and a second judge

Every precision figure so far rests on the author verdicting the author's
tool, the objection §2.9 raises and §7.4 concedes. More verdicts of the same
kind would only narrow an interval around a possibly biased centre, so this
section re-runs the audit with the three defects of §4.4 and §4.9 removed at
once.

**Design.** 242 items: 214 claims the tool emitted and **28 decoys**,
relations it did *not* emit and no annotator labelled, mixed in unmarked.
The decoys are the instrument: every item in the earlier sheets was a tool
assertion, so an auditor who simply agreed scored 100% and looked
calibrated, where here that auditor scores zero. Sampling, the class guard
and the separation of sheet from key are in Appendix E.7, with the pack's
full per-predicate table; its author column reappears as **v3** in the
comparison below. The same 242 images, same definitions, same instruction to
answer wrong when unsure, were put to `gemini-3.6-flash` as a second judge
independent of the author (`scripts/judge_audit_vlm.py`). A model may judge
what §4.13 shows it cannot annotate because the half that failed there was
*coverage* — 44.9% of gold triplets never addressed — while on the pairs it
did judge it was the more precise of the two, and judging a handed claim
asks only for that sound half. The decoys test this rather than assume it:
the model rejected 24 of 28 relations the tool never emitted against the
author's 19 of 28, the stricter of the two and not a judge that agrees with
whatever it is shown, and the two reach Cohen's κ 0.601 over all items and
0.425 over the claims alone — moderate agreement, not an echo. Neither is a
human; the independent human estimate is §4.15's.

**The two precision measurements point in opposite directions, and which way
is diagnostic.** Section 4.3 measured precision on the pairs a human
labelled; this audit measures it on the pairs a human did not. For five
predicates the first badly understates the second (`near` 0.11 against
1.000, the laterals 0.35 and 0.42 against 0.917 and 0.958) — the sparse-gold
artefact §4.1 anticipated. For support the relation inverts, sharply: 0.95
and 0.93 on annotated pairs against 0.372 and 0.431 off them, under the
pre-refit labels these audits used. The direction says what the human record
*is*: a lateral relation holds for nearly every ordered pair and the
annotators wrote down a handful, so their labels are a small sample of a
large truth and the tool's extras are mostly further instances of it;
support is rare and salient, a thing resting on a thing being worth
recording and recorded, so the human labels are close to the complete set of
easy cases and what the tool adds beyond them is mostly not there.
Restricted precision understates a predicate whose gold is a sample and
overstates one whose gold is nearly exhaustive, which is why the protocol
pairs it with an audit rather than reporting either alone.

The lateral, depth and proximity claims survive; support does not. At 0.404
it is less than half what §4.9 reported and outside any interval this
dissertation previously stated. The judges disagree on its level, 0.404
against 0.638, and agree emphatically on its direction, both far below 0.9
(raw agreement 0.814, κ 0.576). Most of the drop is the blinding, not the
sample: the same rules on the same data scored 0.77 unblinded (§4.9) and
0.404 blind, because every row of the earlier sheet was known to be a tool
emission, a prior the decoys remove.

**`near` is the least supported number in the audit, and carries the most
labels.** The judges agree on the laterals and the depth pair to within
0.042 and 0.083, and diverge on `near` by **0.375**: 1.000 against 0.625 on
the same 24 images; the author also accepted two of the four `near` decoys,
against the model's one. That combination sits under the predicate the tool
emits most freely: 43,388 ordered pairs against 717 in the human record. The
threshold generalised to a held-out annotator at recall 1.00 (§3.8), so the
*notion* is calibrated; what 24 samples cannot establish is that a rule
firing sixty times more often than the annotators did is right every time it
fires. The most the 1.000 supports is that no counter-example appeared in 24
draws. The decoys also show this is not an auditor being harsh — both judges
rejected **all eight** support decoys — and they measure an author bias
elsewhere: three of four `behind` decoys accepted against the model's one,
two of four for `in front of` and for `near`, a generosity confined to the
family §4.5 shows the annotators used inconsistently, reported and not
corrected because one instruction governed both judges.

The cause is a threshold fitted where its error was invisible. Sorted by the
contact fraction the rule fires on, audited claims below 0.85 are correct 1
time in 11 (4/44) and above it 2 times in 3 (34/50); the shipped
`on_contact_min` 0.60 came from Appendix D.2's fit on train F1 against gold
covering ~10% of ordered pairs, so a false positive on the other 90% cost
the fit nothing — which is why the plateau D.2 calls "uncritical" from 0.60
to 0.80 is flat. (A second independent signal exists, the supporting
object's size: `on(A, B)` requires B to hold A up, and a 20-pixel cube is
not a surface.)

The repair was fitted, shipped, and re-audited from scratch. A cut-off
chosen by inspecting these 94 verdicts and scored against them would be
optimistic by an unknown amount — the error that produced the 0.9 — so it
was fitted the way every threshold in Chapter 3 is fitted, on the 63 audited
claims from annotator groups 0–5, where precision rises steeply to 0.686 at
0.85 and flattens; on the 31 claims from groups 6–8 that no part of the fit
saw, it predicted **0.367 → 0.667**. `on_contact_min` was then set to 0.85
and every experiment in this dissertation re-run against the new labels.
Because a projection from held-out items selected under the *old* labels is
still an extrapolation, a second pack was drawn from the new emissions — 219
items, 191 claims and 28 decoys, same construction, same blinding, same two
judges — and audited independently. Below, **v3** is the pre-refit pack
drawn at `on_contact_min` 0.60, and **v4** this fresh one, drawn at the
shipped 0.85.

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

The middle column is the shipped tool; the 0.79–1.00 range quoted elsewhere
for the non-support predicates is `near` at its floor and `to the right of`
at its ceiling. The change is confirmed and the projection was optimistic:
both judges record a large improvement on an independent draw, and both
record it below the 0.667 the held-out fit predicted, held-out support
recall falling from 0.92 to 0.843 in exchange. Fitting on one audit and
validating on the next is as far as this design can go towards an unbiased
estimate, and the gap between the two numbers is the price of the
extrapolation the earlier draft quoted. The auditor also improved between
the two packs, which Appendix E.7 reads in full; neither that nor the
model's parallel movement rescues the support figure. At 0.535, the labels
the tool adds beyond the human record on this predicate are right about half
the time.

## 4.15 A disinterested check, against a human baseline

Both judges in §4.14 carry an objection: one built the tool, and the other
is a system §4.13 shows annotates poorly. Appendix E.3 specifies the arm
that answers the first, putting sampled claims to volunteers who did not
build the tool and are not shown what it predicted. It closed with 1,415
usable judgements from 20 raters over 832 of the 1,000 claims in the pool,
83.2%, no rater supplying more than 15% of them.

**The shipped support rule has still not been judged by anyone outside this
project.** The tool claims were drawn on 17 July, a month before
`on_contact_min` was re-fitted from 0.60 to 0.85, so every `on` and `under`
claim below comes from the superseded rule and belongs against §4.14's first
pack, not the shipped second one; for the five predicates the support
threshold does not touch, the labels are identical in both generations.

**What makes this arm readable is that it scores the annotators too.** Half
the pool is drawn from the human annotations rather than from the tool,
rendered through the identical pipeline, interleaved so a rater cannot tell
the two apart, and judged under the same instruction to answer WRONG when
unsure. Without that control a low score on the tool would be
uninterpretable. On the human-written claims the raters answer TRUE 0.940 of
the time (395/420), against 0.726 on the tool's (299/412): they are not
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

The denominators differ because the judges saw different draws from the same
pre-refit generation, not because items were dropped: the volunteers saw 126
support claims from the study pool, the two blind judges 94 from the audit
pack of §4.14, and 147 items across all seven predicates carry both a
volunteer and an author verdict. The columns are three estimates of one
quantity, not three verdicts on one sheet, which is what makes their
agreement worth reporting.

**On support the author's verdicts survive the check exactly**: 0.413 from
the volunteers against the author's 0.404 on the pre-refit labels, a
difference of 0.009 between the person who built the tool, blinded and
working against decoys, and strangers with no stake in the outcome — the
objection §2.9 raises and §7.4 concedes, answered by measurement. The model
sits apart at 0.638, and `near` shows the same shape from the other side,
0.923 and 1.000 from the two human judges against the model's 0.625. Across
all seven predicates the volunteers rank the tool almost exactly as the
author does, Spearman 0.96 against the model's 0.34, landing 0.074 away on
average. Two independent human judges agreeing to that degree is the
strongest evidence available that the audits were not theatre.

**Against the human baseline, five predicates hold and support does not.**
On the five the support threshold does not touch, the tool scores 0.864
against the annotators' 0.926, a gap of 0.063 (on `in front of` it is
marginally ahead). On support it scores 0.413 against 0.975, a gap of 0.563
whose intervals do not come close to overlapping — the same weakness §4.14
found and the refit responded to, now measured against what human annotation
scores on the identical instrument. One asymmetry limits the comparison: the
tool's claims are its *extra* predictions, on pairs the annotators passed
over, while the control claims are pairs they chose to record, so some of
every gap is the difficulty of the claim rather than the quality of the
label. For a 0.063 gap that reservation may account for most of it; for
0.563 it cannot.

**The author-bias check.** On the 147 claims carrying both a volunteer and
an author verdict the two agree 0.871 of the time, Cohen's κ 0.683;
crowd-internal reliability is Krippendorff's α 0.703 across the same claims.
Both are substantial on the conventional reading, and both roughly doubled
as the sample grew, which is what a real signal does and noise does not.
Neither approaches 1.0, and §2.3's account of spatial language predicts that
better than rater carelessness does: some of the residual disagreement is
over what the words mean, not over what the photograph shows.

This arm settles that the author's audit did not run in the author's favour,
and that the raters are not uniformly harsh, because they score human
annotation at 0.940 on the same instrument. It does not settle the shipped
tool, whose support rule postdates the sample, and it cannot fully separate
label quality from claim difficulty, the two arms being drawn from
populations the annotators themselves divided. §7.6 and §9.3 carry what
survives; re-running the arm on post-refit support labels is the one
outstanding item that would close it.
