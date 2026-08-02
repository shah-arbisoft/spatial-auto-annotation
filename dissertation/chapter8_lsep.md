# Chapter 8: Legal, Social, Ethical and Professional Considerations

This chapter assesses the legal, social, ethical and professional (LSEP)
dimensions of building and releasing an automatic annotator: §8.1 the legal
position of the data, the models and the outputs; §8.2 the social
implications of automating annotation work and of the robot systems it
serves; §8.3 the ethical safeguards, including those for the human
validation study; and §8.4 the professional standards the project holds
itself to. The detailed ethics record, including the participant-facing
material, is in Appendix A.

## 8.1 Legal considerations

**Data licensing and attribution.** The source dataset is published under
CC-BY 4.0 (Wang et al., 2025), which permits reuse and derivative works with
attribution. Attribution is given throughout this dissertation, on the
validation website, and in the released repository; the automatic
annotations are a derivative work distributed under the same attribution
terms.

**Model licences as a design constraint.** Licensing was treated as an
engineering requirement, not an afterthought, and in one case it shaped the
pipeline: Depth Anything v2 is released under Apache-2.0 only in its Small
variant, while the larger variants carry a non-commercial licence, so the
Small variant was preferred, a choice later justified independently on
accuracy grounds by ablation A8 (Appendix D.5). SAM2 and Grounding DINO are
Apache-2.0. The YOLO training used for the benchmark detector relies on the
`ultralytics` library, which is AGPL-3.0; this is compatible with academic
research use, but a commercial deployment of that component would need a
licence review, and the note is recorded here so a future user does not
inherit the question unknowingly.

**Data protection.** Some dataset frames contain identifiable people, which
makes them personal data within the meaning of UK data-protection law (Data
Protection Act 2018). This project processes them for research purposes
under the terms of their public release, and applies data minimisation to
everything it republishes: faces are anonymised in all published figures and
in every image on the validation website, and items whose judgement would be
compromised by anonymisation are removed rather than shown. The validation
study itself collects no personal data at all (§8.3). No unauthorised access
to systems or data occurs anywhere in the project, so the Computer Misuse
Act 1990 is noted only for completeness.

## 8.2 Social considerations

**Automation of annotation work.** The project automates a task that is
currently paid human work. The honest framing is the one the measurements
support: at this dataset's scale the manual process produced sparse and
inconsistent labels at a cost of nine annotators, and the realistic social
effect of automation here is not the displacement of a profession but a
change in the human role, from labelling every pair to reviewing a
measurable flagged minority (§4.7). The same shift makes dataset
construction affordable for groups that could not fund manual annotation at
all; the entire pipeline runs on a consumer 6 GB GPU.

**Downstream robot behaviour.** Annotation quality propagates: a robot
planner consuming wrong spatial relations can act wrongly in physical space.
This is precisely why the project's evaluation centres on audited precision,
abstention instead of guessing, and per-failure attribution, and why
Chapter 6 distinguishes labels that are *correct* from labels that are
*human-like*. Trust in automated labels should be calibrated by exactly the
kind of evidence this dissertation supplies, not assumed.

**Bias.** The geometric rules themselves carry no demographic component:
they compute from positions and extents. The perception stack, however, is
learned, and detection quality for the `human` class cannot be assumed
uniform across people; published detectors have documented performance
disparities. In this dataset the people are members of the collecting
research group, so the question is not testable here, and it is recorded as
a deployment consideration rather than a resolved issue.

## 8.3 Ethical considerations

The project's ethical surface has three parts, each with a concrete
safeguard. First, secondary use of images containing identifiable people:
handled by face anonymisation in everything republished, item removal where
anonymisation would bias a judgement, and `noindex` on the validation site.
Second, the human validation study: participation is anonymous, voluntary
and brief; no names, contact details, IP addresses or tracking identifiers
are collected; an information panel states the purpose and data handling
before play; and the collection runs under the University's ethics
self-assessment process (Appendix A). Third, research integrity in
reporting: predictions were registered before the benchmark run and two of
the three are reported as refuted (§6.6); a withdrawn hypothesis remains in
the text (§6.4); and two built refinements are reported as measured and
declined (Appendix D.4). The dissertation treats honest negative results as
results.

## 8.4 Professional considerations

The work is conducted to the standards of the BCS Code of Conduct (BCS,
2022): public interest (privacy safeguards above), professional competence
and integrity (claims bounded by measurements; limitations stated in §7.6),
and duty to the profession (methods and failures documented so others can
build on both). Professional practice in the engineering itself: version
control with a clean history, a unit-and-invariant test suite run before
every change ships, every threshold in one configuration file, seeded and
reproducible runs (Appendix B), and licence-compliant use of third-party
models and data. The reproducibility package is treated as a deliverable
with the same status as the results, because a validation study that cannot
be re-run is an anecdote.
