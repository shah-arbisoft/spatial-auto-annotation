LaTeX build of the dissertation
==============================

GENERATED. Do not edit the .tex files: they are overwritten. The Markdown
chapters in dissertation/ are the source of truth. After editing them:

    python scripts/build_latex.py

To compile:

    latexmk -pdf main.tex          the dissertation

(run either twice if the ToC looks stale)

With no local TeX installation, upload this whole folder to Overleaf
(New Project > Upload Project) and compile there. It needs only packages
present in a standard TeX Live.

One document is submitted. The ethics record is bound into the dissertation
itself: Supplementary A carries it in full, and the countersigned Secondary
Data Checklist binds in there from dissertation/checklist_signed.pdf. That
file is the one item this build cannot generate and is untracked, because it
carries a signature and this repository is public. Supply it before the final
build, and sign the dissertation's declaration page.

The student ID is set, the acknowledgements are written, and the
declaration's word count is computed at build time from the chapter sources
(front matter, references and supplements excluded), so it cannot drift out
of date the way a hand-typed figure does.
