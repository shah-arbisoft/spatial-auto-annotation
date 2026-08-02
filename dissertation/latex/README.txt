LaTeX build of the dissertation
==============================

GENERATED. Do not edit the .tex files: they are overwritten. The Markdown
chapters in dissertation/ are the source of truth. After editing them:

    python scripts/build_latex.py

To compile:

    latexmk -pdf main.tex        (run twice if the ToC looks stale)

With no local TeX installation, upload this whole folder to Overleaf
(New Project > Upload Project) and compile there. It needs only packages
present in a standard TeX Live.

Before submitting, attach the signed ethics self-assessment to Appendix A
(the one item this build cannot generate) and sign the declaration page.

The student ID is set, the acknowledgements are written, and the
declaration's word count is computed at build time from the chapter sources
(front matter, references and appendices excluded), so it cannot drift out
of date the way a hand-typed figure does.
