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

Before submitting, fill the one remaining placeholder in main.tex:
    [INSERT ACKNOWLEDGEMENTS...]

The student ID is set, and the declaration's word count is computed at build
time from the chapter sources (front matter, references and appendices
excluded), so it cannot drift out of date the way a hand-typed figure does.
