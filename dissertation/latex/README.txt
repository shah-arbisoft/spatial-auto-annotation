LaTeX build of the dissertation
==============================

GENERATED. Do not edit the .tex files: they are overwritten. The Markdown
chapters in dissertation/ are the source of truth. After editing them:

    python scripts/build_latex.py

To compile:

    latexmk -pdf main.tex

Run it twice if the table of contents looks stale.

With no local TeX installation, upload this whole folder to Overleaf
(New Project > Upload Project) and compile there. It needs only packages
present in a standard TeX Live.

The ethics record is bound into the dissertation rather than submitted beside
it: Supplementary A carries it in full, and the countersigned Secondary Data
Checklist binds in there. That checklist is the one input this build cannot
generate. It carries a signature and this repository is public, so it is
untracked and is not in this folder; the build prints a note in its place and
is otherwise complete. Where dissertation/checklist_signed.pdf is present,
scripts/build_latex.py copies it into figures/ and binds it in.

The declaration's word count is computed at build time from the chapter
sources, excluding front matter, references and supplements, so it cannot
drift out of date the way a hand-typed figure does.
