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

Before submitting, fill the three placeholders in main.tex:
    [INSERT STUDENT ID]
    [INSERT ACKNOWLEDGEMENTS...]
    [INSERT WORD COUNT]

Word count for the declaration (chapters only, excluding front matter,
references and appendices):

    python -c "import glob,pathlib; print(sum(len(pathlib.Path(f).read_text(encoding='utf-8').split()) for f in sorted(glob.glob('dissertation/chapter*.md'))))"
