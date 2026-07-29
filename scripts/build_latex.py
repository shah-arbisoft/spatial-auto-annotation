"""Assemble the dissertation into a LaTeX source tree.

The chapters are written in Markdown so they stay diffable and greppable and
every number in them can be checked against the artefacts by script. The
submitted artefact is a typeset PDF. This converts one into the other, so
the Markdown remains the single source of truth and the LaTeX is generated,
never hand-edited.

    python scripts/build_latex.py            # -> dissertation/latex/
    cd dissertation/latex && latexmk -pdf main.tex

If no LaTeX is installed locally, upload the generated folder to Overleaf
and compile there; it needs no packages beyond a standard TeX Live.

In-text citations are left as the author-date text already written, and the
reference list is typeset as a formatted list rather than through BibTeX.
The references are correct Harvard as written, and round-tripping them
through .bib would risk changing them.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "dissertation"
OUT = SRC / "latex"
FIGDIR = ROOT / "outputs" / "figures"

CHAPTERS = [
    ("chapter1_introduction.md", "Introduction"),
    ("chapter2_literature_review.md", "Literature Review"),
    ("chapter3_design.md", "Research Methodology and Design"),
    ("chapter4_results_rq1.md", "Fidelity against the Human Annotations"),
    ("chapter5_results_rq2.md", "Downstream Utility"),
    ("chapter6_benchmark.md", "The Direct Benchmark Test"),
    ("chapter7_critical_evaluation.md", "Critical Evaluation"),
    ("chapter8_lsep.md", "Legal, Social, Ethical and Professional Considerations"),
    ("chapter9_conclusions.md", "Conclusions and Future Work"),
]

# figure -> (chapter file it belongs in, caption, anchor text to insert after)
FIGURES = {
    "rq1_recall.png": (
        "chapter4_results_rq1.md",
        "Recall of the human triplets per predicate: the full pipeline against "
        "a box-only ablation and a random baseline. Front and behind have no "
        "box-only bar because that ablation cannot compute depth.",
        "## 4.3 Precision on the annotated pairs"),
    "near_T_sweep.png": (
        "chapter4_results_rq1.md",
        "Fitting the \\texttt{near} threshold. Recall against all annotators "
        "and against the held-out annotator, with restricted precision. The "
        "fitted value is the tightest threshold at the plateau.",
        "## 4.4 Manual audit of extra predictions (true-precision estimate)"),
    "front_behind_decomposition.png": (
        "chapter4_results_rq1.md",
        "Front/behind decomposed per annotator group: agreement where the "
        "tool commits, deliberate abstention, and the two groups that used "
        "the inverted direction convention.",
        "## 4.6 The tenth annotator"),
    "video_stability.png": (
        "chapter4_results_rq1.md",
        "Frame-to-frame stability of the emitted triplets on the two "
        "demonstration clips, before and after temporal smoothing.",
        "## 4.13 Independent validation of the precision estimate"),
    "rq2_comparison.png": (
        "chapter5_results_rq2.md",
        "Downstream recall against held-out human gold for the three label "
        "sources. Self-training improves on the human labels everywhere "
        "except \\texttt{near}, where it falls below them.",
        "## 5.3 Why self-training does not rescue the human labels"),
    "sgg_training_curves.png": (
        "chapter6_benchmark.md",
        "Validation curves for both benchmark arms, each against its own "
        "training-source labels, so only the shapes are comparable. The "
        "human arm peaks early and declines; the automatic arm does not.",
        "## 6.3 Test results"),
}

SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}"}
UNICODE = {
    "\u2013": "--", "\u2014": "---", "\u2018": "`", "\u2019": "'",
    "\u201c": "``", "\u201d": "''", "\u00a7": r"\S", "\u2248": r"$\approx$",
    "\u2264": r"$\leq$", "\u2265": r"$\geq$", "\u00d7": r"$\times$",
    "\u2192": r"$\rightarrow$", "\u2190": r"$\leftarrow$", "\u00b1": r"$\pm$",
    "\u00b0": r"$^{\circ}$", "\u03b5": r"$\epsilon$", "\u2026": r"\ldots",
    "\u2011": "-", "\u00a0": "~", "\u2032": "'", "\u2033": "''",
}


def esc(text: str) -> str:
    """Escape LaTeX specials in plain prose (not inside verbatim)."""
    out = []
    for ch in text:
        if ch in SPECIAL:
            out.append(SPECIAL[ch])
        elif ch in UNICODE:
            out.append(UNICODE[ch])
        elif ch == "\\":
            out.append(r"\textbackslash{}")
        elif ch in "~^":
            out.append(r"\textasciitilde{}" if ch == "~" else r"\textasciicircum{}")
        else:
            out.append(ch)
    return "".join(out)


def inline(text: str) -> str:
    """Markdown inline markup -> LaTeX, protecting code spans from escaping."""
    spans: list[str] = []

    def stash(m):
        spans.append(m.group(1))
        return f"\x00{len(spans) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)      # links -> text
    text = esc(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\\textit{\1}", text)

    def pop(m):
        return r"\texttt{" + esc(spans[int(m.group(1))]) + "}"

    return re.sub(r"\x00(\d+)\x00", pop, text)


def table(rows: list[str]) -> str:
    """A Markdown pipe table -> longtable (survives page breaks)."""
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [c for c in cells if not all(set(x) <= set("-: ") for x in c)]
    if not cells:
        return ""
    ncol = max(len(r) for r in cells)
    head, body = cells[0], cells[1:]
    head += [""] * (ncol - len(head))
    spec = "l" + "r" * (ncol - 1) if ncol > 2 else "l" * ncol
    out = [r"\begin{center}", r"\small",
           r"\begin{longtable}{" + spec + "}", r"\hline",
           " & ".join(inline(h) for h in head) + r" \\", r"\hline",
           r"\endfirsthead", r"\hline",
           " & ".join(inline(h) for h in head) + r" \\", r"\hline",
           r"\endhead"]
    for r in body:
        r += [""] * (ncol - len(r))
        out.append(" & ".join(inline(c) for c in r[:ncol]) + r" \\")
    out += [r"\hline", r"\end{longtable}", r"\end{center}"]
    return "\n".join(out)


def convert(md: str, figures: list[tuple[str, str, str]]) -> str:
    """Markdown chapter body -> LaTeX (chapter heading handled by main.tex)."""
    lines = md.splitlines()
    out: list[str] = []
    i, in_code, buf_tbl, list_stack = 0, False, [], []

    def close_lists():
        while list_stack:
            out.append(r"\end{" + list_stack.pop() + "}")

    def flush_table():
        nonlocal buf_tbl
        if buf_tbl:
            close_lists()
            out.append(table(buf_tbl))
            buf_tbl = []

    while i < len(lines):
        line = lines[i]

        if line.strip().startswith("```"):
            flush_table()
            close_lists()
            in_code = not in_code
            out.append(r"\begin{verbatim}" if in_code else r"\end{verbatim}")
            i += 1
            continue
        if in_code:
            out.append(line)
            i += 1
            continue

        if line.lstrip().startswith("|") and line.count("|") >= 2:
            buf_tbl.append(line)
            i += 1
            continue
        flush_table()

        if line.startswith("# "):                    # chapter title: main.tex owns it
            i += 1
            continue
        m = re.match(r"^(#{2,4})\s+(?:\d+(?:\.\d+)*)?\s*(.*)$", line)
        if m:
            close_lists()
            depth = len(m.group(1))
            cmd = {2: "section", 3: "subsection", 4: "subsubsection"}[min(depth, 4)]
            title = m.group(2).strip()
            # any figure anchored to this heading goes just before it
            for fname, caption, anchor in figures:
                if anchor and line.strip() == anchor.strip():
                    out.append(figure_block(fname, caption))
            out.append("\\" + cmd + "{" + inline(title) + "}")
            i += 1
            continue

        if line.startswith(">"):
            close_lists()
            quote = []
            while i < len(lines) and lines[i].startswith(">"):
                quote.append(lines[i].lstrip("> ").rstrip())
                i += 1
            out += [r"\begin{quote}", inline(" ".join(quote)), r"\end{quote}"]
            continue

        mb = re.match(r"^(\s*)([-*])\s+(.*)$", line)
        mn = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)
        if mb or mn:
            want = "itemize" if mb else "enumerate"
            if not list_stack or list_stack[-1] != want:
                close_lists()
                out.append(r"\begin{" + want + "}")
                list_stack.append(want)
            out.append(r"\item " + inline((mb or mn).group(3)))
            i += 1
            continue

        if not line.strip():
            close_lists()
            out.append("")
            i += 1
            continue

        # Gather the whole paragraph before converting. Markdown wraps lines,
        # so **bold** and `code` routinely straddle a newline; converting line
        # by line leaves the markers behind.
        para = []
        while i < len(lines):
            nxt = lines[i]
            if (not nxt.strip() or nxt.startswith(("#", ">", "|"))
                    or nxt.strip().startswith("```")
                    or re.match(r"^\s*([-*]|\d+\.)\s+", nxt)):
                break
            para.append(nxt.strip())
            i += 1
        out.append(inline(" ".join(para)))

    flush_table()
    close_lists()
    if in_code:
        out.append(r"\end{verbatim}")
    return "\n".join(out)


def tex_figname(fname: str) -> str:
    """Underscores in an \\includegraphics path are a common LaTeX failure;
    the copies are written with hyphens instead."""
    return Path(fname).stem.replace("_", "-") + Path(fname).suffix


def figure_block(fname: str, caption: str) -> str:
    safe = tex_figname(fname)
    return "\n".join([
        r"\begin{figure}[htbp]", r"\centering",
        r"\includegraphics[width=0.95\textwidth]{figures/" + safe + "}",
        r"\caption{" + caption + "}",
        r"\label{fig:" + Path(safe).stem + "}",
        r"\end{figure}", ""])


def build_references(md: str) -> str:
    entries = [b.strip() for b in re.split(r"\n\s*\n", md) if b.strip()
               and not b.strip().startswith("#")]
    out = [r"\begin{description}[leftmargin=2em, labelindent=0em]"]
    for e in entries:
        out.append(r"\item[] " + inline(" ".join(e.split())))
    out.append(r"\end{description}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    for fname in FIGURES:
        src = FIGDIR / fname
        if src.exists():
            shutil.copy(src, out / "figures" / tex_figname(fname))
        else:
            print(f"  WARNING: figure missing, skipped: {fname}")

    for md_name, _title in CHAPTERS:
        md = (SRC / md_name).read_text(encoding="utf-8")
        figs = [(f, c, a) for f, (owner, c, a) in FIGURES.items()
                if owner == md_name and (FIGDIR / f).exists()]
        tex = convert(md, figs)
        placed = sum(1 for f, _c, a in figs if a and a in md)
        (out / (Path(md_name).stem + ".tex")).write_text(tex, encoding="utf-8")
        print(f"  {md_name:34s} -> {Path(md_name).stem}.tex  ({placed} figures)")

    for name, src_name in (("abstract", "abstract.md"), ("appendices", "appendices.md")):
        (out / f"{name}.tex").write_text(
            convert((SRC / src_name).read_text(encoding="utf-8"), []), encoding="utf-8")
    (out / "references.tex").write_text(
        build_references((SRC / "references.md").read_text(encoding="utf-8")),
        encoding="utf-8")

    (out / "main.tex").write_text(MAIN, encoding="utf-8")
    (out / "README.txt").write_text(README, encoding="utf-8")
    print(f"\nwrote {out}")
    print("compile:  cd dissertation/latex && latexmk -pdf main.tex")
    print("or upload the folder to Overleaf and compile there.")


MAIN = r"""% GENERATED by scripts/build_latex.py -- do not edit by hand.
% The Markdown chapters in dissertation/ are the source of truth; rebuild
% with `python scripts/build_latex.py` after editing them.
\documentclass[12pt]{report}
\usepackage[a4paper, margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{setspace}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{caption}
\usepackage{longtable}
\usepackage{tocloft}
\usepackage{enumitem}
\usepackage{amsmath}
\usepackage[hidelinks]{hyperref}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0.8em}
\onehalfspacing
\sloppy

\pagestyle{fancy}
\fancyhf{}
\rhead{\small A Fully-Automatic Spatial-Relationship Annotation Pipeline}
\cfoot{\thepage}
\renewcommand{\headrulewidth}{0.4pt}

\begin{document}

%==================== Title page ====================
\begin{titlepage}
\begin{center}
    \vspace*{1cm}
    {\LARGE \textbf{A Fully-Automatic Spatial-Relationship\\[0.3cm]
     Annotation Pipeline for Robot-Acquired Images,\\[0.3cm]
     Validated Against Human Annotation}}\\[2cm]

    {\Large \textbf{School of Computer Science and Electronic Engineering}}\\[0.5cm]
    {\Large \textbf{MSc Data Science}}\\[0.5cm]
    {\large Academic Year 2025--2026}\\[2cm]

    {\large A project report submitted by: \textbf{Shah Hussain}}\\[0.3cm]
    {\large Student ID: \textbf{[INSERT STUDENT ID]}}\\[0.5cm]
    {\large A project supervised by: \textbf{Dr Peng Wang}}\\[1.5cm]

    {\large A report submitted in partial fulfilment of the requirement\\
    for the degree of Master of Science}\\[1.5cm]

    University of Surrey\\
    School of Computer Science and Electronic Engineering\\
    Guildford, Surrey GU2 7XH, United Kingdom
    \vfill
\end{center}
\end{titlepage}

\pagenumbering{roman}

%==================== Abstract ====================
\chapter*{Abstract}
\addcontentsline{toc}{chapter}{Abstract}
\input{abstract}

%==================== Highlights ====================
\chapter*{Highlights}
\addcontentsline{toc}{chapter}{Highlights}
\begin{itemize}
  \item First fully-automatic annotator for a robot scene-graph dataset's seven spatial predicates
  \item Geometric rules compute every relation; thresholds fitted and validated on held-out annotators
  \item Recovers 85\% of human labels; audited precision near 1.0 on five of seven predicates
  \item Model trained on automatic labels beats its human-label twin 0.76 versus 0.30 on held-out gold
  \item Quantifies two annotation defects: inverted front/behind conventions and selective \texttt{near} usage
  \item Annotates the full dataset in five minutes on a 6\,GB consumer GPU at 20$\times$ human label density
\end{itemize}

%==================== Acknowledgements + declaration ====================
\chapter*{Acknowledgements}
\addcontentsline{toc}{chapter}{Acknowledgements}

[INSERT ACKNOWLEDGEMENTS: supervisor Dr Peng Wang for the dataset and
guidance; family and friends; the volunteers who contributed judgements to
the validation study.]

\vspace{1.5cm}
\noindent\textbf{Declaration of Originality}

\noindent I confirm that the submitted work is my own work. No element has
been previously submitted for assessment, or where it has, it has been
correctly referenced. I have clearly identified and fully acknowledged all
material that should be attributed to others (whether published or
unpublished, and including any content generated by a deep learning /
artificial intelligence tool), and have also included their source references
where relevant, using the referencing system required by my course. I agree
that the University may submit my work to means of checking this, such as the
plagiarism detection service Turnitin\textregistered{} UK. I confirm that I
understand that assessed work that has been shown to have been plagiarised
will be penalised.

\vspace{1cm}
\noindent Signature: \rule{6cm}{0.4pt} \hfill Date: \rule{4cm}{0.4pt}

\vspace{1cm}
\noindent \textbf{Total Number of Words:} [INSERT WORD COUNT]

\cleardoublepage

%==================== Contents ====================
\tableofcontents
\listoftables
\listoffigures
\cleardoublepage

\pagenumbering{arabic}

%==================== Chapters ====================
\chapter{Introduction}
\input{chapter1_introduction}

\chapter{Literature Review}
\input{chapter2_literature_review}

\chapter{Research Methodology and Design}
\input{chapter3_design}

\chapter{Fidelity against the Human Annotations (RQ1)}
\input{chapter4_results_rq1}

\chapter{Downstream Utility (RQ2)}
\input{chapter5_results_rq2}

\chapter{The Direct Benchmark Test}
\input{chapter6_benchmark}

\chapter{Critical Evaluation}
\input{chapter7_critical_evaluation}

\chapter{Legal, Social, Ethical and Professional Considerations}
\input{chapter8_lsep}

\chapter{Conclusions and Future Work}
\input{chapter9_conclusions}

%==================== References ====================
\chapter*{References}
\addcontentsline{toc}{chapter}{References}
\input{references}

%==================== Appendices ====================
\appendix
\chapter{Appendices}
\input{appendices}

\end{document}
"""

README = """LaTeX build of the dissertation
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
"""


if __name__ == "__main__":
    main()
