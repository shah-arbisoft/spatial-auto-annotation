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
        "appendices.md",
        "Fitting the \\texttt{near} threshold. Recall against all annotators "
        "and against the held-out annotator, with restricted precision. The "
        "fitted value is the tightest threshold at the plateau.",
        "after:### C.7"),
    "front_behind_decomposition.png": (
        "appendices.md",
        "Front/behind decomposed per annotator group: agreement where the "
        "tool commits, deliberate abstention, and the two groups that used "
        "the inverted direction convention.",
        "after:### C.6"),
    "video_stability.png": (
        "appendices.md",
        "Frame-to-frame stability of the emitted triplets on the two "
        "demonstration clips, before and after temporal smoothing.",
        "after:### E.4"),
    "planner_sources.png": (
        "appendices.md",
        "Scenes with a safe grasp plan, by the relation source given to the "
        "planner. Neither automatic source matches human annotation alone; "
        "their union does, because their failures are disjoint.",
        "after:### E.5"),
    "rq2_with_vlm.png": (
        "appendices.md",
        "Downstream recall against held-out human gold by label source. "
        "Every arm trains on the same pairs, so the label source is the only "
        "variable. Self-training improves on the human labels everywhere "
        "except \\texttt{near}, where it falls below them.",
        "after:### F.2"),
    "rq1_with_vlm.png": (
        "appendices.md",
        "Recall of the human triplets per predicate: the geometric pipeline "
        "against both vision-language models on the same 30 images, the same "
        "numbered boxes and the same written definitions. The dotted lines "
        "are the two means. Scaling the model lifts every predicate a little "
        "and closes none of the gap.",
        "after:### E.1"),
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
    "\u03c4": r"$\tau$", "\u2212": "-", "\u2260": r"$\neq$",
    "\u00b7": r"$\cdot$", "\u00b0": r"$^{\circ}$", "\u03b5": r"$\epsilon$",
    "\u2026": r"\ldots", "\u2011": "-", "\u00a0": "~", "\u2032": "'",
    "\u2033": "''",
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
    # {{fig:rq1-recall}} -> Figure~\ref{...}. Written this way in the Markdown
    # so a figure can be referred to by name rather than by a number that
    # moves whenever a float does; stashed with the code spans so esc() below
    # does not escape the backslash back out again.
    def stash_ref(m):
        spans.append(None)
        REFS[len(spans) - 1] = r"Figure~\ref{" + m.group(1) + "}"
        return f"\x00{len(spans) - 1}\x00"

    REFS: dict[int, str] = {}
    text = re.sub(r"\{\{(fig:[A-Za-z0-9_-]+)\}\}", stash_ref, text)
    text = esc(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\\textit{\1}", text)

    def pop(m):
        i = int(m.group(1))
        return REFS[i] if i in REFS else r"\texttt{" + esc(spans[i]) + "}"

    return re.sub(r"\x00(\d+)\x00", pop, text)


# Table captions, in order of appearance per source file. The template
# requires a List of Tables, and an uncaptioned tabular never reaches it, so
# every table needs an entry here or the list comes out empty.
TABLES = {
    "chapter1_introduction.md": [
        "Mapping of the CRISP-DM phases onto the chapters and artefacts of "
        "this project.",
    ],
    "chapter2_literature_review.md": [
        "Related work assessed against the requirements of a fully-automatic "
        "spatial-relationship annotator.",
    ],
    "chapter3_design.md": [
        "Pipeline stages, the alternative rejected at each stage, and the "
        "justification.",
    ],
    # One caption per table, in document order. The two Ablation A9 captions
    # that used to sit here moved with their tables to Appendix D.6; leaving
    # them behind shifted every caption from the ablation table onward.
    "chapter4_results_rq1.md": [
        # 4.2
        "Per-predicate recall of the human triplets, pooled and on held-out "
        "annotators, against three baselines.",
        # 4.3
        "Precision, recall and F1 restricted to the human-annotated pairs.",
        # 4.4
        "Manual audit of a stratified sample of extra predictions, with "
        "Wilson intervals.",
        # 4.5
        "Front/behind by annotator group: emission rate, agreement where the "
        "tool commits, and the effect of aligning the direction convention.",
        # 4.9
        "The nine ablations: what each tests, the setting that shipped, and "
        "the verdict on held-out annotators.",
        # 4.14
        "Stability of each predicate across viewpoints of the same scene, "
        "with recall under keyframe propagation against per-frame "
        "computation.",
        # 4.16
        "A vision-language model against the geometric pipeline on the same "
        "30 images, the same pairs and the same human gold: recall.",
    ],
    "chapter5_results_rq2.md": [
        "Downstream recall against held-out human gold for the three label "
        "sources.",
        "Planner experiment: whether the plan clears the occluding object "
        "before grasping the target, over 25 held-out scenes under each "
        "label source.",
    ],
    "chapter6_benchmark.md": [
        "Benchmark test results (SGDet) for the human-trained and "
        "auto-trained arms.",
        "Seed replication: which differences separate across three seeds "
        "per arm.",
    ],
    # Consumed in document order across the appendices, one per table.
    "appendices.md": [
        # B
        "Commands reproducing every experiment from the cached geometry, "
        "with run times.",
        # C.10
        "The seven predicates: core geometric test, shipped threshold values "
        "and the symmetry each rule guarantees by construction.",
        # D.6
        "Ablation A9: two-view triangulation against the monocular cascade "
        "on the same front/behind pairs.",
        # D.6
        "Ablation A9 by depth separation: multi-frame ordering accuracy is at "
        "chance where the two objects sit at nearly equal camera distance.",
        # D.7
        "For each missed human triplet, the predicates the tool emitted "
        "instead.",
        # D.7
        "Diagnosed cause of every missed human triplet, by predicate.",
        # E.1
        "The vision-language comparison of \\S4.16 restricted to the pairs "
        "carrying a human label, where precision is defined. The model is the "
        "more precise labeller and loses F1 on every predicate.",
        # F.1
        "Benchmark per-predicate mean recall at 100 for both arms, seed 42.",
        # F.1
        "Benchmark mean recall at 100 by test slice for both arms, seed 42. "
        "Section 6.3.1 replicates every row but the convention-aligned one "
        "across three seeds.",
        # F.3
        "Summary of the design decisions, the alternatives rejected and the "
        "reasons.",
        # F.4
        "A third label source in the benchmark: mean recall at 100 by test "
        "slice for the human, automatic and vision-language arms.",
        # F.5
        "The downstream experiment on four further indicators. The automatic "
        "arm leads on recall and trails on every other column, which is the "
        "sparse-gold artefact of \\S4.3 reappearing downstream rather than a "
        "verdict on label quality.",
    ],
}


# How many characters of \small text fit across the a4 text block, given that
# longtable adds 2*\tabcolsep of padding per column. A table under the budget
# can use natural-width columns, which look better; over it, the columns must
# wrap or the table runs off the page and drags its caption with it.
TEXT_PT = 452.0        # a4 width less 1in margins, in points
COLPAD_PT = 12.0       # 2 * \tabcolsep
CHAR_PT = 5.2          # mean glyph advance at \small in a 12pt document


def char_budget(ncol: int) -> float:
    return max(20.0, (TEXT_PT - COLPAD_PT * ncol) / CHAR_PT)


def is_numeric_col(body: list[list[str]], i: int) -> bool:
    """True when the column's body is dominated by numbers, so it reads
    better right-aligned."""
    vals = [r[i] for r in body if i < len(r) and r[i].strip()]
    if not vals:
        return False
    numish = sum(bool(re.fullmatch(r"[\d.,%/()+\-– ]*\*{0,2}[\d.,%/()+\-– ]*", v))
                 for v in vals)
    return numish >= 0.7 * len(vals)


def col_spec(cells: list[list[str]], ncol: int) -> str:
    """Column spec that fits the text block.

    Short tables keep natural widths, which look better. Wide ones get
    \\p{} columns in proportion to their longest cell, so the text wraps
    instead of overflowing the margin. Widths are expressed as fractions of
    \\textwidth and deliberately sum to less than 1 to leave room for the
    inter-column padding longtable adds."""
    body = cells[1:] or cells
    widest = [max((len(r[i]) for r in cells if i < len(r)), default=1)
              for i in range(ncol)]
    numeric = [is_numeric_col(body, i) for i in range(ncol)]

    if sum(widest) <= char_budget(ncol):
        if ncol <= 2:
            return "l" * ncol
        return "l" + "".join("r" if numeric[i] else "l" for i in range(1, ncol))

    # Give every column a floor so a narrow numeric column stays readable,
    # then share the rest out by how much text each actually holds.
    floor = 0.055
    share = [max(w, 4) for w in widest]
    total = sum(share)
    frac = [floor + (1.0 - floor * ncol) * (s / total) for s in share]
    scale = 0.94 / sum(frac)
    frac = [f * scale for f in frac]
    align = [r">{\raggedleft\arraybackslash}" if numeric[i]
             else r">{\raggedright\arraybackslash}" for i in range(ncol)]
    return "".join(f"{align[i]}p{{{frac[i]:.3f}\\textwidth}}"
                   for i in range(ncol))


def table(rows: list[str], caption: str = "") -> str:
    """A Markdown pipe table -> longtable (survives page breaks)."""
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [c for c in cells if not all(set(x) <= set("-: ") for x in c)]
    if not cells:
        return ""
    ncol = max(len(r) for r in cells)
    head, body = cells[0], cells[1:]
    head += [""] * (ncol - len(head))
    spec = col_spec(cells, ncol)
    out = [r"\begin{center}", r"\small",
           r"\begin{longtable}{" + spec + "}"]
    if caption:
        # longtable takes its caption as the first row; this is what puts the
        # table into \listoftables and gives it a number.
        out.append(r"\caption{" + inline(caption) + r"}\\")
    out += [r"\hline",
            " & ".join(inline(h) for h in head) + r" \\", r"\hline",
            r"\endfirsthead", r"\hline",
            " & ".join(inline(h) for h in head) + r" \\", r"\hline",
            r"\endhead"]
    for r in body:
        r += [""] * (ncol - len(r))
        out.append(" & ".join(inline(c) for c in r[:ncol]) + r" \\")
    out += [r"\hline", r"\end{longtable}", r"\end{center}"]
    return "\n".join(out)


def convert(md: str, figures: list[tuple[str, str, str]],
            captions: list[str] | None = None) -> str:
    """Markdown chapter body -> LaTeX (chapter heading handled by main.tex)."""
    lines = md.splitlines()
    out: list[str] = []
    i, in_code, buf_tbl, list_stack = 0, False, [], []
    caps = list(captions or [])
    n_tables = 0
    placed_figs: set[str] = set()

    def close_lists():
        while list_stack:
            out.append(r"\end{" + list_stack.pop() + "}")

    def flush_table():
        nonlocal buf_tbl, n_tables
        if buf_tbl:
            close_lists()
            cap = caps[n_tables] if n_tables < len(caps) else ""
            out.append(table(buf_tbl, cap))
            n_tables += 1
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
            # A figure anchored to this heading goes just before it, which
            # puts it at the end of the *previous* section: that is the right
            # place for a figure illustrating what was just reported. An
            # anchor written "after:## 4.16 ..." goes after the heading
            # instead, which is what a figure belonging to the final section
            # of a chapter needs, since there is no following heading to
            # anchor it to.
            # Prefix match, not equality: section titles get reworded, and an
            # anchor that silently stops matching used to drop the figure
            # from the PDF without a word (it dropped two).
            for fname, caption, anchor in figures:
                if anchor and not anchor.startswith("after:") \
                        and line.strip().startswith(anchor.strip()):
                    out.append(figure_block(fname, caption))
                    placed_figs.add(fname)
            out.append("\\" + cmd + "{" + inline(title) + "}")
            for fname, caption, anchor in figures:
                if anchor.startswith("after:") \
                        and line.strip().startswith(anchor[len("after:"):].strip()):
                    out.append(figure_block(fname, caption))
                    placed_figs.add(fname)
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

    # A figure declared for this chapter whose anchor never matched would
    # otherwise vanish from the PDF in silence. Append it and say so loudly.
    for fname, caption, _anchor in figures:
        if fname not in placed_figs:
            print(f"    WARNING: no anchor matched for {fname}; "
                  f"appended at the end of the chapter")
            out.append(figure_block(fname, caption))
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

    logo = ROOT / "assets" / "UniofSurrey.pdf"
    if logo.exists():
        shutil.copy(logo, out / "figures" / "UniofSurrey.pdf")
    else:
        print("  WARNING: assets/UniofSurrey.pdf missing; the title page "
              "will not compile")
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
        tex = convert(md, figs, TABLES.get(md_name))
        placed = sum(1 for f, _c, a in figs
                     if a and a.removeprefix("after:") in md)
        n_tab = tex.count(r"\begin{longtable}")
        n_cap = tex.count(r"\caption{") - sum(1 for f, _c, a in figs
                                              if (FIGDIR / f).exists())
        stem = Path(md_name).stem.replace("_", "-")
        (out / (stem + ".tex")).write_text(tex, encoding="utf-8")
        # n_cap == n_tab only tells us every table got *a* caption. A registry
        # holding more captions than the chapter has tables silently shifts
        # them all, which is how five of chapter 4's captions came to sit on
        # the wrong tables, so the count is checked explicitly.
        declared = len(TABLES.get(md_name) or [])
        if n_cap != n_tab:
            note = f"  WARNING: {n_tab - n_cap} UNCAPTIONED"
        elif declared != n_tab:
            note = (f"  WARNING: {declared} captions declared for {n_tab} "
                    f"tables; captions are positional and will be misaligned")
        else:
            note = ""
        print(f"  {md_name:34s} -> {stem}.tex  "
              f"({placed} figures, {n_tab} tables){note}")

    (out / "abstract.tex").write_text(
        convert((SRC / "abstract.md").read_text(encoding="utf-8"), []), encoding="utf-8")

    # Appendices: each "## Appendix X: Title" becomes its own \chapter so the
    # contents page reads "Appendix A: Ethical Approval", as the template shows.
    app_md = (SRC / "appendices.md").read_text(encoding="utf-8")
    parts = re.split(r"(?m)^##\s+Appendix\s+[A-Z]\s*[:.]?\s*(.+)$", app_md)
    chunks = []
    app_caps = list(TABLES.get("appendices.md", []))
    for title, body in zip(parts[1::2], parts[2::2]):
        # captions are consumed in document order across the appendices
        # count tables, not table rows: one caption is consumed per table, so
        # an appendix with two tables must take two.
        n_here = sum(1 for ln in body.splitlines()
                     if ln.strip().startswith("|---"))
        take, app_caps = app_caps[:n_here], app_caps[n_here:]
        # Figures owned by the appendices are anchored on a subsection
        # heading, so each belongs to whichever appendix chunk holds its
        # anchor; without this filter every appendix would get every figure.
        here = [(f, c, a) for f, (owner, c, a) in FIGURES.items()
                if owner == "appendices.md" and (FIGDIR / f).exists()
                and a and a.removeprefix("after:") in body]
        chunks.append("\\chapter{" + inline(title.strip()) + "}\n"
                      + convert(body, here, take))
    (out / "appendices.tex").write_text(
        "\n\n".join(chunks) if chunks else convert(app_md, []), encoding="utf-8")
    print(f"  appendices.md{'':21s} -> appendices.tex  ({len(chunks)} appendices)")
    (out / "references.tex").write_text(
        build_references((SRC / "references.md").read_text(encoding="utf-8")),
        encoding="utf-8")

    # The declaration's word count is computed here rather than typed in, so
    # it cannot drift as chapters are edited. Chapters plus abstract; front
    # matter, references and appendices excluded, and the editorial ">" notes
    # at the top of each chapter with them.
    words = len((SRC / "abstract.md").read_text(encoding="utf-8").split())
    for md_name, _title in CHAPTERS:
        body = (SRC / md_name).read_text(encoding="utf-8")
        words += len(re.sub(r"(?m)^>.*$", "", body).split())
    (out / "main.tex").write_text(
        MAIN.replace("__WORDCOUNT__", f"{words:,}"), encoding="utf-8")
    (out / "README.txt").write_text(README, encoding="utf-8")
    print(f"\nwrote {out}  (declared word count {words:,})")
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
\usepackage{mathptmx}   % Times, as the Word template specifies
\usepackage{array}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{textcomp}

% Any stray unicode the Markdown converter did not map would otherwise stop
% the build with "Unicode character not set up for use with LaTeX". These
% definitions make it print instead. The converter should catch them first;
% this is the net under it.
\DeclareUnicodeCharacter{03C4}{$\tau$}
\DeclareUnicodeCharacter{03B5}{$\varepsilon$}
\DeclareUnicodeCharacter{2212}{-}
\DeclareUnicodeCharacter{2260}{$\neq$}
\DeclareUnicodeCharacter{00B7}{$\cdot$}
\DeclareUnicodeCharacter{2192}{$\rightarrow$}
\DeclareUnicodeCharacter{2190}{$\leftarrow$}
\DeclareUnicodeCharacter{2500}{-}
\DeclareUnicodeCharacter{2502}{|}
\DeclareUnicodeCharacter{2026}{\dots}
\usepackage[hidelinks]{hyperref}

% The supplied Word template (COMM070 Data Science - Template.docx) sets its
% body text single-spaced in 12pt Times: 75 of its paragraphs carry
% w:line="240" w:lineRule="auto" explicitly, and its Normal style is
% "Times Roman" at w:sz 24. The LaTeX conversion shipped alongside it uses
% \onehalfspacing, which disagrees; the Word file is the template the module
% names, and the 60-page hard limit makes the difference material rather
% than cosmetic.
% Paragraphs are separated by a first-line indent rather than by vertical
% space. The template's own body paragraphs carry neither (its Normal style
% sets no space-after and no indent, and it separates paragraphs with the
% occasional empty one), so some choice has to be made; this is the one that
% costs no page space. At 320 body paragraphs a 0.6em skip was 2,304pt, or
% 3.3 pages of the 60 the module allows for the chapters.
\setlength{\parindent}{1.5em}
\setlength{\parskip}{0pt}
\singlespacing
\sloppy

% The template's contents page lists chapters as "CHAPTER 1: INTRODUCTION",
% so the chapter heading is formatted to match rather than the report
% class's default "Chapter 1 / Title" on two lines.
\titleformat{\chapter}[hang]
  {\normalfont\Large\bfseries\MakeUppercase}{\chaptertitlename\ \thechapter:}{0.5em}{}
\titlespacing*{\chapter}{0pt}{0pt}{20pt}

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
    \includegraphics[width=0.4\textwidth]{figures/UniofSurrey.pdf}\\[1.5cm]

    {\LARGE \textbf{A Fully-Automatic Spatial-Relationship\\[0.3cm]
     Annotation Pipeline for Robot-Acquired Images,\\[0.3cm]
     Validated Against Human Annotation}}\\[2cm]

    {\Large \textbf{School of Computer Science and Electronic Engineering}}\\[0.5cm]
    {\Large \textbf{MSc Data Science}}\\[0.5cm]
    {\large Academic Year 2025--2026}\\[2cm]

    {\large A project report submitted by: \textbf{Shah Hussain}}\\[0.3cm]
    {\large Student ID: \textbf{6949963}}\\[0.5cm]
    {\large A project supervised by: \textbf{Dr Peng Wang}}\\[1.5cm]

    {\large A report submitted in partial fulfilment of the requirement\\
    for the degree of Master of Science}\\[1.5cm]

    University of Surrey\\
    School of Computer Science and Electronic Engineering\\
    Guildford, Surrey GU2 7XH, United Kingdom\\
    Tel: +44 (0)1483 300800
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

I thank my supervisor, Dr Peng Wang, for the dataset this project is built
on, for the pointers to the annotation and scene-graph tooling that shaped
its early direction, and for steady guidance throughout. I am grateful to
his research group both for collecting and releasing the dataset and for
supplying the full robot capture from which the released images were cut,
which made the scale and viewpoint-stability measurements possible.

I thank the volunteers who gave their time to judge sampled predictions in
the validation study, without which the precision estimates would have
rested on my own verdicts alone.

Finally, I thank my family and friends for their patience and support over
the course of this work.

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
\noindent \textbf{Total Number of Words:} __WORDCOUNT__

\cleardoublepage

%==================== Contents ====================
\tableofcontents
\listoftables
\listoffigures

%==================== Abbreviations ====================
% The template's LaTeX variant offers a list of abbreviations; this document
% uses enough metric and setting names to warrant one.
\chapter*{List of Abbreviations}
\addcontentsline{toc}{chapter}{List of Abbreviations}
\begin{description}[leftmargin=6em, style=nextline, font=\normalfont\bfseries]
  \item[CRISP-DM] Cross-Industry Standard Process for Data Mining
  \item[EXIF] Exchangeable Image File Format (carries image orientation)
  \item[IoU] Intersection over Union
  \item[mAP] mean Average Precision
  \item[mR@K] mean Recall at K, averaged per predicate
  \item[PredCls] Predicate Classification (ground-truth boxes and classes given)
  \item[RQ] Research Question
  \item[R@K] Recall at K, pooled over predicates
  \item[SAM2] Segment Anything Model 2
  \item[SGDet] Scene Graph Detection (detection, classification and relations end to end)
  \item[SGG] Scene Graph Generation
  \item[VG] Visual Genome (the annotation format this dataset inherits)
  \item[YOLO] You Only Look Once (single-stage object detector family)
  \item[zR@K] zero-shot Recall at K, over triplet types unseen in training
\end{description}
\cleardoublepage

\pagenumbering{arabic}

%==================== Chapters ====================
\chapter{Introduction}
\input{chapter1-introduction}

\chapter{Literature Review}
\input{chapter2-literature-review}

\chapter{Research Methodology and Design}
\input{chapter3-design}

\chapter{Fidelity against the Human Annotations (RQ1)}
\input{chapter4-results-rq1}

\chapter{Downstream Utility (RQ2)}
\input{chapter5-results-rq2}

\chapter{The Direct Benchmark Test}
\input{chapter6-benchmark}

\chapter{Critical Evaluation}
\input{chapter7-critical-evaluation}

\chapter{Legal, Social, Ethical and Professional Considerations}
\input{chapter8-lsep}

\chapter{Conclusions and Future Work}
\input{chapter9-conclusions}

%==================== References ====================
\chapter*{References}
\addcontentsline{toc}{chapter}{References}
\input{references}

%==================== Appendices ====================
% Named as the template's contents page requires: Appendix A is the ethics
% record, later appendices are supplementary material.
\appendix
\renewcommand{\chaptertitlename}{Appendix}
% Appendix subsections carry their own labels (C.1, D.1, E.1 ...) inside the
% heading text, so LaTeX must not number them a second time; without this a
% heading reads "C.0.1 C.1 Notation ...". secnumdepth 0 keeps the chapter
% numbering the contents page needs (Appendix A, B, C ...) and drops the rest.
\setcounter{secnumdepth}{0}
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

Before submitting, attach the signed ethics self-assessment to Appendix A
(the one item this build cannot generate) and sign the declaration page.

The student ID is set, the acknowledgements are written, and the
declaration's word count is computed at build time from the chapter sources
(front matter, references and appendices excluded), so it cannot drift out
of date the way a hand-typed figure does.
"""


if __name__ == "__main__":
    main()
