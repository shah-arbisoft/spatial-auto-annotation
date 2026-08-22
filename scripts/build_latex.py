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
    "qualitative_examples.png": (
        "chapter4_results_rq1.md",
        "Two scenes as the pipeline labels them, with the detected boxes and "
        "class instances drawn. Left, a calibration group on which every emitted "
        "relation is correct. Right, a held-out group whose annotators recorded "
        "front and behind under the opposite convention, which is the failure "
        "Section 4.5 decomposes. Faces are pixelated from the dataset's own "
        "human boxes, as Section 8.1 requires of anything republished.",
        "## 4.2 Headline: recall of the human triplets"),
    "rq1_recall.png": (
        "chapter4_results_rq1.md",
        "Recall of the human triplets per predicate: the full pipeline against "
        "a box-only ablation and a random baseline. Front and behind have no "
        "box-only bar because that ablation cannot compute depth.",
        "## 4.3 Precision on the annotated pairs"),
    "near_T_sweep.png": (
        "chapter3_design.md",
        "Fitting the \\texttt{near} threshold. Recall against all annotators "
        "and against the held-out annotator, with restricted precision. The "
        "fitted value is the tightest threshold at the plateau.",
        "## 3.9 Modularity"),
    "front_behind_decomposition.png": (
        "chapter4_results_rq1.md",
        "Front/behind decomposed per annotator group: agreement where the "
        "tool commits, deliberate abstention, and the two groups that used "
        "the inverted direction convention.",
        "## 4.6 The tenth annotator"),
    "video_stability.png": (
        "appendices.md",
        "Frame-to-frame stability of the emitted triplets on the two "
        "demonstration clips, before and after temporal smoothing.",
        "after:### E.4"),
    "planner_sources.png": (
        "chapter5_results_rq2.md",
        "Scenes with a safe grasp plan, by the relation source given to the "
        "planner. Neither automatic source matches human annotation alone, and "
        "under the shipped support threshold the vision-language source is "
        "marginally ahead of the geometric one; their union matches it, "
        "because their failures are disjoint.",
        "## 5.8 Answer to RQ2"),
    "rq2_with_vlm.png": (
        "chapter5_results_rq2.md",
        "Downstream recall against held-out human gold by label source. "
        "Every arm trains on the same pairs, so the label source is the only "
        "variable. Self-training improves on the human labels everywhere "
        "except \\texttt{near}, where it falls below them.",
        "## 5.3 Why self-training"),
    "rq1_with_vlm.png": (
        "chapter4_results_rq1.md",
        "Recall of the human triplets per predicate: the geometric pipeline "
        "against both vision-language models on the same 30 images, the same "
        "numbered boxes and the same written definitions. The dotted lines "
        "are the two means. Scaling the model lifts every predicate a little "
        "and closes none of the gap.",
        "after:## 4.13 Would a vision-language"),
    "sgg_training_curves.png": (
        "chapter6_benchmark.md",
        "Validation curves for both benchmark arms, each against its own "
        "training-source labels, so only the shapes are comparable. The "
        "human arm peaks early and declines; the automatic arm does not.",
        "## 6.3 Test results"),
}

# An identifier like lateral_center_eps is one unbreakable word inside a
# narrow table column, and ran 34pt into the margin there. Allowing a break
# after each underscore lets it wrap at a natural seam, with no hyphen added.
SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": "\\_\\allowbreak{}", "{": r"\{", "}": r"\}"}
UNICODE = {
    "\u2013": "--", "\u2014": "---", "\u2018": "`", "\u2019": "'",
    # Control words that could be followed by a letter carry a {} terminator:
    # "60281435...e1bd" became "\ldotse1bd", which TeX reads as one undefined
    # command and refuses to compile. A digit terminates a control word, so
    # \S4.12 was always safe, but \S is braced for the same reason.
    "\u201c": "``", "\u201d": "''", "\u00a7": r"\S{}", "\u2248": r"$\approx$",
    "\u2264": r"$\leq$", "\u2265": r"$\geq$", "\u00d7": r"$\times$",
    "\u2192": r"$\rightarrow$", "\u2190": r"$\leftarrow$", "\u00b1": r"$\pm$",
    "\u03c4": r"$\tau$", "\u2212": "-", "\u2260": r"$\neq$",
    "\u00b7": r"$\cdot$", "\u00b0": r"$^{\circ}$", "\u03b5": r"$\epsilon$",
    "\u2026": r"\ldots{}", "\u2011": "-", "\u00a0": "~", "\u2032": "'",
    "\u2033": "''",
    # Greek used as statistics notation. Chapter 2 names Cohen's kappa and
    # Krippendorff's alpha, so the letters appear in prose as well as in the
    # tables that report them.
    "\u03ba": r"$\kappa$", "\u03b1": r"$\alpha$", "\u03c3": r"$\sigma$",
    "\u03bc": r"$\mu$", "\u03c1": r"$\rho$", "\u03bb": r"$\lambda$",
}


def unmapped_unicode(text: str, where: str) -> list[str]:
    """Characters with no mapping that LaTeX cannot render on its own.

    A single one fails the whole document with "not set up for use with
    LaTeX", and the error names the character but not the file, so this
    reports them at build time while the source is still in hand.

    Latin letters with diacritics are excluded: inputenc renders Fréchet from
    UTF-8 without help, and a guard that fires on correct text is a guard that
    gets ignored. Everything past Latin Extended-B -- Greek, arrows, maths --
    does need an entry here.
    """
    return sorted({c for c in text
                   if ord(c) > 0x24F and c not in UNICODE})


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

    text = re.sub(r"\x00(\d+)\x00", pop, text)
    # A URL is one unbreakable word to TeX, so an 85-character one runs 35pt
    # into the margin however much the surrounding line is stretched. Marking
    # the separators as break opportunities lets it wrap; \allowbreak adds no
    # hyphen, which is what a URL needs.
    return re.sub(r"https?://[^\s}]+",
                  lambda m: re.sub(r"([/._-])", r"\1\\allowbreak{}", m.group(0)),
                  text)


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
        # 4.4
        "Manual audit of a stratified sample of extra predictions against the "
        "pre-gate box rule, with Wilson intervals. Superseded by the blind "
        "re-audit of Section 4.14; reported because it motivated the repair.",
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
        # 4.13
        "A vision-language model against the geometric pipeline on the same "
        "30 images, the same pairs and the same human gold: recall.",
        # 4.14 (the blind re-audit; last table in the chapter)
        "Blind, decoy-controlled audit of 242 sampled items, verdicted "
        "independently by the author and by a vision-language model, with "
        "Wilson intervals. The final row is the decoy control: relations the "
        "tool did not emit, which a judge who agreed with everything would "
        "reject none of.",
        # 4.14, second table: the re-audit after the threshold was re-fitted
        "Per-predicate precision of the shipped tool: the re-fitted "
        "threshold audited on a fresh draw, beside the superseded v3 column, "
        "so the comparison is between two independent "
        "draws and not between two readings of one.",
        # 4.15, the volunteer arm against both audit judges
        "Precision on the same pre-refit label generation under three "
        "judges: volunteers who did not build the tool (Appendix E.3), the "
        "blinded author, and the vision-language model.",
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
        # D.8
        "Ablation A10: whether contact height can replace the class guard. "
        "Drop fraction is where the subject's bottom edge falls inside the "
        "object's vertical extent, 0 being the object's top surface. No "
        "threshold both keeps the gold resting pairs and blocks the held ones.",
        # E.1
        "The vision-language comparison of §4.13 restricted to the pairs "
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
        "sparse-gold artefact of §4.3 reappearing downstream rather than a "
        "verdict on label quality.",
        # F.6
        "Restricted precision, recall and F1 on the human-annotated pairs "
        "only, where an unannotated pair cannot be scored and precision is "
        "therefore a floor.",
        # F.8
        "Per-predicate recall on the held-out annotator groups with 95% "
        "cluster-bootstrap intervals over images.",
    ],
}


# How many characters of \small text fit across the a4 text block, given that
# longtable adds 2*\tabcolsep of padding per column. A table under the budget
# can use natural-width columns, which look better; over it, the columns must
# wrap or the table runs off the page and drags its caption with it.
TEXT_PT = 452.0        # a4 width less 1in margins, in points
COLPAD_PT = 12.0       # 2 * \tabcolsep
CHAR_PT = 5.2          # mean glyph advance at \small in a 12pt document


# Sizes a wide table may fall back to before its columns are made to wrap,
# with the mean glyph advance each implies and the \tabcolsep it uses.
SIZE_STEPS = [(r"\small", CHAR_PT, 6.0),
              (r"\footnotesize", CHAR_PT * 10.0 / 11.0, 3.0),
              (r"\footnotesize", CHAR_PT * 10.0 / 11.0, 2.0),
              (r"\scriptsize", CHAR_PT * 8.0 / 11.0, 3.0)]


def char_budget(ncol: int, char_pt: float = CHAR_PT,
                tabcolsep: float = 6.0) -> float:
    return max(20.0, (TEXT_PT - 2.0 * tabcolsep * ncol) / char_pt)


def fit_size(widest_total: float, ncol: int):
    """Smallest shrink that lets the row sit on one line; None if none does."""
    for size, char_pt, sep in SIZE_STEPS:
        if widest_total <= char_budget(ncol, char_pt, sep):
            return size, sep
    return None, None


def is_numeric_col(body: list[list[str]], i: int) -> bool:
    """True when the column's body is dominated by numbers, so it reads
    better right-aligned."""
    vals = [r[i] for r in body if i < len(r) and r[i].strip()]
    if not vals:
        return False
    numish = sum(bool(re.fullmatch(r"[\d.,%/()+\-– ]*\*{0,2}[\d.,%/()+\-– ]*", v))
                 for v in vals)
    return numish >= 0.7 * len(vals)


def longest_token(cells: list[list[str]], ncol: int) -> list[int]:
    """Longest unbreakable word in each column, in characters.

    Emphasis and code markers are stripped: they do not survive into the
    typeset word, so counting them would overstate the width needed."""
    out = []
    for i in range(ncol):
        longest = 1
        for r in cells:
            if i < len(r):
                for w in re.sub(r"[*`_]", "", r[i]).split():
                    longest = max(longest, len(w))
        out.append(longest)
    return out


def plan_table(cells: list[list[str]], ncol: int):
    """Type size, inter-column padding and column spec, chosen together.

    A table that fits keeps natural column widths, which look better. One
    that does not is shrunk a step at a time, and only when no size fits are
    the columns made to wrap. Wrapped columns are floored at the width of
    their own longest word so that nothing can overprint, and whatever room
    is left over is shared out by how much text each column actually holds."""
    body = cells[1:] or cells
    widest = [max((len(r[i]) for r in cells if i < len(r)), default=1)
              for i in range(ncol)]
    numeric = [is_numeric_col(body, i) for i in range(ncol)]

    size, sep = fit_size(sum(widest), ncol)
    if size is not None:
        if ncol <= 2:
            return "l" * ncol, size, sep
        return ("l" + "".join("r" if numeric[i] else "l"
                              for i in range(1, ncol)), size, sep)

    # Wrapping. TT glyphs in code spans run wider than the mean proportional
    # advance CHAR_PT measures, so the floors carry 6% slack.
    # A column whose widest cell is itself short should never wrap: breaking
    # "<1 min" over two lines to save four points helps nobody.
    tok = [max(k, min(w, 10)) for k, w in
           zip(longest_token(cells, ncol), widest)]
    share = [max(w, 4) for w in widest]
    total = sum(share)
    # Wrapped tables start a step down. A table that has to wrap is already
    # dense, and setting it at the largest size that merely fits widens every
    # floor, which buys wider columns at the cost of more wrapped lines.
    frac = None
    for size, char_pt, sep in SIZE_STEPS[1:]:
        budget = 1.0 - 2.0 * sep * ncol / TEXT_PT - 0.01
        floors = [1.06 * k * char_pt / TEXT_PT for k in tok]
        slack = budget - sum(floors)
        if slack > 0.0:
            frac = [f + slack * (s / total) for f, s in zip(floors, share)]
            break
    if frac is None:
        # Even the smallest size cannot hold every long word. Nothing avoids
        # a break here, so share the budget out in proportion and let LaTeX
        # report the overfull box rather than hiding it.
        size, char_pt, sep = SIZE_STEPS[-1]
        budget = 1.0 - 2.0 * sep * ncol / TEXT_PT - 0.01
        floors = [1.06 * k * char_pt / TEXT_PT for k in tok]
        frac = [f * budget / sum(floors) for f in floors]

    align = [r">{\raggedleft\arraybackslash}" if numeric[i]
             else r">{\raggedright\arraybackslash}" for i in range(ncol)]
    spec = "".join(f"{align[i]}p{{{frac[i]:.3f}\\textwidth}}"
                   for i in range(ncol))
    return spec, size, sep


def table(rows: list[str], caption: str = "") -> str:
    """A Markdown pipe table -> longtable (survives page breaks)."""
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [c for c in cells if not all(set(x) <= set("-: ") for x in c)]
    if not cells:
        return ""
    ncol = max(len(r) for r in cells)
    head, body = cells[0], cells[1:]
    head += [""] * (ncol - len(head))
    spec, size, sep = plan_table(cells, ncol)
    out = [r"\begin{center}",
           r"\setlength{\tabcolsep}{" + f"{sep:g}" + "pt}", size,
           r"\begin{longtable}{" + spec + "}"]
    if caption:
        # longtable takes its caption as the first row; this is what puts the
        # table into \listoftables and gives it a number.
        out.append(caption_cmd(caption) + r"\\")
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
            out.append(r"\begingroup\small\begin{verbatim}" if in_code
                       else r"\end{verbatim}\endgroup")
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
                if fname in placed_figs:
                    continue
                if anchor and not anchor.startswith("after:") \
                        and line.strip().startswith(anchor.strip()):
                    out.append(figure_block(fname, caption))
                    placed_figs.add(fname)
            out.append("\\" + cmd + "{" + inline(title) + "}")
            for fname, caption, anchor in figures:
                if fname in placed_figs:
                    continue
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
        raw_para = " ".join(para)
        out.append(inline(raw_para))
        # A figure belongs on the page that points at it. Anchoring by heading
        # put three of them two or three pages past their own sentence, so a
        # figure whose {{fig:...}} marker appears in this paragraph is emitted
        # right after it; the heading anchor stays as the fallback.
        for fname, caption, _a in figures:
            if fname in placed_figs:
                continue
            key = "{{fig:" + Path(fname).stem.replace("_", "-") + "}}"
            if key in raw_para:
                out.append(figure_block(fname, caption))
                placed_figs.add(fname)

    flush_table()
    close_lists()
    if in_code:
        out.append(r"\end{verbatim}\endgroup")

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


# Words a shortened caption should not end on.
CAPTION_STOP = {"the", "a", "an", "of", "on", "for", "in", "with", "and",
                "to", "by", "at", "from", "its", "their", "every", "each",
                "that", "as", "is", "are", "was", "which", "where", "when",
                "than", "against", "over", "under", "into", "per",
                "both", "two", "three", "same", "such", "this",
                "these", "those", "one"}

# Short forms already handed out, keyed by the full caption so that
# asking twice for the same caption gives the same answer.
_SHORT_USED: dict[str, str] = {}

def short_caption(caption: str) -> str:
    """An identifying form of a caption, for the List of Tables/Figures.

    The full caption belongs under the float, where the reader is looking at
    the thing it describes; in the lists it only has to be distinguishable.
    Cuts at the first sentence or clause boundary, then at a word boundary."""
    full = " ".join(caption.split())
    if full in _SHORT_USED:
        return _SHORT_USED[full]
    s = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", full)  # \texttt{near} -> near
    plain = s
    for sep in (". ", ": ", "; ", ", which ", " -- "):
        if sep in s[:90]:
            s = s.split(sep)[0]
            break
    if s in _SHORT_USED.values():
        # Two captions opening with the same clause would be listed under one
        # name. Take the plain 62-character cut instead, which keeps whatever
        # distinguishes them.
        s = plain
    # 62 characters is what fits on one line of the List of Tables once the
    # number box, the dot leader and the folio have taken their share. Above
    # that the entry wraps, and a wrapped list entry costs more than the
    # words it carries are worth.
    if len(s) > 62:
        s = s[:62].rsplit(" ", 1)[0]
        # A cut at a word boundary can still land on "of" or "the", which
        # reads as an abandoned sentence rather than a short title.
        words = s.split()
        while len(words) > 3 and words[-1].lower().strip(".,;:") in CAPTION_STOP:
            words.pop()
        s = " ".join(words)
    # a ] would close the optional argument early
    s = s.split("]")[0].rstrip(" ,;:.")
    _SHORT_USED[full] = s
    return s


def caption_cmd(caption: str, escape: bool = True) -> str:
    """\\caption, with a short form when the full one is long enough to want
    one. Figure captions carry hand-written LaTeX and are passed through."""
    body = inline(caption) if escape else caption
    short = short_caption(caption)
    # The optional argument is LaTeX too: an unescaped % there comments out
    # the rest of the line, which ends the longtable's caption mid-scan.
    short_tex = inline(short) if escape else short.replace("%", r"\%")
    if len(short) < len(" ".join(caption.split())) - 12:
        return r"\caption[" + short_tex + "]{" + body + "}"
    return r"\caption{" + body + "}"

def figure_block(fname: str, caption: str) -> str:
    safe = tex_figname(fname)
    return "\n".join([
        r"\begin{figure}[!htbp]", r"\centering",
        r"\includegraphics[width=0.80\textwidth]{figures/" + safe + "}",
        caption_cmd(caption, escape=False),
        r"\label{fig:" + Path(safe).stem + "}",
        r"\end{figure}", ""])


def build_references(md: str) -> str:
    entries = [b.strip() for b in re.split(r"\n\s*\n", md) if b.strip()
               and not b.strip().startswith("#")]
    # Set a step down, as reference lists conventionally are: fifty-seven
    # hanging-indent entries at body size run four lines past the third
    # page and leave a fourth one almost empty.
    out = [r"\small",
           r"\begin{description}[leftmargin=2em, labelindent=0em,"
           r" itemsep=0.1em, parsep=0pt, topsep=0.3em]"]
    for e in entries:
        out.append(r"\item[] " + inline(" ".join(e.split())))
    out.append(r"\end{description}")
    out.append(r"\normalsize")
    return "\n".join(out)


# pdflatex writes these next to main.tex; they are outputs, not sources. The
# check is by name, not by extension: figures/UniofSurrey.pdf is a source, and
# excluding "*.pdf" once left the submitted title page with no crest.
BUILD_PRODUCTS = {".aux", ".log", ".out", ".toc", ".lot", ".lof", ".fls",
                  ".fdb_latexmk", ".synctex.gz", ".bbl", ".blg"}


def write_zip(out: Path) -> None:
    """Zip the generated tree for Overleaf, sources only."""
    import zipfile
    z_path = out.parent / "latex.zip"
    files = []
    for f in sorted(out.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(out)
        if rel.parent == Path(".") and (f.suffix in BUILD_PRODUCTS
                                        or f.name == "main.pdf"):
            continue
        files.append((f, rel))
    with zipfile.ZipFile(z_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f, rel in files:
            z.write(f, "latex/" + str(rel).replace("\\", "/"))
    figs = sum(1 for _, rel in files if rel.parts[0] == "figures")
    logo = any(rel.name == "UniofSurrey.pdf" for _, rel in files)
    print(f"  latex.zip: {len(files)} files, {figs} figures, "
          f"logo {'present' if logo else 'MISSING'}")
    if not logo:
        print("  WARNING: the title page will compile without the crest")


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
        stray = unmapped_unicode(md, md_name)
        if stray:
            print(f"  WARNING: {md_name} has unmapped characters, the compile "
                  f"will fail on them: {' '.join(stray)}")
        figs = [(f, c, a) for f, (owner, c, a) in FIGURES.items()
                if owner == md_name and (FIGDIR / f).exists()]
        tex = convert(md, figs, TABLES.get(md_name))
        placed = sum(1 for f, _c, a in figs
                     if a and a.removeprefix("after:") in md)
        n_tab = tex.count(r"\begin{longtable}")
        n_cap = len(re.findall(r"\\caption[\[{]", tex)) - sum(1 for f, _c, a in figs
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
    # The chapter loop checks the caption registry against the tables it is
    # applied to; the appendices did not. A table added to D.8 without a
    # caption shifted every caption after it onto the wrong table, down to
    # F.8, which ended up with none. Same check, same reason.
    n_app_tab = sum(1 for ln in app_md.splitlines()
                    if ln.strip().startswith("|---"))
    n_app_cap = len(TABLES.get("appendices.md", []))
    app_note = ""
    if n_app_cap != n_app_tab:
        app_note = (f"  WARNING: {n_app_cap} captions declared for "
                    f"{n_app_tab} tables; captions are positional, so every "
                    f"one after the mismatch sits on the wrong table")
    elif app_caps:
        app_note = f"  WARNING: {len(app_caps)} captions never consumed"
    print(f"  appendices.md{'':21s} -> appendices.tex  ({len(chunks)} appendices, {n_app_tab} tables){app_note}")
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
    write_zip(out)
    print(f"\nwrote {out}  (declared word count {words:,})")
    print("compile:  cd dissertation/latex && latexmk -pdf main.tex")
    print("or upload the folder to Overleaf and compile there.")


MAIN = r"""% GENERATED by scripts/build_latex.py -- do not edit by hand.
% The Markdown chapters in dissertation/ are the source of truth; rebuild
% with `python scripts/build_latex.py` after editing them.
\documentclass[12pt]{report}
% 1in on a4. The template's own sectPr asks for 1.248in left and right, but
% that is Word's legacy A4 default rather than a stated requirement, and
% adopting it costs five pages against a hard 60-page limit and pushes two
% tables back into the margin. Nothing in the template or the module
% documents states a margin rule.
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
% Float placement. LaTeX's defaults reserve so much of a page for text that a
% figure at 0.8\textwidth is often deferred, and three of the eight were
% landing two or three pages after the sentence that points at them, which
% makes the reader hunt. Loosening the fractions lets a figure sit on the page
% that refers to it.
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.7}
\renewcommand{\textfraction}{0.08}
\renewcommand{\floatpagefraction}{0.75}
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{4}

\sloppy
% Lets TeX stretch a stubborn line rather than push it into the margin. Without
% it, prose containing a long unbreakable token (a URL, a file path, a hash)
% overflows instead of loosening.
\setlength{\emergencystretch}{3em}

% Heading hierarchy taken from the Word template's styles.xml rather than from
% the report class. The template defines Heading1 (chapters) as Times New
% Roman, bold, all-capitals and CENTRED, applied at 11-12pt, and Heading2
% (sections) as bold at 11pt against a 12pt Normal body. The class defaults
% are \Large for both and left-aligned, which is two steps too large and the
% wrong alignment for every heading in the document.
\titleformat{\chapter}[display]
  {\normalfont\normalsize\bfseries\centering}
  {\MakeUppercase{\chaptertitlename\ \thechapter}}{0.4em}{\MakeUppercase}
\titlespacing*{\chapter}{0pt}{0pt}{18pt}

% sections and below: bold, one step under body text, left aligned
\titleformat{\section}{\normalfont\small\bfseries}{\thesection}{0.6em}{}
\titlespacing*{\section}{0pt}{1.4ex plus .2ex}{0.7ex}
\titleformat{\subsection}{\normalfont\small\bfseries}{\thesubsection}{0.6em}{}
\titlespacing*{\subsection}{0pt}{1.2ex plus .2ex}{0.6ex}
\titleformat{\subsubsection}{\normalfont\small\bfseries\itshape}{\thesubsubsection}{0.6em}{}
\titlespacing*{\subsubsection}{0pt}{1.1ex plus .2ex}{0.5ex}

% The template's contents page lists chapters as "CHAPTER 1: INTRODUCTION";
% the class default is "1 Introduction". tocloft is already loaded, so the
% prefix, the separator and a wider number box are all that is needed.
\renewcommand{\cftchappresnum}{CHAPTER~}
\renewcommand{\cftchapaftersnum}{:\hspace{0.5em}}
\settowidth{\cftchapnumwidth}{\bfseries CHAPTER~9:\hspace{0.75em}}
\renewcommand{\cftchapfont}{\bfseries}
% tocloft leaves a line of air before every chapter block. Over nineteen
% of them that is a page of contents spent on nothing.
\setlength{\cftbeforechapskip}{0.35em}
\setlength{\cftbeforesecskip}{0pt}
\setlength{\cftbeforesubsecskip}{0pt}
\setlength{\cftbeforetabskip}{0pt}
\setlength{\cftbeforefigskip}{0pt}
\renewcommand{\cftchappagefont}{\bfseries}

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

    {\fontsize{22}{26}\selectfont \textbf{A Fully-Automatic Spatial-Relationship\\[0.3cm]
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
  \item[LLM] Large Language Model
  \item[mAP] mean Average Precision
  \item[mR@K] mean Recall at K, averaged per predicate
  \item[PredCls] Predicate Classification (ground-truth boxes and classes given)
  \item[REACT++] the source paper's scene-graph framework (Neau and Falomir, 2026)
  \item[RGB-D] colour imagery with a registered depth channel
  \item[RQ] Research Question
  \item[R@K] Recall at K, pooled over predicates
  \item[SAM2] Segment Anything Model 2
  \item[SGDet] Scene Graph Detection (detection, classification and relations end to end)
  \item[SGDET-Annotate] the manual annotation tool used to build the source dataset
  \item[SGG] Scene Graph Generation
  \item[VG] Visual Genome (the annotation format this dataset inherits)
  \item[VLM] Vision-Language Model
  \item[VQA] Visual Question Answering
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
