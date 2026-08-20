"""Regenerate every dissertation figure from cached result artefacts.

Reads only the JSON/markdown reports already written by the eval scripts, so it
needs no GPU, no torch and no dataset — just matplotlib. Run after the fidelity,
downstream and ablation passes have produced their outputs.

    python scripts/make_figures.py

Figures are written to outputs/figures/ at 200 dpi.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# --- shared house style -----------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 10,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

C_MAIN = "#2b6cb0"   # ours / full pipeline / auto-trained
C_SEC = "#8bb8de"    # box-only ablation
C_GRAY = "#c4c4c4"   # random / human-trained baseline
C_RED = "#c0392b"    # disagreement / inverted convention
C_VLM = "#b8860b"    # vision-language model, reasoning
C_VLM2 = "#e0c068"   # vision-language model, small
C_GREEN = "#2e7d52"  # the two automatic sources combined

PRED_ORDER = ["on", "under", "to the left of", "to the right of",
              "in front of", "behind", "near"]
PRED_SHORT = {"on": "on", "under": "under", "to the left of": "left of",
              "to the right of": "right of", "in front of": "in front of",
              "behind": "behind", "near": "near"}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8)


def _bar_labels(ax, bars, fmt="{:.2f}", dy=0.012, fontsize=8.5, color="#333"):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=fontsize, color=color)


# --- Figure 1: RQ1 recall by predicate --------------------------------------
def fig_rq1_recall():
    rep = json.loads((ROOT / "outputs" / "fidelity_report.json").read_text())
    rec = rep["recall"]
    labels = [PRED_SHORT[p] for p in PRED_ORDER]
    ours = [rec["ours"][p]["recall"] for p in PRED_ORDER]
    box = [rec["box_only"][p]["recall"] for p in PRED_ORDER]
    rand = [rec["random"][p]["recall"] for p in PRED_ORDER]

    x = range(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    b1 = ax.bar([i - w for i in x], ours, w, color=C_MAIN,
                label="Full pipeline (shipped rules)")
    ax.bar(list(x), box, w, color=C_SEC, label="Box-only (no masks / depth)")
    ax.bar([i + w for i in x], rand, w, color=C_GRAY, label="Random predicate")

    _bar_labels(ax, b1)                      # label the headline series only
    ax.set_ylabel("Recall of human triplets")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    _style(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=3, frameon=False, columnspacing=1.4, handlelength=1.3)
    out = FIG / "rq1_recall.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"figure -> {out.relative_to(ROOT)}")


# --- Figure 7: RQ1 recall with the vision-language baselines -----------------
def fig_rq1_with_vlm():
    """Pipeline against two VLMs on the pairs all three were scored on.

    The 30-image pilot is its own population, so the pipeline column here is
    the pipeline's score on those same 30 images rather than its full-dataset
    figure. Mixing the two would compare different denominators.
    """
    a = ROOT / "outputs" / "vlm_pilot" / "scores.json"
    b = ROOT / "outputs" / "vlm_pilot" / "scores_pro.json"
    if not (a.exists() and b.exists()):
        print("skip rq1_with_vlm: run eval/score_vlm_pilot.py for both models")
        return
    A, B = json.loads(a.read_text()), json.loads(b.read_text())
    labels = [PRED_SHORT[p] for p in PRED_ORDER]
    pipe = [A["pipeline_recall"][p] for p in PRED_ORDER]
    flash = [A["vlm_recall"][p] for p in PRED_ORDER]
    pro = [B["vlm_recall"][p] for p in PRED_ORDER]

    x = range(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    b1 = ax.bar([i - w for i in x], pipe, w, color=C_MAIN, label="Geometric pipeline")
    ax.bar(list(x), pro, w, color=C_VLM, label="Gemini 3.1 Pro (reasoning)")
    ax.bar([i + w for i in x], flash, w, color=C_VLM2, label="Gemini Flash")

    _bar_labels(ax, b1)
    ax.set_ylabel("Recall of human triplets")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    _style(ax)
    # The two means were annotated inline at the left edge, in the same colour
    # as the bars they sat on top of, which made both unreadable where they
    # overlapped. They go in the legend instead: same information, no collision
    # possible, and the value is stated rather than left to be read off the axis.
    mean_lines = []
    for value, colour, name in ((A["pipeline_mean"], C_MAIN, "pipeline"),
                                (B["vlm_mean"], C_VLM, "Gemini Pro")):
        ax.axhline(value, color=colour, ls=":", lw=1.2, alpha=0.75)
        mean_lines.append(Line2D([], [], color=colour, ls=":", lw=1.2,
                                 label=f"{name} mean {value:.2f}"))
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + mean_lines,
              loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              frameon=False, columnspacing=1.4, handlelength=1.6)
    out = FIG / "rq1_with_vlm.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"figure -> {out.relative_to(ROOT)}")


# --- Figure 8: what a planner does with each relation source ----------------
def fig_planner_sources():
    """Safe plans by relation source, including the union of the two automatic
    sources. The union is the point: neither automatic source matches human
    annotation alone, and together they do."""
    f = ROOT / "outputs" / "planner_scores_abcde.json"
    if not f.exists():
        print("skip planner_sources: run eval/score_planner.py on the 5-condition set")
        return
    s = json.loads(f.read_text())["summary"]
    order = ["A", "D", "C", "E", "B"]
    names = {"A": "No relations", "B": "Human", "C": "Pipeline",
             "D": "Gemini", "E": "Pipeline\n+ Gemini"}
    vals = [s[c]["clears_first_count"] for c in order]
    cols = {"A": C_GRAY, "B": "#4a4a4a", "C": C_MAIN, "D": C_VLM, "E": C_GREEN}

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bars = ax.bar([names[c] for c in order], vals,
                  color=[cols[c] for c in order], width=0.62)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4, f"{v}/25",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Scenes with a safe plan (of 25)")
    ax.set_ylim(0, 28)
    ax.axhline(25, color="#4a4a4a", ls=":", lw=1.1, alpha=0.6)
    _style(ax)
    ax.set_title("Occluder cleared before grasping, by relation source",
                 pad=10)
    out = FIG / "planner_sources.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"figure -> {out.relative_to(ROOT)}")


# --- Figure 9: learning curves per label source, including a VLM ------------
def fig_learning_curves():
    """Held-out recall epoch by epoch for each label source.

    The companion to the SGG training curves of Chapter 6, which cannot carry
    a vision-language arm because they need a GPU run. Here every arm shares
    features, architecture, seed and training pairs, so the curves differ only
    by who supplied the labels.
    """
    f = ROOT / "outputs" / "learning_curves.json"
    if not f.exists():
        print("skip learning_curves: run eval/learning_curves.py first")
        return
    d = json.loads(f.read_text())
    curves = d["curves"]
    style = {"pipeline": (C_MAIN, "Pipeline labels", "-"),
             "human":    ("#4a4a4a", "Human labels", "-"),
             "vlm":      (C_VLM, "Gemini labels", "-")}

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for name in ("pipeline", "human", "vlm"):
        if name not in curves:
            continue
        y = curves[name]["mean"]
        col, lab, ls = style[name]
        ax.plot(range(1, len(y) + 1), y, ls, color=col, lw=2.0, label=lab)
        best = max(y)
        ax.plot(y.index(best) + 1, best, "o", color=col, ms=5)
        ax.annotate(f"{best:.2f}", (y.index(best) + 1, best),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=9, color=col)

    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Mean recall vs held-out human gold")
    ax.set_xlim(1, d["epochs"])
    ax.set_ylim(0, 1.0)
    _style(ax)
    ax.legend(loc="lower right", frameon=False)
    n_img = d.get("vlm_images")
    sub = f"{d['train_pairs']:,} training pairs"
    if n_img:
        sub += f" from {n_img} images"
    ax.set_title(f"What each label source has to teach ({sub})", pad=10)
    out = FIG / "learning_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"figure -> {out.relative_to(ROOT)}")


# --- Figure 10: RQ2 with a vision-language label source ---------------------
def fig_rq2_with_vlm():
    """Downstream recall by label source, with the VLM arm alongside.

    Read from the four-arm report rather than the three-arm one, because in
    that run every arm trains on the pairs the VLM covers; the two reports
    are not interchangeable.
    """
    f = ROOT / "outputs" / "rq2_report_vlm.json"
    if not f.exists():
        print("skip rq2_with_vlm: run downstream.py --vlm-replies first")
        return
    rep = json.loads(f.read_text())
    order = [("human-trained", "Human", "#4a4a4a"),
             ("pseudo-labelled", "Human + self-training", C_GRAY),
             ("vlm-trained", "Gemini", C_VLM),
             ("auto-trained", "Pipeline", C_MAIN)]
    order = [o for o in order if o[0] in rep]
    labels = [PRED_SHORT[p] for p in PRED_ORDER]

    x = range(len(labels))
    w = 0.8 / len(order)
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    for j, (key, name, col) in enumerate(order):
        d = rep[key]
        vals = [d[p]["recall"] for p in PRED_ORDER]
        off = (j - (len(order) - 1) / 2) * w
        bars = ax.bar([i + off for i in x], vals, w, color=col, label=name)
        if key == "auto-trained":
            _bar_labels(ax, bars, dy=0.01, fontsize=8)

    ax.set_ylabel("Recall vs held-out human gold")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    _style(ax)
    means = {name: sum(rep[key][p]["recall"] for p in PRED_ORDER) / len(PRED_ORDER)
             for key, name, _ in order}
    sub = "   ".join(f"{n} {m:.2f}" for n, m in means.items())
    ax.set_title(f"mean recall:   {sub}", pad=10, fontsize=10)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08),
              ncol=len(order), frameon=False, columnspacing=1.4,
              handlelength=1.3)
    out = FIG / "rq2_with_vlm.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"figure -> {out.relative_to(ROOT)}")


# --- Figure 2: RQ2 human- vs auto-trained downstream recall ------------------
def fig_rq2_comparison():
    rep = json.loads((ROOT / "outputs" / "rq2_report.json").read_text())
    hu, au = rep["human-trained"], rep["auto-trained"]
    ps = rep.get("pseudo-labelled")
    labels = [PRED_SHORT[p] for p in PRED_ORDER]

    def series(d):
        vals = [d[p]["recall"] for p in PRED_ORDER]
        lo = [max(0.0, d[p]["recall"] - d[p]["recall_min"]) for p in PRED_ORDER]
        hi = [max(0.0, d[p]["recall_max"] - d[p]["recall"]) for p in PRED_ORDER]
        return vals, [lo, hi]

    hv, herr = series(hu)
    av, aerr = series(au)
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ekw = dict(ecolor="#555", capsize=3, elinewidth=1.1)

    if ps:  # three arms: human, self-trained, automatic
        pv, perr = series(ps)
        w = 0.27
        ax.bar([i - w for i in x], hv, w, yerr=herr, color=C_GRAY,
               label="trained on human labels", error_kw=ekw)
        ax.bar(list(x), pv, w, yerr=perr, color="#9ecae1",
               label="human labels + self-training", error_kw=ekw)
        ax.bar([i + w for i in x], av, w, yerr=aerr, color=C_MAIN,
               label="trained on automatic labels", error_kw=ekw)
        note = (f"mean recall:  human {sum(hv)/len(hv):.2f}    "
                f"self-trained {sum(pv)/len(pv):.2f}    auto {sum(av)/len(av):.2f}")
        ncol = 3
    else:
        w = 0.38
        ax.bar([i - w / 2 for i in x], hv, w, yerr=herr, color=C_GRAY,
               label="trained on human labels", error_kw=ekw)
        ax.bar([i + w / 2 for i in x], av, w, yerr=aerr, color=C_MAIN,
               label="trained on automatic labels", error_kw=ekw)
        note = (f"mean recall:  human {sum(hv)/len(hv):.2f}    "
                f"auto {sum(av)/len(av):.2f}")
        ncol = 2

    ax.set_ylabel("Recall vs held-out human gold")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    _style(ax)
    ax.text(0.015, 0.98, note,
            transform=ax.transAxes, fontsize=9.5, color="#333", va="top")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=ncol, frameon=False, columnspacing=1.6, handlelength=1.3)
    out = FIG / "rq2_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"figure -> {out.relative_to(ROOT)}")


# --- Figure 3: front/behind decomposition per annotator group ----------------
def fig_front_behind():
    rep = json.loads((ROOT / "outputs" / "fidelity_report.json").read_text())
    fb = rep["front_behind_decomposition"]
    groups = sorted((g for g in fb if g.startswith("group_")),
                    key=lambda g: int(g.split("_")[1]))

    recovered, disagree, abstain, ticks, inverted_idx = [], [], [], [], []
    for i, g in enumerate(groups):
        d = fb[g]
        emit, agr = d["emit_rate"], d["direction_agreement_when_committed"]
        rec = emit * agr
        recovered.append(rec)
        disagree.append(emit - rec)
        abstain.append(1.0 - emit)
        ticks.append(f"g{g.split('_')[1]}\nn={d['gold']}")
        if d.get("convention") == "inverted":
            inverted_idx.append(i)

    x = range(len(groups))
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ax.bar(list(x), recovered, color=C_MAIN, label="recovered (agrees with annotator)")
    ax.bar(list(x), disagree, bottom=recovered, color=C_RED,
           label="committed, annotator disagrees")
    bot = [r + d for r, d in zip(recovered, disagree)]
    ax.bar(list(x), abstain, bottom=bot, color="#d9d9d9",
           label="abstained (depth-ambiguity band)")

    ax.set_ylabel("Share of the group's front/behind gold")
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticks(list(x))
    ax.set_xticklabels(ticks, fontsize=9.5)
    _style(ax)
    for i in inverted_idx:                    # clean callout above the full bar
        ax.annotate("inverted\nconvention", xy=(i, 1.005), xytext=(i, 1.15),
                    ha="center", va="top", fontsize=8.5, color=C_RED,
                    arrowprops=dict(arrowstyle="-", color=C_RED, lw=1.0))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=3, frameon=False, columnspacing=1.4, handlelength=1.3)
    out = FIG / "front_behind_decomposition.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"figure -> {out.relative_to(ROOT)}")


# --- Figure 4: near_T threshold sweep ----------------------------------------
def _parse_near_table():
    md = (ROOT / "outputs" / "tables" / "ablations.md").read_text()
    m = re.search(r"A4[^\n]*\n(.*?)(?:\n## |\Z)", md, re.S)
    if not m:
        return None
    rows = []
    for line in m.group(1).splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 4:
            try:
                rows.append((float(cells[0]), float(cells[1]),
                             float(cells[2]), float(cells[3])))
            except ValueError:
                continue
    return rows or None


def fig_near_sweep():
    rows = _parse_near_table()
    if not rows:
        print("near_T table not found; skipped near_T_sweep")
        return
    fit = json.loads((ROOT / "outputs" / "near_fit.json").read_text())
    T = fit["fitted_near_T"]
    xs = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(xs, [r[1] for r in rows], "o-", color=C_MAIN, label="near recall (all)")
    ax.plot(xs, [r[2] for r in rows], "^-", color="#2e8b57",
            label="near recall (held-out)")
    ax.plot(xs, [r[3] for r in rows], "s--", color="#e08a1e",
            label="near precision (restricted)")
    ax.axvline(T, color="#888", ls=":", lw=1.2)
    ax.text(T + 0.02, 0.05, f"fitted T = {T:.3f}", color="#555", fontsize=9)
    ax.set_xlabel("near_T  (edge gap / mean object size)")
    ax.set_ylabel("recall / precision")
    ax.set_ylim(0, 1.05)
    _style(ax)
    ax.legend(loc="center right", frameon=False)
    out = FIG / "near_T_sweep.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"figure -> {out.relative_to(ROOT)}")


# --- Figure 5: video-demo temporal stability -------------------------------
def fig_video_stability():
    import json
    clips = [("clip 1 — moving camera, static scene",
              ROOT / "outputs" / "video" / "clip1" / "frames.jsonl"),
             ("clip 2 — static camera, moving hands",
              ROOT / "outputs" / "video" / "clip2" / "frames.jsonl")]
    if not all(p.exists() for _, p in clips):
        print("video frames.jsonl not found; skipped video_stability")
        return

    def jac(a, b):
        return 1.0 if not a and not b else len(a & b) / len(a | b)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for ax, (title, path) in zip(axes, clips):
        fr = [json.loads(l) for l in open(path, encoding="utf-8")]
        raw = [set(map(tuple, f["triplets_raw"])) for f in fr]
        smo = [set(map(tuple, f["triplets_smoothed"])) for f in fr]
        t = [f["time"] for f in fr][1:]
        ax.plot(t, [jac(raw[i], raw[i + 1]) for i in range(len(raw) - 1)],
                color=C_GRAY, lw=1.2, label="raw")
        ax.plot(t, [jac(smo[i], smo[i + 1]) for i in range(len(smo) - 1)],
                color=C_MAIN, lw=1.6, label="smoothed (±2 frames)")
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("time (s)")
        ax.set_ylim(0, 1.02)
        _style(ax)
    axes[0].set_ylabel("frame-to-frame\ntriplet agreement (Jaccard)")
    axes[1].legend(loc="lower right", frameon=False)
    out = FIG / "video_stability.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"figure -> {out.relative_to(ROOT)}")


# --- Figure 6: SGG-Benchmark training curves (Week-7 experiment) ------------
def fig_sgg_curves():
    import json
    p = ROOT / "outputs" / "sgg_benchmark" / "curves.json"
    if not p.exists():
        print("sgg curves.json not found; skipped sgg_curves")
        return
    curves = json.loads(p.read_text())
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    styles = {"react_human": (C_GRAY, "trained on human labels"),
              "react_auto": (C_MAIN, "trained on pipeline labels"),
              "react_vlm": (C_VLM, "trained on Gemini labels")}
    # an arm absent from curves.json is simply not drawn
    for arm, (color, label) in ((a, s) for a, s in styles.items()
                                if a in curves):
        mr = curves[arm]["val_mr_per_epoch"]
        ax.plot(range(len(mr)), mr, color=color, lw=1.8, label=label)
        # best_epoch is 1-indexed in curves.json; the series is 0-indexed,
        # so the marker sat one epoch late and reported the wrong value.
        b = curves[arm]["best_epoch"]
        i = b - 1
        ax.plot(i, mr[i], "o", color=color, ms=7)
        ax.annotate(f"best {mr[i]:.3f} (ep {b})", (i, mr[i]),
                    textcoords="offset points", xytext=(8, 6),
                    fontsize=9, color=color)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation mR@100\n(each arm vs its own training-source labels)")
    ax.set_ylim(0, None)
    _style(ax)
    ax.legend(loc="lower right", frameon=False)
    out = FIG / "sgg_training_curves.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"figure -> {out.relative_to(ROOT)}")


def main():
    fig_rq1_recall()
    fig_rq1_with_vlm()
    fig_planner_sources()
    fig_learning_curves()
    fig_rq2_with_vlm()
    fig_rq2_comparison()
    fig_front_behind()
    fig_near_sweep()
    fig_video_stability()
    fig_sgg_curves()


if __name__ == "__main__":
    main()
