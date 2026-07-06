"""Ablation sweeps over the rule thresholds, offline from the geometry cache.

Every sweep rebuilds the predicate outputs for all 836 images under a modified
Thresholds and scores them against the human labels, so a full curve costs
seconds-per-point and no GPU. Sweeps:

  - on_depth_eps : the support depth-co-location gate (calibrates it)
  - depth_eps    : front/behind abstention band (recall/precision trade)
  - lateral_center_eps : left/right abstention band
  - near_T       : proximity threshold curve around the fitted value

Metrics per point: recall of human triplets (all + held-out groups), P/R/F1
restricted to human-annotated pairs, and emission counts. Calibration picks the
value maximising restricted F1 on the TRAIN groups (0-5); held-out numbers are
reported, never optimised.

    python eval/ablations.py            # tables -> outputs/tables/ablations.md
                                        # figures -> outputs/figures/
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import SpatialDataset
from src.pipeline import load_config, thresholds_from_config
from src.predicates import Obj, PREDICATES, evaluate_scene

TRAIN_GROUPS = {f"group_{i}" for i in range(6)}


def load_scenes(cfg):
    """[(image_id, group, objs, gold {(s,o): set(preds)})] from cache + dataset."""
    ds = SpatialDataset(cfg["dataset"]["root"])
    scenes = []
    for gt in ds:
        group, stem = gt.image_id.split("/")
        gp = Path("outputs/geometry") / group / f"{stem}.json"
        if not gp.exists():
            continue
        geo = json.loads(gp.read_text(encoding="utf-8"))
        objs = [Obj(o["idx"], o["label"], tuple(o["box"]), o["cx"], o["cy"],
                    o["depth"], np.array(o["pos3d"])) for o in geo]
        gold: dict[tuple[int, int], set] = {}
        for r in gt.relations:
            gold.setdefault((r.subject, r.object), set()).add(r.predicate)
        scenes.append((gt.image_id, group, objs, gold))
    return scenes


def score(scenes, thresholds, correct=True):
    """Recall (all/held-out), restricted P/R/F1 and emissions, per predicate."""
    stats = {k: {"gold": 0, "rec": 0, "gold_ho": 0, "rec_ho": 0,
                 "tp": 0, "fp": 0, "fn": 0, "emit": 0} for k in PREDICATES}
    for image_id, group, objs, gold in scenes:
        pred = {(p.subject, p.object): set(p.predicates)
                for p in evaluate_scene(objs, thresholds, correct=correct)}
        ho = group not in TRAIN_GROUPS
        for pair, gset in gold.items():
            pset = pred.get(pair, set())
            for k in gset:
                if k not in stats:
                    continue
                s = stats[k]
                s["gold"] += 1
                s["rec"] += k in pset
                if ho:
                    s["gold_ho"] += 1
                    s["rec_ho"] += k in pset
                if k in pset:
                    s["tp"] += 1
                else:
                    s["fn"] += 1
            for k in pset - gset:
                stats[k]["fp"] += 1 if gold else 0  # restricted to annotated pairs
        for pset in pred.values():
            for k in pset:
                stats[k]["emit"] += 1
    out = {}
    for k, s in stats.items():
        p = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 0.0
        r = s["rec"] / s["gold"] if s["gold"] else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        out[k] = {"recall": r,
                  "recall_ho": s["rec_ho"] / s["gold_ho"] if s["gold_ho"] else None,
                  "precision_restricted": p, "f1_restricted": f1, "emitted": s["emit"]}
    return out


def sweep(scenes, base, param, values):
    rows = []
    for v in values:
        t = dataclasses.replace(base, **{param: v})
        rows.append((v, score(scenes, t)))
    return rows


def support_f1_by_split(scenes, thresholds):
    """Restricted F1 over on+under, separately for train and held-out groups."""
    agg = {True: {"tp": 0, "fp": 0, "fn": 0}, False: {"tp": 0, "fp": 0, "fn": 0}}
    for image_id, group, objs, gold in scenes:
        pred = {(p.subject, p.object): set(p.predicates)
                for p in evaluate_scene(objs, thresholds)}
        a = agg[group in TRAIN_GROUPS]
        for pair, gset in gold.items():
            pset = pred.get(pair, set())
            for k in ("on", "under"):
                if k in gset and k in pset:
                    a["tp"] += 1
                elif k in gset:
                    a["fn"] += 1
                elif k in pset:
                    a["fp"] += 1
    out = []
    for tr in (True, False):
        a = agg[tr]
        p = a["tp"] / (a["tp"] + a["fp"]) if a["tp"] + a["fp"] else 0.0
        r = a["tp"] / (a["tp"] + a["fn"]) if a["tp"] + a["fn"] else 0.0
        out.append(2 * p * r / (p + r) if p + r else 0.0)
    return out  # (train_f1, heldout_f1)


def fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def main():
    cfg = load_config("configs/default.yaml")
    base = thresholds_from_config(cfg)
    print("loading scenes from cache ...")
    scenes = load_scenes(cfg)
    md = ["# Ablation sweeps (offline re-evaluation from cached geometry)\n"]

    # ---- 1. on_depth_eps: calibrate the support depth gate (train-only) ----
    md.append("## A1 — Support depth-co-location gate (`on_depth_eps`)\n")
    md.append("| eps | on recall | on P(restr.) | under recall | support F1 train | support F1 held-out | on emitted |")
    md.append("|---|---|---|---|---|---|---|")
    best, best_tr = None, -1
    for v in [9.9, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03]:
        t = dataclasses.replace(base, on_depth_eps=v)
        s = score(scenes, t)
        tr_f1, ho_f1 = support_f1_by_split(scenes, t)
        on, un = s["on"], s["under"]
        label = "off" if v > 1 else f"{v:.2f}"
        md.append(f"| {label} | {fmt(on['recall'])} | {fmt(on['precision_restricted'])} | "
                  f"{fmt(un['recall'])} | {fmt(tr_f1)} | {fmt(ho_f1)} | {on['emitted']} |")
        if tr_f1 > best_tr:
            best, best_tr = v, tr_f1
    md.append(f"\ncalibrated on_depth_eps = **{best}** — selected by support F1 on TRAIN "
              "groups only; the held-out column is reported, never optimised.\n")

    # ---- 2. depth_eps: front/behind abstention band ----
    md.append("## A2 — Front/behind abstention band (`depth_eps`)\n")
    md.append("| eps | front recall | front P(restr.) | behind recall | behind P(restr.) |")
    md.append("|---|---|---|---|---|")
    for v, s in sweep(scenes, base, "depth_eps", [0.0, 0.01, 0.02, 0.03, 0.05, 0.08]):
        f, b = s["in front of"], s["behind"]
        md.append(f"| {v:.2f} | {fmt(f['recall'])} | {fmt(f['precision_restricted'])} | "
                  f"{fmt(b['recall'])} | {fmt(b['precision_restricted'])} |")

    # ---- 3. lateral band ----
    md.append("\n## A3 — Lateral abstention band (`lateral_center_eps`)\n")
    md.append("| eps | left recall | left P(restr.) | right recall | right P(restr.) |")
    md.append("|---|---|---|---|---|")
    for v, s in sweep(scenes, base, "lateral_center_eps", [0.0, 0.005, 0.01, 0.02, 0.04]):
        l, r = s["to the left of"], s["to the right of"]
        md.append(f"| {v:.3f} | {fmt(l['recall'])} | {fmt(l['precision_restricted'])} | "
                  f"{fmt(r['recall'])} | {fmt(r['precision_restricted'])} |")

    # ---- 4. near_T curve ----
    md.append("\n## A4 — Proximity threshold (`near_T`)\n")
    md.append("| T | near recall | near recall (held-out) | near P(restr.) | emitted |")
    md.append("|---|---|---|---|---|")
    near_curve = []
    for v, s in sweep(scenes, base, "near_T", [0.6, 0.8, 1.0, 1.2, 1.372, 1.6, 1.9, 2.2]):
        n = s["near"]
        near_curve.append((v, n["recall"], n["precision_restricted"]))
        md.append(f"| {v:.3f} | {fmt(n['recall'])} | {fmt(n['recall_ho'])} | "
                  f"{fmt(n['precision_restricted'])} | {n['emitted']} |")

    Path("outputs/tables/ablations.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    # figure: near_T curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [c[0] for c in near_curve]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(xs, [c[1] for c in near_curve], "o-", label="near recall")
        ax.plot(xs, [c[2] for c in near_curve], "s--", label="near precision (restricted)")
        ax.axvline(1.372, color="gray", ls=":", label="fitted T = 1.372")
        ax.set_xlabel("near_T (gap / mean object size)")
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig("outputs/figures/near_T_sweep.png", dpi=200)
        print("figure -> outputs/figures/near_T_sweep.png")
    except ImportError:
        print("matplotlib unavailable; skipped figure")


if __name__ == "__main__":
    main()
