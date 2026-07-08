"""RQ2 — the controlled downstream experiment.

Train the identical lightweight classifier twice — once on the HUMAN labels,
once on the tool's AUTOMATIC labels — with the same features, architecture,
seed, oversampling and group split, then evaluate both against the HELD-OUT
HUMAN gold (annotator groups 6-8, whose data influenced nothing upstream).
The only variable is the label source, so any performance difference is
attributable to label quality alone.

Labels: y[k] = 1 for an ordered pair iff that source recorded predicate k on
it. Human labels are sparse (annotators labelled ~10% of pairs); automatic
labels are dense and rule-consistent. Both models inherit their source's
character — that contrast is the experiment.

    python eval/downstream.py        # table -> outputs/tables/rq2.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import SpatialDataset
from src.pipeline import load_config, thresholds_from_config
from src.predicates import Obj, PREDICATES, evaluate_scene
from eval.classifier import make_mlp, oversample_positives, pair_features

TRAIN_GROUPS = {f"group_{i}" for i in range(6)}


def load_matrix(cfg):
    """X, y_human, y_auto, is_train for every ordered pair in the cache."""
    ds = SpatialDataset(cfg["dataset"]["root"])
    t = thresholds_from_config(cfg)
    X, yh, ya, tr = [], [], [], []
    for gt in ds:
        group, stem = gt.image_id.split("/")
        gp = Path("outputs/geometry") / group / f"{stem}.json"
        if not gp.exists():
            continue
        geo = json.loads(gp.read_text(encoding="utf-8"))
        objs = [Obj(o["idx"], o["label"], tuple(o["box"]), o["cx"], o["cy"],
                    o["depth"], np.array(o["pos3d"])) for o in geo]
        cpath = gp.parent / f"{gp.stem}.contact.json"
        contact = {}
        if cpath.exists():
            contact = {tuple(map(int, k.split("-"))): v
                       for k, v in json.loads(cpath.read_text(encoding="utf-8")).items()}
        gold = {}
        for r in gt.relations:
            gold.setdefault((r.subject, r.object), set()).add(r.predicate)
        auto = {(p.subject, p.object): set(p.predicates)
                for p in evaluate_scene(objs, t, contact=contact)}
        is_tr = group in TRAIN_GROUPS
        for a in objs:
            for b in objs:
                if a.idx == b.idx:
                    continue
                c_ab = contact.get((a.idx, b.idx), 0.0)
                c_ba = contact.get((b.idx, a.idx), 0.0)
                X.append(pair_features(a, b, c_ab, c_ba))
                g = gold.get((a.idx, b.idx), set())
                p = auto.get((a.idx, b.idx), set())
                yh.append([int(k in g) for k in PREDICATES])
                ya.append([int(k in p) for k in PREDICATES])
                tr.append(is_tr)
    return (np.array(X), np.array(yh), np.array(ya), np.array(tr))


def train_and_eval(X_tr, y_tr, X_te, gold_te, seed=42):
    """One classifier per predicate; recall/precision vs held-out human gold."""
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    scaler = StandardScaler().fit(X_tr)
    Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)
    out = {}
    for i, k in enumerate(PREDICATES):
        Xi, yi = oversample_positives(Xs_tr, y_tr[:, i], seed=seed)
        clf = make_mlp(seed)
        clf.fit(Xi, yi)
        pred = clf.predict(Xs_te)
        g = gold_te[:, i]
        tp = int(((pred == 1) & (g == 1)).sum())
        fn = int(((pred == 0) & (g == 1)).sum())
        fp = int(((pred == 1) & (g == 0)).sum())  # fp vs sparse gold: report, don't over-read
        out[k] = {"recall": tp / (tp + fn) if tp + fn else 0.0,
                  "precision_sparse": tp / (tp + fp) if tp + fp else 0.0,
                  "support": int(g.sum())}
    return out


def main():
    cfg = load_config("configs/default.yaml")
    print("building the feature matrix from the cache ...")
    X, yh, ya, tr = load_matrix(cfg)
    print(f"pairs: {len(X)} (train {int(tr.sum())}, held-out {int((~tr).sum())})")

    gold_te = yh[~tr]
    results = {}
    for name, labels in (("human-trained", yh), ("auto-trained", ya)):
        print(f"training {name} (7 classifiers) ...")
        results[name] = train_and_eval(X[tr], labels[tr], X[~tr], gold_te)

    md = ["# RQ2 — downstream classifier: human vs automatic training labels\n",
          "Identical features, model, seed, oversampling and split; evaluated "
          "against held-out human gold (groups 6-8). Only the label source differs.\n",
          "| predicate | human-trained recall | auto-trained recall | gold (held-out) |",
          "|---|---|---|---|"]
    hv, av = [], []
    for k in PREDICATES:
        h, a = results["human-trained"][k], results["auto-trained"][k]
        hv.append(h["recall"]); av.append(a["recall"])
        md.append(f"| {k} | {h['recall']:.2f} | {a['recall']:.2f} | {h['support']} |")
    md.append(f"| **mean** | **{np.mean(hv):.2f}** | **{np.mean(av):.2f}** | |")

    Path("outputs/tables/rq2.md").write_text("\n".join(md), encoding="utf-8")
    Path("outputs/rq2_report.json").write_text(json.dumps(results, indent=2),
                                               encoding="utf-8")
    print("\n".join(md))
    print("\nreport -> outputs/rq2_report.json ; table -> outputs/tables/rq2.md")


if __name__ == "__main__":
    main()
