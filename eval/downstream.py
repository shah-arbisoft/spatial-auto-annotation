"""RQ2 - the controlled downstream experiment.

Train the identical lightweight classifier on three label sources - the
HUMAN labels, the tool's AUTOMATIC labels, and a PSEUDO-LABEL (self-training)
arm that stretches the human labels the way the semi-supervised literature
prescribes - with the same features, architecture, seed, oversampling and
group split, then evaluate all three against the HELD-OUT HUMAN gold
(annotator groups 6-8, whose data influenced nothing upstream). The only
variable is the label source, so any performance difference is attributable
to label quality alone.

Labels: y[k] = 1 for an ordered pair iff that source recorded predicate k on
it. Human labels are sparse (annotators labelled ~10% of pairs); automatic
labels are dense and rule-consistent. Both models inherit their source's
character - that contrast is the experiment.

The pseudo-label arm is the standard rival remedy for expensive annotation
(Lee, 2013): a teacher trained on the sparse human labels predicts the
unannotated pairs, its confident predictions become training labels, and a
student is retrained on the union. It answers the obvious question "why not
just self-train on the labels you already have?" with a measurement rather
than an argument.

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


def load_vlm_labels(path: Path) -> dict:
    """(image_id, subj, obj) -> set of predicates, from a VLM replies file.

    The VLM was shown the ground-truth boxes numbered by their position in
    the dataset's object list, which is the same order the geometry cache
    stores them in, so its indices are the cache's indices.
    """
    out: dict = {}
    seen: set = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        seen.add(r["image_id"])
        for rel in r.get("relations", []):
            key = (r["image_id"], int(rel["s"]), int(rel["o"]))
            out.setdefault(key, set()).add(rel["p"])
    return {"rels": out, "images": seen}


def load_matrix(cfg, vlm=None):
    """X, y_human, y_auto, is_train for every ordered pair in the cache.

    With `vlm`, also returns the VLM's labels and a mask marking the pairs it
    actually saw. Absence of a VLM reply for an image means "not asked", not
    "no relations here", so those pairs are excluded from that arm's training
    set rather than being fed to it as negatives.
    """
    ds = SpatialDataset(cfg["dataset"]["root"])
    t = thresholds_from_config(cfg)
    X, yh, ya, tr = [], [], [], []
    yv, vseen = [], []
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
                if vlm is not None:
                    v = vlm["rels"].get((gt.image_id, a.idx, b.idx), set())
                    yv.append([int(k in v) for k in PREDICATES])
                    vseen.append(gt.image_id in vlm["images"])
    if vlm is None:
        return (np.array(X), np.array(yh), np.array(ya), np.array(tr))
    return (np.array(X), np.array(yh), np.array(ya), np.array(tr),
            np.array(yv), np.array(vseen))


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


def train_pseudo_and_eval(X_tr, yh_tr, X_te, gold_te, seed=42, conf=0.90):
    """Self-training arm: teacher on the sparse human labels, its confident
    predictions on the UNANNOTATED training pairs become pseudo-labels, and a
    student retrains on the union (Lee, 2013).

    "Unannotated" is a property of the pair, not the predicate: a pair the
    annotators touched at all is treated as labelled (their silence on the
    other predicates is informative), while a pair they never recorded is the
    unlabelled pool the teacher fills in. Everything else - features, model,
    oversampling, seed, split - matches the other two arms exactly.
    """
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    scaler = StandardScaler().fit(X_tr)
    Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)
    annotated = yh_tr.sum(axis=1) > 0        # pairs a human actually recorded
    unlabelled = ~annotated

    out, stats = {}, {}
    for i, k in enumerate(PREDICATES):
        # stage 1: teacher, identical to the human-trained arm
        Xi, yi = oversample_positives(Xs_tr, yh_tr[:, i], seed=seed)
        teacher = make_mlp(seed)
        teacher.fit(Xi, yi)

        # stage 2: pseudo-label the unlabelled pool where the teacher is confident
        y_student = yh_tr[:, i].copy()
        keep = annotated.copy()
        n_pos = n_neg = 0
        if unlabelled.any():
            prob = teacher.predict_proba(Xs_tr[unlabelled])[:, 1]
            idx = np.where(unlabelled)[0]
            conf_pos, conf_neg = idx[prob >= conf], idx[prob <= 1.0 - conf]
            y_student[conf_pos] = 1
            y_student[conf_neg] = 0
            keep[conf_pos] = True
            keep[conf_neg] = True
            n_pos, n_neg = len(conf_pos), len(conf_neg)

        # stage 3: student on human labels + confident pseudo-labels
        Xi, yi = oversample_positives(Xs_tr[keep], y_student[keep], seed=seed)
        student = make_mlp(seed)
        student.fit(Xi, yi)

        pred = student.predict(Xs_te)
        g = gold_te[:, i]
        tp = int(((pred == 1) & (g == 1)).sum())
        fn = int(((pred == 0) & (g == 1)).sum())
        out[k] = {"recall": tp / (tp + fn) if tp + fn else 0.0,
                  "support": int(g.sum())}
        stats[k] = {"pseudo_pos": n_pos, "pseudo_neg": n_neg,
                    "annotated": int(annotated.sum())}
    return out, stats


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm-replies", default=None, dest="vlm_replies",
                    help="a VLM replies file; adds a fourth arm trained on "
                         "that model's labels over the same pairs")
    ap.add_argument("--out", default="outputs/rq2_report.json")
    ap.add_argument("--table", default="outputs/tables/rq2.md")
    ap.add_argument("--seeds", default="42,43,44",
                    help="comma-separated seeds; results are averaged (the "
                         "human-trained model is sensitive to sampling noise, "
                         "so single-seed numbers over/under-state it)")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    cfg = load_config("configs/default.yaml")
    vlm = None
    if args.vlm_replies:
        vlm = load_vlm_labels(Path(args.vlm_replies))
        print(f"VLM labels: {len(vlm['rels'])} labelled pairs over "
              f"{len(vlm['images'])} images")
    print("building the feature matrix from the cache ...")
    if vlm is None:
        X, yh, ya, tr = load_matrix(cfg)
        yv = vseen = None
    else:
        X, yh, ya, tr, yv, vseen = load_matrix(cfg, vlm)
    print(f"pairs: {len(X)} (train {int(tr.sum())}, held-out {int((~tr).sum())})")

    gold_te = yh[~tr]

    # When a VLM arm is present every arm trains on exactly the pairs the VLM
    # was asked about. Otherwise the comparison would confound the label
    # source with how much data each source covers, which is a different
    # experiment from the one this runs.
    fit = tr if yv is None else (tr & vseen)
    if yv is not None:
        print(f"all arms restricted to the {int(fit.sum())} train pairs the "
              f"VLM covers ({fit.sum() / max(1, tr.sum()):.1%} of the split, "
              f"{len(vlm['images'])} images), so only the label source differs")

    results = {}
    for name, labels in (("human-trained", yh), ("auto-trained", ya)):
        per_seed = []
        for seed in seeds:
            print(f"training {name}, seed {seed} ...")
            per_seed.append(train_and_eval(X[fit], labels[fit], X[~tr], gold_te, seed=seed))
        agg = {}
        for k in PREDICATES:
            rs = [r[k]["recall"] for r in per_seed]
            agg[k] = {"recall": float(np.mean(rs)),
                      "recall_min": float(np.min(rs)), "recall_max": float(np.max(rs)),
                      "support": per_seed[0][k]["support"]}
        results[name] = agg

    # third arm: self-training on the sparse human labels (the rival remedy)
    per_seed, pstats = [], None
    for seed in seeds:
        print(f"training pseudo-labelled (self-training), seed {seed} ...")
        r, pstats = train_pseudo_and_eval(X[fit], yh[fit], X[~tr], gold_te, seed=seed)
        per_seed.append(r)
    agg = {}
    for k in PREDICATES:
        rs = [r[k]["recall"] for r in per_seed]
        agg[k] = {"recall": float(np.mean(rs)),
                  "recall_min": float(np.min(rs)), "recall_max": float(np.max(rs)),
                  "support": per_seed[0][k]["support"]}
    results["pseudo-labelled"] = agg
    results["_pseudo_label_counts"] = pstats

    # fourth arm: a vision-language model as the label source. It trains only
    # on pairs from images it was actually asked about, because a missing
    # reply is missing data rather than an assertion that nothing holds.
    if yv is not None:
        per_seed = []
        for seed in seeds:
            print(f"training vlm-trained, seed {seed} ...")
            per_seed.append(train_and_eval(X[fit], yv[fit], X[~tr], gold_te,
                                           seed=seed))
        agg = {}
        for k in PREDICATES:
            rs = [r[k]["recall"] for r in per_seed]
            agg[k] = {"recall": float(np.mean(rs)),
                      "recall_min": float(np.min(rs)),
                      "recall_max": float(np.max(rs)),
                      "support": per_seed[0][k]["support"]}
        results["vlm-trained"] = agg
        results["_vlm"] = {"images": len(vlm["images"]),
                           "labelled_pairs": len(vlm["rels"]),
                           "train_pairs_all_arms": int(fit.sum()),
                           "replies_file": args.vlm_replies}

    md = ["# RQ2 - downstream classifier: human vs automatic vs self-trained labels\n",
          f"Identical features, model, oversampling and split; averaged over "
          f"seeds {seeds}; evaluated against held-out human gold (groups 6-8). "
          "Only the label source differs. The pseudo-labelled arm self-trains "
          "on the human labels (teacher, confident pseudo-labels, student).\n",
          "| predicate | human-trained | pseudo-labelled | auto-trained | gold (held-out) |",
          "|---|---|---|---|---|"]
    hv, av, pv = [], [], []
    for k in PREDICATES:
        h = results["human-trained"][k]
        a = results["auto-trained"][k]
        p = results["pseudo-labelled"][k]
        hv.append(h["recall"]); av.append(a["recall"]); pv.append(p["recall"])
        md.append(f"| {k} | {h['recall']:.2f} ({h['recall_min']:.2f}-{h['recall_max']:.2f}) | "
                  f"{p['recall']:.2f} ({p['recall_min']:.2f}-{p['recall_max']:.2f}) | "
                  f"{a['recall']:.2f} ({a['recall_min']:.2f}-{a['recall_max']:.2f}) | {h['support']} |")
    md.append(f"| **mean** | **{np.mean(hv):.2f}** | **{np.mean(pv):.2f}** | "
              f"**{np.mean(av):.2f}** | |")

    Path(args.table).parent.mkdir(parents=True, exist_ok=True)
    Path(args.table).write_text("\n".join(md), encoding="utf-8")
    Path(args.out).write_text(json.dumps(results, indent=2),
                                               encoding="utf-8")
    print("\n".join(md))
    print("\nreport -> outputs/rq2_report.json ; table -> outputs/tables/rq2.md")


if __name__ == "__main__":
    main()
