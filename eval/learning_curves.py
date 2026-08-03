"""Per-epoch learning curves for every label source, including a VLM.

Chapter 6's training curves come from the SGG benchmark, which needs a GPU and
a Kaggle run, so a vision-language arm cannot be added to them here. This does
the same thing for the RQ2 classifier, which trains locally in minutes: one
curve per label source, showing held-out recall against human gold as training
proceeds.

What the shape shows is not accuracy so much as what each source has to teach.
A source that saturates early and then declines is one the model has finished
extracting signal from; a source that keeps climbing has more in it.

Every arm shares the features, the architecture, the seed, the oversampling
and the training pairs, so the curves differ only by who supplied the labels.

    python eval/learning_curves.py --vlm-replies outputs/vlm_pilot/replies_train_f35.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import load_config
from src.predicates import PREDICATES
from eval.classifier import oversample_positives
from eval.downstream import load_matrix, load_vlm_labels


def curve_for(X_tr, y_tr, X_te, gold_te, epochs, seed):
    """Mean held-out recall after each epoch, averaged over the predicates.

    warm_start with max_iter=1 advances the same optimiser one epoch at a
    time, so this is the identical training run the other scripts do, merely
    observed while it happens rather than only at the end.
    """
    from sklearn.neural_network import MLPClassifier      # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler      # noqa: PLC0415
    import warnings
    from sklearn.exceptions import ConvergenceWarning     # noqa: PLC0415

    scaler = StandardScaler().fit(X_tr)
    Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)

    per_pred = {}
    for i, k in enumerate(PREDICATES):
        Xi, yi = oversample_positives(Xs_tr, y_tr[:, i], seed=seed)
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                            max_iter=1, warm_start=True, random_state=seed,
                            tol=1e-4)
        g = gold_te[:, i]
        rec = []
        with warnings.catch_warnings():
            # one epoch at a time never "converges"; the warning is expected
            warnings.simplefilter("ignore", ConvergenceWarning)
            for _ in range(epochs):
                if len(np.unique(yi)) < 2:
                    rec.append(0.0)
                    continue
                clf.fit(Xi, yi)
                pred = clf.predict(Xs_te)
                tp = int(((pred == 1) & (g == 1)).sum())
                fn = int(((pred == 0) & (g == 1)).sum())
                rec.append(tp / (tp + fn) if tp + fn else 0.0)
        per_pred[k] = rec
    mean = [float(np.mean([per_pred[k][e] for k in PREDICATES]))
            for e in range(epochs)]
    return {"mean": mean, "per_predicate": per_pred}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm-replies", default=None, dest="vlm_replies")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/learning_curves.json")
    args = ap.parse_args()

    cfg = load_config("configs/default.yaml")
    vlm = load_vlm_labels(Path(args.vlm_replies)) if args.vlm_replies else None
    if vlm:
        print(f"VLM labels: {len(vlm['rels'])} pairs over {len(vlm['images'])} images")

    print("building the feature matrix ...")
    if vlm is None:
        X, yh, ya, tr = load_matrix(cfg)
        yv = vseen = None
    else:
        X, yh, ya, tr, yv, vseen = load_matrix(cfg, vlm)

    # identical to downstream.py: with a VLM arm present, every arm trains on
    # exactly the pairs the VLM covers
    fit = tr if yv is None else (tr & vseen)
    if fit.sum() == 0:
        sys.exit("no training pairs: the VLM replies cover none of the "
                 "training groups (0-5). Label training images first, e.g. "
                 "run_vlm_pilot.py --make --groups group_0,...,group_5")
    gold_te = yh[~tr]
    print(f"train pairs {int(fit.sum())}, held-out {int((~tr).sum())}, "
          f"{args.epochs} epochs")

    sources = [("human", yh), ("pipeline", ya)]
    if yv is not None:
        sources.append(("vlm", yv))

    out = {"epochs": args.epochs, "seed": args.seed,
           "train_pairs": int(fit.sum()), "curves": {}}
    for name, labels in sources:
        print(f"  {name} ...", flush=True)
        out["curves"][name] = curve_for(X[fit], labels[fit], X[~tr], gold_te,
                                        args.epochs, args.seed)
    if vlm:
        out["vlm_images"] = len(vlm["images"])
        out["vlm_replies"] = args.vlm_replies

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nfinal mean recall after {args.epochs} epochs:")
    for name in out["curves"]:
        c = out["curves"][name]["mean"]
        print(f"  {name:9} {c[-1]:.3f}   (best {max(c):.3f} at epoch "
              f"{c.index(max(c)) + 1})")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
