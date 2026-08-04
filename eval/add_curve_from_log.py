"""Add an arm's per-epoch validation curve to outputs/sgg_benchmark/curves.json.

The first benchmark run parsed its own training logs into curves.json, which
is what Chapter 6's saturation figure draws from. A later arm trained in a
separate notebook arrives as a log file instead, so this recovers the same
series from it: one validation mR per epoch, in order, plus the best epoch.

Reads only the log, writes only the named arm's entry, and refuses to
overwrite an existing one unless asked, so re-running cannot quietly replace
a curve that came from somewhere else.

    python eval/add_curve_from_log.py \
        --log outputs/sgg_benchmark/seeds/checkpoints/spatial/react_vlm_s42/log.txt \
        --arm react_vlm
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CURVES = Path("outputs/sgg_benchmark/curves.json")
VAL_MR = re.compile(r"Average validation Result for mR:\s*([\d.]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--arm", required=True,
                    help="key to write, e.g. react_vlm")
    ap.add_argument("--curves", default=str(CURVES))
    ap.add_argument("--force", action="store_true",
                    help="replace the arm if it already exists")
    args = ap.parse_args()

    text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    series = [float(x) for x in VAL_MR.findall(text)]
    if not series:
        raise SystemExit(f"no validation mR lines in {args.log}")

    out = Path(args.curves)
    data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    if args.arm in data and not args.force:
        raise SystemExit(f"{args.arm} already present in {out}; pass --force "
                         f"to replace it")

    best = max(series)
    data[args.arm] = {"val_mr_per_epoch": series,
                      "best_epoch": series.index(best) + 1,
                      "best_val_mr": best,
                      "source_log": args.log}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")

    lens = {k: len(v["val_mr_per_epoch"]) for k, v in data.items()
            if isinstance(v, dict) and "val_mr_per_epoch" in v}
    print(f"{args.arm}: {len(series)} epochs, best {best:.4f} at epoch "
          f"{series.index(best) + 1}")
    print(f"arms now in {out}: {lens}")
    if len(set(lens.values())) > 1:
        print("NOTE: the arms ran different numbers of epochs; the figure "
              "plots each to its own length, which is correct but worth "
              "saying in any caption that compares their shapes.")


if __name__ == "__main__":
    main()
