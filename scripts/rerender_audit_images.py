"""Redraw an audit pack's images without touching its sample.

The first render anchored both tags to the top-left of their own box, so on any
item where the two boxes start near each other the blue OBJECT tag covered the
red SUBJECT tag. That is worst on support claims, which are about things in
contact, and those are the claims the pack exists to measure: an auditor who
cannot tell subject from object is being asked which way round a relation goes
without being shown.

Re-running make_audit_pack_v3.py would fix the drawing but redraws the sample
too, and a sample drawn again is a sample that has to be re-verified against its
key. This reads the key that already exists and redraws each item in place, so
audit_sheet_blind.csv and _key_do_not_share.csv are left byte-identical and the
ids stay aligned with the verdicts by construction.

    python scripts/rerender_audit_images.py --pack outputs/audit_v3
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline import load_config                    # noqa: E402
from scripts.make_audit_pack import render              # noqa: E402


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="outputs/audit_v3")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--no-anonymise", action="store_true",
                    help="leave faces unmasked; local inspection only")
    a = ap.parse_args()

    pack = Path(a.pack)
    sheet, key = pack / "audit_sheet_blind.csv", pack / "_key_do_not_share.csv"
    before = {p: digest(p) for p in (sheet, key)}

    root = Path(load_config(a.config)["dataset"]["root"])
    rows = list(csv.DictReader(open(key, encoding="utf-8")))

    masked, obscured = 0, []
    for r in rows:
        group, stem = r["image_id"].split("/")
        geo = json.loads((ROOT / f"outputs/geometry/{group}/{stem}.json")
                         .read_text(encoding="utf-8"))
        n, bad = render(int(r["id"]), root / "img_data" / group / f"{stem}.jpg",
                        geo, int(r["subj"]), int(r["obj"]), r["predicate"],
                        pack / "img" / f"{int(r['id']):03d}.png",
                        anonymise=not a.no_anonymise)
        masked += bool(n)
        if bad:
            obscured.append(int(r["id"]))

    print(f"  redrew {len(rows)} images in {pack}/img")
    if a.no_anonymise:
        print("    *** anonymisation DISABLED -- do not send these anywhere ***")
    else:
        print(f"    {masked} of {len(rows)} carried an annotated person, faces masked")
        print(f"    {len(obscured)} items had a mask land on a claim object"
              + (f": {obscured}" if obscured else ""))
    for p, d in before.items():
        same = digest(p) == d
        print(f"    {p.name:26} {'unchanged' if same else '*** CHANGED ***'} ({d})")
        if not same:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
