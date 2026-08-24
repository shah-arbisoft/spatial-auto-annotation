"""Say whether a PDF was built from the current source.

    python scripts/pdf_is_current.py path/to/whatever.pdf
    python scripts/pdf_is_current.py ~/Downloads/*.pdf

Every build stamps its commit and time into the PDF's Subject field, so a
downloaded copy can be told apart from a fresh one. Without that, they look
identical, and reviewing an old one produces reports about defects that were
fixed days earlier.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def head() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__.strip())
        return 2
    try:
        import fitz
    except ImportError:
        print("  needs pymupdf")
        return 2

    now = head()
    print(f"  current source is at {now}\n")
    stale = 0
    for a in argv:
        p = Path(a).expanduser()
        if not p.exists():
            print(f"  {p.name:26} not found")
            continue
        try:
            subj = (fitz.open(p).metadata or {}).get("subject") or ""
        except Exception as e:
            print(f"  {p.name:26} unreadable ({e})")
            continue
        if not subj.startswith("build "):
            print(f"  {p.name:26} UNSTAMPED — built before this check existed,"
                  " so it predates 2026-08-24")
            stale += 1
            continue
        stamped = subj.split()[1].split("+")[0]
        ok = stamped == now
        if not ok:
            stale += 1
        print(f"  {p.name:26} {'CURRENT' if ok else 'STALE  '}  ({subj})")

    if stale:
        print(f"\n  {stale} file(s) are not the current build. Rebuild, or"
              " re-download from Overleaf, before reviewing.")
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
