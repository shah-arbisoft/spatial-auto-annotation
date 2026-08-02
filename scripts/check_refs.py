"""Verify every section cross-reference in the dissertation resolves.

The chapters refer to each other constantly (§4.9.6, Appendix B, Chapter 6).
Moving material between chapters and appendices silently breaks those
pointers: LaTeX has no idea, because they are prose, not \\ref. This script
is the check that catches it.

    python scripts/check_refs.py

Exits non-zero if any reference points at a heading that does not exist.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "dissertation"

FILES = sorted(SRC.glob("chapter*.md")) + [
    SRC / "abstract.md", SRC / "appendices.md"]


def headings() -> tuple[set[str], set[str]]:
    """Every section number and every appendix letter that actually exists."""
    secs: set[str] = set()
    apps: set[str] = set()
    for f in SRC.glob("*.md"):
        for line in f.read_text(encoding="utf-8").split("\n"):
            m = re.match(r"^#{2,4}\s+(\d+\.\d+(?:\.\d+)?)\s", line)
            if m:
                secs.add(m.group(1))
            m = re.match(r"^##\s+Appendix\s+([A-Z])\s*[:.]", line)
            if m:
                apps.add(m.group(1))
    return secs, apps


def main() -> int:
    secs, apps = headings()
    chapters = {str(n) for n in range(1, 10)}
    bad: list[str] = []

    for f in FILES:
        if not f.exists():
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").split("\n"), 1):
            for ref in re.findall(r"§\s?(\d+\.\d+(?:\.\d+)?)", line):
                if ref not in secs:
                    bad.append(f"{f.name}:{i}  §{ref} -> no such section")
            for ref in re.findall(r"Appendix\s+([A-Z])\b", line):
                if ref not in apps:
                    bad.append(f"{f.name}:{i}  Appendix {ref} -> no such appendix")
            for ref in re.findall(r"Chapters?\s+(\d)\b", line):
                if ref not in chapters:
                    bad.append(f"{f.name}:{i}  Chapter {ref} -> no such chapter")

    print(f"sections defined : {len(secs)}")
    print(f"appendices       : {', '.join(sorted(apps)) or '(none)'}")
    if bad:
        print(f"\nBROKEN REFERENCES ({len(bad)}):")
        for b in bad:
            print("  " + b)
        return 1
    print("\nall cross-references resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
