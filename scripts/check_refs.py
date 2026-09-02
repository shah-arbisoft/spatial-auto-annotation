"""Verify every section cross-reference in the dissertation resolves.

The chapters refer to each other constantly (§4.9.6, Supplementary B, Chapter 6).
Moving material between chapters and supplements silently breaks those
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
    """Every section number and every supplement letter that actually exists."""
    secs: set[str] = set()
    apps: set[str] = set()
    for f in SRC.glob("*.md"):
        for line in f.read_text(encoding="utf-8").split("\n"):
            m = re.match(r"^#{2,4}\s+(\d+\.\d+(?:\.\d+)?)\s", line)
            if m:
                secs.add(m.group(1))
            m = re.match(r"^##\s+Supplementary\s+([A-Z])\s*[:.]", line)
            if m:
                apps.add(m.group(1))
    return secs, apps


def renumbering() -> dict[str, str]:
    """Map each written section number to the one LaTeX will print.

    build_latex.py strips "4.14" from "## 4.14 Temporal redundancy" and emits a
    bare \\section, so LaTeX counts the sections itself. Delete a section and
    every later one shifts up, leaving the prose pointing at numbers that now
    belong to different sections -- and the heading still exists, so the
    existence check above passes. Removing 4.12, 4.13 and 4.15 once left 36
    references pointing one or two sections off in the PDF.
    """
    out: dict[str, str] = {}
    for f in sorted(SRC.glob("chapter*.md")):
        m = re.match(r"chapter(\d+)_", f.name)
        if not m:
            continue
        ch, sec, sub = m.group(1), 0, 0
        body = re.sub(r"```.*?```", " ", f.read_text(encoding="utf-8"), flags=re.S)
        for line in body.split("\n"):
            h2 = re.match(r"^##\s+(\d+\.\d+)\s", line)
            h3 = re.match(r"^###\s+(\d+\.\d+\.\d+)\s", line)
            if h2:
                sec += 1; sub = 0
                out[h2.group(1)] = f"{ch}.{sec}"
            elif h3:
                sub += 1
                out[h3.group(1)] = f"{ch}.{sec}.{sub}"
    return out


def main() -> int:
    secs, apps = headings()
    printed = renumbering()
    chapters = {str(n) for n in range(1, 10)}
    bad: list[str] = []

    for written, prints_as in sorted(printed.items()):
        if written != prints_as:
            bad.append(f"heading {written} will print as {prints_as}; "
                       f"renumber it and its references, or the PDF's "
                       f"cross-references point at the wrong section")

    for f in FILES:
        if not f.exists():
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").split("\n"), 1):
            for ref in re.findall(r"§\s?(\d+\.\d+(?:\.\d+)?)", line):
                if ref not in secs:
                    bad.append(f"{f.name}:{i}  §{ref} -> no such section")
            for ref in re.findall(r"Supplementary\s+([A-Z])\b", line):
                if ref not in apps:
                    bad.append(f"{f.name}:{i}  Supplementary {ref} -> no such supplement")
            for ref in re.findall(r"Chapters?\s+(\d)\b", line):
                if ref not in chapters:
                    bad.append(f"{f.name}:{i}  Chapter {ref} -> no such chapter")

    print(f"sections defined : {len(secs)}")
    print(f"supplements      : {', '.join(sorted(apps)) or '(none)'}")
    if bad:
        print(f"\nBROKEN REFERENCES ({len(bad)}):")
        for b in bad:
            print("  " + b)
        return 1
    print("\nall cross-references resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
