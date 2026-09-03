"""Verify every section cross-reference in the dissertation resolves.

The chapters refer to each other constantly (§4.9.6, Supplementary B, Chapter 6).
Moving material between chapters and supplements silently breaks those
pointers: LaTeX has no idea, because they are prose, not \\ref. This script
is the check that catches it.

    python scripts/check_refs.py
    python scripts/check_refs.py --map     # what each reference resolves to

Exits non-zero if any reference points at a heading that does not exist.

Existing is not the same as being right. Renumbering Chapter 9 to Chapter 8
turned a batch of correct \u00a78.1 pointers into \u00a73.12, and every one of them
still resolved, because \u00a73.12 exists too -- it is just the licensing section
rather than the summary of the dissertation. --map prints each distinct
reference beside the heading it lands on, which is the only cheap way to see
that a pointer is aimed at the wrong place.
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


def titles() -> dict[str, str]:
    """Section number -> its heading text, for the resolution map."""
    out: dict[str, str] = {}
    for f in SRC.glob("*.md"):
        for line in f.read_text(encoding="utf-8").split("\n"):
            m = re.match(r"^#{2,4}\s+(\d+\.\d+(?:\.\d+)?)\s+(.+?)\s*$", line)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def show_map() -> int:
    """Print every distinct section reference beside where it lands."""
    tit = titles()
    used: dict[str, set[str]] = {}
    for f in FILES:
        if not f.exists():
            continue
        for ref in re.findall(r"\u00a7\s?(\d+\.\d+(?:\.\d+)?)",
                              f.read_text(encoding="utf-8")):
            used.setdefault(ref, set()).add(f.name.replace(".md", ""))
        for ref in re.findall(r"Section\s+(\d+\.\d+(?:\.\d+)?)",
                              f.read_text(encoding="utf-8")):
            used.setdefault(ref, set()).add(f.name.replace(".md", ""))
    for ref in sorted(used, key=lambda r: [int(x) for x in r.split(".")]):
        where = ", ".join(sorted(s.replace("chapter", "ch")[:14]
                                 for s in used[ref]))
        print(f"  \u00a7{ref:<7} {tit.get(ref, '(MISSING)')[:46]:<48} <- {where}")
    print(f"\n  {len(used)} distinct section references")
    return 0


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
    if "--map" in sys.argv:
        return show_map()
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
