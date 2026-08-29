"""Guard against losing a fact while shortening prose.

Compares the current chapter files against a git revision and reports any
numeric value, citation or cross-reference that has disappeared. Rewriting for
length is safe only if the set of things being claimed is unchanged; this makes
that checkable instead of hoped for.

    python scripts/check_content_preserved.py [rev]
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISS = ROOT / "dissertation"

NUM = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?![\w])")
CITE = re.compile(r"\b([A-Z][A-Za-z\-']+)(?:\s+et al\.)?,?\s+(\d{4})[a-z]?\)")
XREF = re.compile(r"§(\d+\.\d+(?:\.\d+)?)|Appendix ([A-F](?:\.\d+)?)")


def tokens(text: str):
    # Markdown is hard-wrapped, so a citation can split across two lines and
    # the pattern then misses it. Flatten first: a reflowed paragraph must not
    # be reported as a lost reference.
    text = " ".join(text.split())
    nums = set(NUM.findall(text))
    cites = {f"{a} {b}" for a, b in CITE.findall(text)}
    xrefs = {a or b for a, b in XREF.findall(text)}
    return nums, cites, xrefs


def main() -> int:
    rev = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    files = sorted(DISS.glob("chapter*.md")) + [DISS / "abstract.md"]
    # A token that leaves a chapter for an appendix has been MOVED, which is the
    # intended operation when rebalancing against the template's chapter bands.
    # Without this the script called such a token "no longer present anywhere"
    # and every move looked like a loss.
    app = DISS / "appendices.md"
    an, ac, ax = tokens(app.read_text(encoding="utf-8")) if app.exists() else (set(), set(), set())
    total_lost = total_moved = 0
    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        old = subprocess.run(["git", "show", f"{rev}:{rel}"], cwd=ROOT,
                             capture_output=True, text=True, encoding="utf-8").stdout
        if not old:
            continue
        new = f.read_text(encoding="utf-8")
        on, oc, ox = tokens(old)
        nn, nc, nx = tokens(new)
        # split what left this chapter into "went to an appendix" and "gone"
        gone_n, gone_c, gone_x = on - nn, oc - nc, ox - nx
        moved_n, moved_c, moved_x = gone_n & an, gone_c & ac, gone_x & ax
        lost_n, lost_c, lost_x = gone_n - an, gone_c - ac, gone_x - ax
        total_moved += len(moved_n) + len(moved_c) + len(moved_x)
        if lost_n or lost_c or lost_x or moved_n or moved_c or moved_x:
            print(f"  {f.stem}")
            if moved_n or moved_c or moved_x:
                moved = sorted(moved_c | moved_x | moved_n)
                print(f"     -> appendix ({len(moved)}): {moved[:24]}")
            if lost_c:
                print(f"     citations gone: {sorted(lost_c)}")
            if lost_x:
                print(f"     cross-refs gone: {sorted(lost_x)}")
            if lost_n:
                print(f"     numbers gone ({len(lost_n)}): {sorted(lost_n)[:24]}")
        total_lost += len(lost_n) + len(lost_c) + len(lost_x)
    print(f"\n  moved to an appendix: {total_moved}")
    print(f"  tokens no longer present anywhere: {total_lost}")
    print("  (a number may legitimately go if the sentence quoting it went too;"
          " a citation or cross-reference going is almost always a mistake)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
