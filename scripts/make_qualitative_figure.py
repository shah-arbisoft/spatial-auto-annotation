"""Two annotated scenes for the results chapter: one the rules get right, one they do not.

The dissertation reports the pipeline entirely in tables, which is fair to the
numbers and unfair to the reader: nothing in it shows what a labelled scene
actually looks like. This builds that figure from two frames already rendered
by `scripts/annotate_image.py`, one from a calibration group and one from a
held-out group whose front/behind labels §4.5 convicts of an inverted
convention.

Faces are pixelated first, from the dataset's own `human` boxes, because §8.1
commits to data minimisation on everything the project republishes and a figure
in a submitted document is republication. The regions come from `src.anonymise`,
so this figure, the audit pack and the validation website all apply one rule.

Each panel is cropped to the union of its annotated boxes plus a margin. The
frames are three-quarters empty carpet, and at the width a page allows an
uncropped pair renders the class labels at about four points, which is a figure
nobody can read. Panels are then stacked, since two 640-wide frames side by side
halve the scale again.

    python scripts/make_qualitative_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.anonymise import face_regions, pixelate  # noqa: E402

PANELS = [
    ("outputs/sample_result_group0_000018.png", "outputs/geometry/group_0/000018.json",
     "(a) group 0 (calibration): every relation the rules emit is correct."),
    ("outputs/sample_failure_group2_000205.png", "outputs/geometry/group_2/000205.json",
     "(b) group 2 (held out): same rules, annotator inverted front/behind."),
]

MARGIN = 26          # px of context to keep around the annotated content
STRIP = 24           # px of caption strip under each panel
GAP = 12


def _font(size: int):
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _content_box(size, geo):
    """Union of the annotated boxes, in pixels, padded and clipped to the frame."""
    W, H = size
    xs = [o["box"][0] for o in geo] + [o["box"][2] for o in geo]
    ys = [o["box"][1] for o in geo] + [o["box"][3] for o in geo]
    return (max(0, int(min(xs) * W) - MARGIN),
            max(0, int(min(ys) * H) - MARGIN),
            min(W, int(max(xs) * W) + MARGIN),
            min(H, int(max(ys) * H) + MARGIN))


def main() -> int:
    panels, total = [], 0
    for png, geo_path, caption in PANELS:
        p = ROOT / png
        if not p.exists():
            print(f"missing {png}; run scripts/annotate_image.py first")
            return 1
        im = Image.open(p).convert("RGB")
        geo = json.loads((ROOT / geo_path).read_text(encoding="utf-8"))
        n = pixelate(im, face_regions(im.size, geo))
        total += n
        crop = _content_box(im.size, geo)
        im = im.crop(crop)
        print(f"{png}: {n} face region(s) pixelated, cropped to {im.width}x{im.height}")
        panels.append((im, caption))

    font = _font(13)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    text_w = max(probe.textbbox((0, 0), c, font=font)[2] for _, c in panels)
    w = max(max(i.width for i, _ in panels), text_w + 6)
    h = sum(i.height + STRIP for i, _ in panels) + GAP
    canvas = Image.new("RGB", (w, h), "white")
    dr = ImageDraw.Draw(canvas)

    y = 0
    for im, caption in panels:
        canvas.paste(im, ((w - im.width) // 2, y))
        dr.text((3, y + im.height + 5), caption, fill="black", font=font)
        y += im.height + STRIP + GAP

    out = ROOT / "outputs" / "figures" / "qualitative_examples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"figure -> {out.relative_to(ROOT)}  ({canvas.width}x{canvas.height}, "
          f"{total} face regions pixelated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
