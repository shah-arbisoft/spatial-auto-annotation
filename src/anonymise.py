"""Face anonymisation for anything the project shows outside itself.

Chapter 8 records that some frames contain identifiable people, which makes
them personal data under the Data Protection Act 2018, and commits to data
minimisation on everything republished. The validation website has applied this
since it went live. Anything else that leaves the machine -- an audit pack sent
to a third-party API, a figure in the dissertation -- is the same disclosure and
needs the same treatment, so the rule lives here rather than in one consumer.

The region is derived from the dataset's own `human` boxes rather than from a
face detector: the annotators marked every person, so the top of an annotated
box is where a standing person's head is, and no detector can fail to fire on a
face this way. It is the same rule and the same geometry the website uses.
"""

from __future__ import annotations

from PIL import Image

HEAD_FRACTION = 0.30            # top of a standing person's box
PAD_X, PAD_Y = 5.0, 8.0         # a little slack, since the box bounds the body


def face_regions(size: tuple[int, int], geo, label: str = "human"):
    """Pixel regions to anonymise: the head area of every annotated person."""
    W, H = size
    regs = []
    for o in geo:
        if o.get("label") == label:
            x1, y1, x2, y2 = o["box"]
            head = HEAD_FRACTION * (y2 - y1)
            regs.append((x1 * W - PAD_X, y1 * H - PAD_Y,
                         x2 * W + PAD_X, (y1 + head) * H))
    return regs


def overlap_frac(a, b) -> float:
    """Intersection area as a fraction of box a's area."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return (ix * iy / area) if area > 0 else 0.0


def pixelate(img: Image.Image, regions) -> int:
    """Mosaic each [x1, y1, x2, y2] pixel region in place. Returns the count."""
    n = 0
    for x1, y1, x2, y2 in regions:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(img.width, int(x2)), min(img.height, int(y2))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        crop = img.crop((x1, y1, x2, y2))
        w, h = crop.size
        cell = max(2, min(w, h) // 6)          # ~6 mosaic cells across the short side
        small = crop.resize((max(1, w // cell), max(1, h // cell)), Image.BILINEAR)
        img.paste(small.resize((w, h), Image.NEAREST), (x1, y1))
        n += 1
    return n
