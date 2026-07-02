"""Export scene graphs in the dataset's exact formats: VG-style JSON, h5, YOLO.

The schema below is reconciled with the REAL SGDET-Annotate exports
(annotated_data/group_k/NNNNNN.json) so auto-labels are drop-in compatible with
the human labels for RQ1/RQ2. Verified fields (see docs/DATASET_NOTES.md):

  image-name, width, height
  boxes_1024 / boxes_512   [x_center, y_center, w, h] at 1024/512 longest-side resize
  attribute                N x 10 int array (we don't predict attributes -> zeros)
  labels                   object label IDs
  relationships            [[subj_idx, obj_idx], ...]
  predicates               predicate ID per relationship (1-to-1)

Object label IDs come from the dataset's labels.json; predicate IDs from
src.predicates.PREDICATE_IDS.
"""

from __future__ import annotations

import json
from pathlib import Path

from .predicates import PREDICATE_IDS, PairResult

# object vocabulary (annotated_data/*/labels.json — identical across groups)
LABEL_IDS = {"book": 1, "bottle": 2, "box": 3, "cube": 4, "human": 5, "remote": 6}


def _norm_xyxy_to_center_xywh(box, longest: int, width: int, height: int):
    """Normalised (x1,y1,x2,y2) -> [xc,yc,w,h] in a `longest`-side resize space."""
    scale = longest / max(width, height)
    rw, rh = width * scale, height * scale
    x1, y1, x2, y2 = box
    xc = (x1 + x2) / 2 * rw
    yc = (y1 + y2) / 2 * rh
    w = (x2 - x1) * rw
    h = (y2 - y1) * rh
    return [round(xc), round(yc), round(w), round(h)]


def _records(objects: list[dict], pairs: list[PairResult]):
    """Build the parallel arrays the dataset uses from our objects + pair results.

    `objects`: list of {label, box(normalised x1y1x2y2)} in a fixed order; the
    PairResult subject/object indices refer to this order.
    """
    labels = [LABEL_IDS.get(o["label"], 0) for o in objects]
    attribute = [[0] * 10 for _ in objects]            # no attribute prediction
    relationships, predicates = [], []
    for p in pairs:
        for pred in p.predicates:
            relationships.append([p.subject, p.object])
            predicates.append(PREDICATE_IDS[pred])
    return labels, attribute, relationships, predicates


def write_vg_json(image_name: str, width: int, height: int,
                  objects: list[dict], pairs: list[PairResult],
                  out_path: str | Path) -> dict:
    """Write one image's scene graph as dataset-compatible JSON. Returns the dict."""
    labels, attribute, relationships, predicates = _records(objects, pairs)
    scene = {
        "image-name": image_name,
        "width": width,
        "height": height,
        "boxes_1024": [_norm_xyxy_to_center_xywh(o["box"], 1024, width, height) for o in objects],
        "boxes_512": [_norm_xyxy_to_center_xywh(o["box"], 512, width, height) for o in objects],
        "attribute": attribute,
        "labels": labels,
        "relationships": relationships,
        "predicates": predicates,
        # side channel (not in the human schema): ambiguity flags for review
        "review_flags": [
            {"pair": [p.subject, p.object], "flags": p.flags} for p in pairs if p.flags
        ],
    }
    Path(out_path).write_text(json.dumps(scene, indent=2), encoding="utf-8")
    return scene


def write_h5(image_name: str, width: int, height: int,
             objects: list[dict], pairs: list[PairResult], out_path: str | Path) -> None:
    """Mirror the JSON arrays into HDF5: image-name/width/height as attributes,
    the arrays as int32 datasets — matching the dataset's h5 layout."""
    import h5py
    import numpy as np

    labels, attribute, relationships, predicates = _records(objects, pairs)
    boxes_1024 = [_norm_xyxy_to_center_xywh(o["box"], 1024, width, height) for o in objects]
    boxes_512 = [_norm_xyxy_to_center_xywh(o["box"], 512, width, height) for o in objects]
    with h5py.File(out_path, "w") as f:
        f.attrs["image-name"] = image_name
        f.attrs["width"] = np.int64(width)    # int64, matching the real exports
        f.attrs["height"] = np.int64(height)
        i32 = lambda a: np.asarray(a, dtype=np.int32)
        f.create_dataset("boxes_1024", data=i32(boxes_1024))
        f.create_dataset("boxes_512", data=i32(boxes_512))
        f.create_dataset("attribute", data=i32(attribute))
        f.create_dataset("labels", data=i32(labels))
        f.create_dataset("relationships", data=i32(relationships) if relationships else i32([]).reshape(0, 2))
        f.create_dataset("predicates", data=i32(predicates))


def write_yolo_txt(objects: list[dict], out_path: str | Path) -> None:
    """YOLO detection txt: one line per object `cls cx cy w h`, all normalised.

    Class IDs follow LABEL_IDS minus 1 (YOLO is 0-indexed); boxes are already
    normalised (x1,y1,x2,y2) so we just convert to centre form.
    """
    lines = []
    for o in objects:
        cls = LABEL_IDS.get(o["label"])
        if cls is None:
            # out-of-vocabulary object (e.g. the dataset's two nameless id=7
            # instances) — skipping beats silently mislabelling it as class 0
            continue
        x1, y1, x2, y2 = o["box"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        lines.append(f"{cls - 1} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    Path(out_path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
