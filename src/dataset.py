"""Loader for the SpatialAwareRobotDataset (Wang et al., ACM MM 2025).

Parses the SGDET-Annotate JSON exports into a clean, model-agnostic structure
used by the fidelity (RQ1) and downstream (RQ2) studies. The on-disk schema
(verified against the real files — see docs/DATASET_NOTES.md):

  annotated_data/group_k/NNNNNN.json   per-image annotation
  annotated_data/group_k/labels.json        {name: id}   object vocabulary
  annotated_data/group_k/relationships.json {name: id}   predicate vocabulary
  img_data/group_k/NNNNNN.jpg          the image

Per-image JSON keys we use:
  width, height           original image size
  boxes_1024              [x_center, y_center, w, h] in a 1024-longest-side resize
  labels                  object label IDs, one per object
  relationships           [[subj_idx, obj_idx], ...] indices into the object list
  predicates              predicate ID per relationship (1-to-1 with relationships)

Boxes are returned NORMALISED to [0, 1] as (x1, y1, x2, y2), matching the
convention in src/predicates.py. group_4 ships without mapping files; since all
other groups share an identical vocabulary, we fall back to a reference map.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

# The seven target spatial predicates and their dataset IDs (see predicates.py).
from .predicates import PREDICATE_IDS

ID_TO_PREDICATE = {v: k for k, v in PREDICATE_IDS.items()}
TARGET_PREDICATE_IDS = set(PREDICATE_IDS.values())


@dataclass
class GtObject:
    label: str
    label_id: int
    box: tuple[float, float, float, float]  # normalised (x1, y1, x2, y2)


@dataclass
class GtRelation:
    subject: int          # index into objects
    object: int           # index into objects
    predicate_id: int
    predicate: str        # name, or "id_<n>" for non-target predicates


@dataclass
class GtImage:
    image_id: str         # e.g. "group_0/000000"
    image_path: Path
    width: int
    height: int
    objects: list[GtObject] = field(default_factory=list)
    relations: list[GtRelation] = field(default_factory=list)


def _norm_box_from_1024(box_xywh: list[int], width: int, height: int
                        ) -> tuple[float, float, float, float]:
    """Convert a boxes_1024 [xc, yc, w, h] entry to normalised (x1,y1,x2,y2).

    boxes_1024 live in a resize where the longest side is 1024px; dividing by the
    resized dimensions recovers the same normalised coordinates as the original
    image (verified: boxes_1024*0.625 == boxes_512*1.25 == original 640x480 px).
    """
    scale = 1024.0 / max(width, height)
    rw, rh = width * scale, height * scale
    xc, yc, w, h = box_xywh
    x1 = (xc - w / 2) / rw
    y1 = (yc - h / 2) / rh
    x2 = (xc + w / 2) / rw
    y2 = (yc + h / 2) / rh
    # clamp sub-pixel rounding overflow from the resize (max observed ~0.0007)
    clamp = lambda v: min(1.0, max(0.0, v))
    return (clamp(x1), clamp(y1), clamp(x2), clamp(y2))


class SpatialDataset:
    def __init__(self, root: str | Path, target_only: bool = True):
        """root points at the inner SpatialAwareRobotDataset-main folder
        (the one containing img_data/ and annotated_data/).

        target_only: if True, keep only the seven target spatial predicates;
        otherwise keep every relation, naming non-targets "id_<n>".
        """
        self.root = Path(root)
        self.ann_root = self.root / "annotated_data"
        self.img_root = self.root / "img_data"
        self.target_only = target_only
        self._ref_labels = self._load_first_map("labels.json")
        self._ref_rels = self._load_first_map("relationships.json")
        self._id_to_label = {v: k for k, v in self._ref_labels.items()}

    def _load_first_map(self, name: str) -> dict:
        for g in sorted(self.ann_root.glob("group_*")):
            for cand in (g / name, g / "output" / name):
                if cand.exists():
                    return json.loads(cand.read_text(encoding="utf-8"))
        raise FileNotFoundError(f"no {name} found under {self.ann_root}")

    def groups(self) -> list[str]:
        return sorted(p.name for p in self.ann_root.glob("group_*"))

    def _ann_files(self) -> Iterator[Path]:
        for g in self.groups():
            for p in sorted((self.ann_root / g).rglob("*.json")):
                if p.stem[0].isdigit():           # skip mapping files
                    yield p

    def load_image(self, ann_path: Path) -> Optional[GtImage]:
        d = json.loads(ann_path.read_text(encoding="utf-8"))
        boxes = d.get("boxes_1024")
        labels = d.get("labels")
        if not boxes or not labels:               # empty / unannotated frame
            return None

        group = ann_path.parent.name
        if group == "output":                     # group_8/output/...
            group = ann_path.parent.parent.name
        w, h = int(d["width"]), int(d["height"])

        objects = [
            GtObject(
                label=self._id_to_label.get(lid, f"label_{lid}"),
                label_id=int(lid),
                box=_norm_box_from_1024(bx, w, h),
            )
            for lid, bx in zip(labels, boxes)
        ]

        relations: list[GtRelation] = []
        pairs = d.get("relationships", [])
        preds = d.get("predicates", [])
        for (subj, obj), pid in zip(pairs, preds):
            if self.target_only and pid not in TARGET_PREDICATE_IDS:
                continue
            relations.append(GtRelation(
                subject=int(subj), object=int(obj), predicate_id=int(pid),
                predicate=ID_TO_PREDICATE.get(pid, f"id_{pid}"),
            ))

        return GtImage(
            image_id=f"{group}/{ann_path.stem}",
            image_path=self.img_root / group / f"{ann_path.stem}.jpg",
            width=w, height=h, objects=objects, relations=relations,
        )

    def __iter__(self) -> Iterator[GtImage]:
        for p in self._ann_files():
            img = self.load_image(p)
            if img is not None:
                yield img

    def __len__(self) -> int:
        return sum(1 for _ in self._ann_files())
