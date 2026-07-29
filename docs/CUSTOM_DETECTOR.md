# Using your own object detector

The annotator is built so the detector is the one piece you are expected to
replace. Everything downstream, segmentation, depth, the seven geometric
rules, the writers, is indifferent to where the boxes came from.

## Why this works

The rule layer (`src/predicates.py`) imports nothing but `numpy`. It never
receives an image. The pipeline entry point takes boxes as an argument:

```python
annotate_image(image_rgb, boxes_px, labels, segmenter, depther, cfg)
```

So a detector is not wired into the pipeline; it is simply what you call
first. Chapter 4's measurement follows from this: with the boxes held fixed,
the relation layer scores the same (0.85) whether the boxes came from a
detector or from ground truth, so detector quality and relation quality are
separable, and improving the detector improves the whole system without
touching a rule.

## The contract

One method, three keys:

```python
class MyDetector:
    def detect(self, image_rgb):          # RGB uint8 (H, W, 3)
        return [
            {"box_px": [x1, y1, x2, y2],  # pixels, top-left origin
             "label": "book",             # a dataset class name
             "score": 0.91},              # confidence in [0, 1]
        ]
```

`box_px` must be in the same upright frame as the image you were handed.
(The dataset's own images are stored with a 180° EXIF rotation; load them
through `src.dataset.load_rgb`, which corrects it, and the frames agree.)

That is the whole interface. `src/detectors.py` declares it as a
`typing.Protocol`, so an editor or `mypy` will check your class against it,
but nothing requires you to inherit from anything.

## Three detectors ship with the project

```yaml
# configs/default.yaml
detector:
  kind: grounding_dino        # open-vocabulary, no training needed
  threshold: 0.30
```

```yaml
detector:
  kind: yolo                  # any ultralytics checkpoint
  weights: ./checkpoints/yolov10m_spatial.pt
  threshold: 0.25
  class_map: {person: human}  # only if the names differ
```

```yaml
detector:
  kind: json                  # detections computed elsewhere
  root: ./outputs/my_detector
```

Then:

```python
from src.detectors import from_config
detector = from_config(cfg)
```

The `json` kind is the escape hatch: run any detector at all, in any
language, write one JSON file per image, and the pipeline consumes it
without knowing what produced it.

## Worked example: a custom detector

```python
import numpy as np
from src.dataset import load_rgb
from src.pipeline import load_config, annotate_image
from src.segment import Segmenter
from src.depth import DepthEstimator

class MyDetector:
    """Wraps whatever you already have."""
    def __init__(self, model):
        self.model = model

    def detect(self, image_rgb):
        out = []
        for box, cls, conf in self.model.run(image_rgb):   # your API
            out.append({"box_px": list(box), "label": cls, "score": float(conf)})
        return out

cfg = load_config("configs/default.yaml")
segmenter = Segmenter(cfg["segmentation"]["hf_model"]).load()
depther = DepthEstimator(cfg["depth"]["hf_model"]).load()
detector = MyDetector(my_model)

image = load_rgb("photo.jpg")
dets = detector.detect(image)
objs, pairs, _, _ = annotate_image(
    image,
    [d["box_px"] for d in dets],
    [d["label"] for d in dets],
    segmenter, depther, cfg,
)
for p in pairs:
    for predicate in p.predicates:
        print(objs[p.subject].label, predicate, objs[p.object].label)
```

## Two things to check when you swap detector

**Class names.** The rules are mostly class-agnostic, with one exception:
the support guard suppresses `on`/`under` when either object is a person,
because the annotators never labelled person-support (0 of 2,466 gold
triplets). That guard keys on the literal label `human`, set in
`predicates.no_support_classes`. If your detector calls it `person`, either
map the name or change the config; otherwise the guard silently stops
firing.

**Box tightness.** Thresholds such as the contact fraction (0.60) and the
ground-plane band (0.005) were fitted against this dataset's hand-drawn
boxes. A detector with systematically looser or tighter boxes shifts those
geometries. The constants are all in one config file and the geometry cache
makes re-fitting cheap: `scripts/reannotate_from_cache.py` re-evaluates the
whole dataset in about twenty seconds without a GPU. What transfers between
detectors is the calibration *procedure* (fit on some annotators, validate
on held-out ones), not the constants.

## The other two components

Segmentation and depth are swappable on the same principle, by duck typing
rather than a declared protocol:

```python
class Segmenter:     masks_from_boxes(image, boxes) -> list[np.ndarray]
class DepthEstimator: estimate(pil_image) -> np.ndarray   # smaller = nearer
```

The depth convention matters: the pipeline expects **smaller values to mean
nearer**. Depth Anything emits the opposite and the wrapper inverts it. A
replacement that gets this backwards will not error; it will silently
reverse every front/behind label, which is exactly the bug that cost this
project a day (§3.5).
