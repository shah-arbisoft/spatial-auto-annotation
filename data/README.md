# data/

**The dataset is not stored in this repository, and nothing is written here.**

It is released separately under CC-BY 4.0 by the group that collected it, and
this project reads it in place rather than copying it:

  https://github.com/PengPaulWang/SpatialAwareRobotDataset

Point `dataset.root` in `configs/default.yaml` at the inner folder holding
`img_data/` and `annotated_data/`:

```yaml
dataset:
  root: ../SpatialAwareRobotDataset-main/SpatialAwareRobotDataset-main
```

That folder holds 884 released images across nine annotator groups:

```
img_data/group_<0-8>/<frame>.jpg          the images
annotated_data/group_<0-8>/<frame>.json   the human relationship labels
```

The fidelity study of Chapter 4 is measured on the 836 of those that carry
relationship annotations in the seven predicates the project targets.

Nothing is preprocessed on disk. 883 of the 884 images are stored
180-degree rotated with an EXIF orientation tag, and the loader corrects that
in memory (`src/dataset.py`), so no derived copy exists that could drift from
the release. The pipeline's outputs go to `outputs/`, and the writers that
reproduce the dataset's own annotation formats are in `src/writers.py`.

This directory stays empty: `.gitignore` excludes its contents and keeps only
this file.
