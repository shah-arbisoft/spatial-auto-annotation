# data/ (gitignored)

Place the dataset and the exact SGDET-Annotate label files here once the
supervisor provides them:

  data/images/                 # ~900 cleaned RGB images
  data/labels_vg.json          # human Visual Genome scene graphs (ground truth)
  data/labels_yolo/            # YOLO txt per image
  data/labels.h5               # h5 export

These are the ground truth for RQ1 and the human-label training set for RQ2.
Match the auto-annotator output to these formats byte-for-byte (see src/writers.py).
Dataset licence: CC-BY 4.0. Repo: https://github.com/PengPaulWang/SpatialAwareRobotDataset
