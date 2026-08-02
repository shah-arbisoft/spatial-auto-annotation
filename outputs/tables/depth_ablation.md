# A8 - Depth model ablation (Depth Anything v2 Small vs Base)

| predicate | Small | Base | delta |
|---|---|---|---|
| on | 0.879 | 0.873 | -0.006 |
| under | 0.813 | 0.810 | -0.003 |
| to the left of | 0.965 | 0.965 | +0.000 |
| to the right of | 0.985 | 0.985 | +0.000 |
| in front of | 0.640 | 0.641 | +0.001 |
| behind | 0.654 | 0.656 | +0.002 |
| near | 0.997 | 0.997 | +0.000 |
| **mean** | **0.848** | **0.847** | **-0.001** |

front/behind emit rate (commit vs abstain): Small 0.862, Base 0.863 (+0.001).

A 4x-larger depth model moves front/behind recall by +0.001/+0.002 and mean recall by -0.001. The depth-predicate limit is monocular ambiguity - two objects at a similar distance are inseparable by *any* monocular model - not the depth network's fidelity. This is why the fix that worked (the ground-plane fallback, A7) is a geometric cue, not a bigger perception model. The shipped tool keeps the Small variant: identical accuracy, Apache-2.0 licence, half the VRAM.
