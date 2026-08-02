# Recall with 95% cluster-bootstrap intervals

Resampling whole images with replacement (2000 draws, seed 42), which preserves the correlation between triplets that share an image.

| predicate | pooled recall (95% CI) | n | held-out recall (95% CI) | n |
|---|---|---|---|---|
| on | 0.879 (0.863-0.895) | 1465 | 0.922 (0.891-0.951) | 348 |
| under | 0.813 (0.790-0.836) | 1001 | 0.922 (0.879-0.960) | 192 |
| to the left of | 0.965 (0.954-0.975) | 972 | 0.953 (0.937-0.970) | 446 |
| to the right of | 0.985 (0.977-0.992) | 1174 | 0.985 (0.974-0.996) | 550 |
| in front of | 0.640 (0.602-0.678) | 2013 | 0.197 (0.147-0.255) | 609 |
| behind | 0.655 (0.615-0.692) | 1584 | 0.347 (0.281-0.417) | 580 |
| near | 0.997 (0.993-1.000) | 717 | 1.000 (1.000-1.000) | 93 |
| **mean** | **0.848** | | **0.761** | |