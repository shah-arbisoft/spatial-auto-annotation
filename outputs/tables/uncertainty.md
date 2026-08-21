# Recall with 95% cluster-bootstrap intervals

Resampling whole images with replacement (2000 draws, seed 42), which preserves the correlation between triplets that share an image.

| predicate | pooled recall (95% CI) | n | held-out recall (95% CI) | n |
|---|---|---|---|---|
| on | 0.812 (0.794-0.830) | 1465 | 0.853 (0.813-0.890) | 348 |
| under | 0.747 (0.723-0.771) | 1001 | 0.823 (0.768-0.875) | 192 |
| to the left of | 0.965 (0.954-0.975) | 972 | 0.953 (0.937-0.970) | 446 |
| to the right of | 0.985 (0.977-0.992) | 1174 | 0.985 (0.974-0.996) | 550 |
| in front of | 0.696 (0.657-0.736) | 2013 | 0.199 (0.148-0.257) | 609 |
| behind | 0.711 (0.672-0.748) | 1584 | 0.369 (0.300-0.443) | 580 |
| near | 0.997 (0.993-1.000) | 717 | 1.000 (1.000-1.000) | 93 |
| **mean** | **0.845** | | **0.740** | |