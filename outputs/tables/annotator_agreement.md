# Annotator heterogeneity and estimated human-human agreement

The tool is deterministic, so it is the same labeller for every annotator group; variation in its agreement is variation in the annotators.

| annotator group | tool agreement | gold triplets | note |
|---|---|---|---|
| group_0 | 0.851 | 1949 |  |
| group_1 | 0.899 | 1489 |  |
| group_2 | 0.921 | 1192 |  |
| group_3 | 0.880 | 435 |  |
| group_4 | 0.860 | 356 |  |
| group_5 | 0.933 | 687 |  |
| group_6 | 0.556 | 970 | convention-inverted |
| group_7 | 0.899 | 796 |  |
| group_8 | 0.566 | 1052 | convention-inverted |

Consistent annotators (n=7): agreement spans 0.851-0.933 (spread 0.082, sd 0.028, mean 0.892). A single fixed labeller varying this much across annotators is a direct measure of how much the annotators differ from one another.

Frechet bounds over all 21 annotator pairs place annotator-to-annotator agreement in **[0.78, 0.96]** on average, under the batch-exchangeability assumption stated in the script. The tool's own mean agreement, 0.892, lies inside that interval.