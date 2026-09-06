# notebooks/

Empty. The project's notebooks are in **[`scripts/kaggle/`](../scripts/kaggle/)**,
next to the config and the generators that write them.

They are there because they are not exploratory: each one is a Chapter 6
experiment run on Kaggle's T4, and the executed copies keep their outputs as
the record of what was run.

| Notebook | What it produced |
|---|---|
| `executed_1_train_arms_seed42.ipynb` | both label arms trained at seed 42 |
| `executed_1b_eval_seed42.ipynb` | the seed-42 evaluation |
| `executed_2_seed_replication.ipynb` | seeds 43 and 44 |
| `executed_3_vlm_arm.ipynb` | the vision-language arm |
| `executed_4_reeval_group_slices.ipynb` | corrected zero-shot reference, per-group slices (§6.3.1) |
| `executed_5_auto085_refit.ipynb` | the support refit at 0.85 |
| `executed_6_all_arms_085.ipynb` | all arms at the shipped threshold |
| `template_seed_replication.ipynb` | the same run, outputs stripped |
| `template_reeval_group_slices.ipynb` | the same run, outputs stripped |
| `unrun_seed_power_10x.ipynb` | the ten-seed power analysis: written, deliberately not run |

`scripts/kaggle/README.md` gives the running order and the session
constraints. Training logs and per-epoch series are in
`outputs/sgg_benchmark/`.
