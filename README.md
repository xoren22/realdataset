# realdataset

Minimal fixed-alpha ridge comparison for the Zenodo 15791300 indoor RSSI data.

The repo now has one runnable experiment script:

```bash
/home/kpetrosyan/miniconda3/envs/c/bin/python experiments/comparing_feature_sets_ridge.py
```

It refits AP-held-out ridge regressions over 10 random support draws and
compares only:

- `sparse12_ridge`: 12 sparse/geometry features.
- `sparse12_plus_r101_pl_ridge`: those 12 features plus `r101_pl_pred`.

Ridge alpha is fixed at `0.1`; there is no alpha grid search.

Historical result summaries were concatenated verbatim into
`results_summaries.md`.
