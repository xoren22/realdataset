# realdataset

This repo has been reduced to one runnable experiment:

```bash
/home/kpetrosyan/miniconda3/envs/c/bin/python experiments/comparing_feature_sets_ridge.py
```

The script expects `data/RSSI_raw_data.csv`, `data/WiFi_RSSI_data.h5`, and the
existing cached R101 predictions under
`runs/comparing_feature_sets_ridge/r101_global_feature_cache/`.

It compares fixed-alpha ridge models only:

- 12 sparse/geometry features.
- 13 features: the same 12 plus `r101_pl_pred`.

Historical markdown reports were concatenated without rewriting into
`results_summaries.md`.
