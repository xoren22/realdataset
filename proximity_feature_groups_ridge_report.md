# Proximity Feature Groups Ridge Report

Generated from `experiments/proximity_feature_groups_ridge.py`.

This experiment adds three AP-to-receiver proximity indicators to the earlier
random-support ridge comparison and tests every non-empty combination of three
feature groups:

- `sparse12`: the 12 non-constant sparse engineered features.
- `prox3`: `ap_rx_same_point`, `ap_rx_lt_1m`, `ap_rx_lt_2m`.
- `r101_pl`: the cached decoded `r101_pl_pred` scalar.

I interpreted the requested `<0 (same point)` feature as exact same-point
distance, implemented as `distance_m <= 1e-6`. The other two thresholds are
strict `< 1.0 m` and `< 2.0 m`.

The protocol is unchanged from the ridge feature comparison:

- 10 random support draws, seeds `1000..1009`.
- 5 uniformly sampled support points per setup.
- 16 leave-one-AP-location-out folds.
- Query points only are scored.
- `StandardScaler` + ridge regression.
- Alpha sweep: `0.01, 0.1, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0`.
- Cached R101 outputs from `runs/comparing_feature_sets_ridge/r101_global_feature_cache`; no R101 inference.

## Best Alpha Results

Mean and standard deviation are across the 10 random support draws.

| model | feature_count | alpha | train_RMSE | val_RMSE | val_mae_mean |
|---|---|---|---|---|---|
| sparse12_plus_prox3_plus_r101_pl_ridge | 16 | 100.000 | 5.228 +/- 0.120 | 5.794 +/- 0.196 | 4.404 |
| sparse12_plus_r101_pl_ridge | 13 | 100.000 | 5.252 +/- 0.119 | 5.799 +/- 0.188 | 4.419 |
| sparse12_ridge | 12 | 100.000 | 5.608 +/- 0.115 | 6.238 +/- 0.207 | 4.812 |
| sparse12_plus_prox3_ridge | 15 | 100.000 | 5.589 +/- 0.116 | 6.238 +/- 0.213 | 4.799 |
| prox3_plus_r101_pl_ridge | 4 | 10.000 | 6.956 +/- 0.255 | 7.115 +/- 0.266 | 5.452 |
| r101_pl_ridge | 1 | 10.000 | 7.032 +/- 0.257 | 7.167 +/- 0.271 | 5.525 |
| prox3_ridge | 3 | 30.000 | 12.221 +/- 0.102 | 12.343 +/- 0.105 | 9.970 |

## Per-Draw Validation RMSE

Each column uses that model's selected best alpha above.

| support seed | sparse12_plus_prox3_plus_r101_pl_ridge | sparse12_plus_r101_pl_ridge | sparse12_ridge | sparse12_plus_prox3_ridge | prox3_plus_r101_pl_ridge | r101_pl_ridge | prox3_ridge |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 5.671 | 5.680 | 5.945 | 5.941 | 7.089 | 7.138 | 12.510 |
| 1001 | 6.054 | 6.038 | 6.609 | 6.629 | 6.962 | 7.018 | 12.384 |
| 1002 | 6.182 | 6.178 | 6.467 | 6.478 | 7.569 | 7.620 | 12.339 |
| 1003 | 5.908 | 5.923 | 6.393 | 6.388 | 7.409 | 7.484 | 12.426 |
| 1004 | 5.587 | 5.612 | 6.090 | 6.066 | 6.910 | 6.964 | 12.314 |
| 1005 | 5.717 | 5.710 | 6.011 | 6.025 | 6.791 | 6.827 | 12.160 |
| 1006 | 5.788 | 5.790 | 6.253 | 6.254 | 7.233 | 7.261 | 12.359 |
| 1007 | 5.719 | 5.714 | 6.131 | 6.136 | 7.047 | 7.121 | 12.214 |
| 1008 | 5.600 | 5.621 | 6.214 | 6.199 | 6.796 | 6.836 | 12.292 |
| 1009 | 5.716 | 5.723 | 6.265 | 6.266 | 7.347 | 7.397 | 12.434 |

## Main Deltas

Positive values mean the added feature group improved validation RMSE.

| comparison | paired RMSE gain mean | paired RMSE gain std | paired t-test p-value |
|---|---:|---:|---:|
| `sparse12` -> `sparse12 + prox3` | -0.001 | 0.013 | 0.896 |
| `sparse12` -> `sparse12 + r101_pl` | 0.439 | 0.119 | 9.7e-7 |
| `sparse12 + r101_pl` -> `sparse12 + prox3 + r101_pl` | 0.005 | 0.013 | 0.300 |
| `r101_pl` -> `prox3 + r101_pl` | 0.051 | 0.015 | 1.8e-6 |

## Interpretation

The proximity indicators do not meaningfully help once the sparse engineered
features are present. `sparse12 + prox3` is effectively tied with `sparse12`,
and `sparse12 + prox3 + r101_pl` is only `0.005 dB` better than
`sparse12 + r101_pl`, with a non-significant paired p-value of `0.300`.
`r101_pl_pred` remains the dominant useful extra scalar beyond the sparse
feature block.

The best model is the full `sparse12 + prox3 + r101_pl` group, but its gain over
`sparse12 + r101_pl` is too small to matter. The new proximity features mostly
encode a local near-AP exception already represented by `distance_m` and
`log_distance_m`, so they are redundant with the existing sparse feature block.

The combinations without `sparse12` are much worse, including `prox3 + r101_pl`.
So the sparse/interpolation feature block remains necessary; neither the new
near-distance flags nor r101's decoded prediction replaces it.

## Artifacts

- `experiments/proximity_feature_groups_ridge.py`
- `runs/proximity_feature_groups_ridge/proximity_feature_groups_ridge_fold_metrics.csv`
- `runs/proximity_feature_groups_ridge/proximity_feature_groups_ridge_by_seed_alpha_summary.csv`
- `runs/proximity_feature_groups_ridge/proximity_feature_groups_ridge_overall_by_alpha.csv`
- `runs/proximity_feature_groups_ridge/proximity_feature_groups_ridge_best_by_model.csv`
- `runs/proximity_feature_groups_ridge/manifest.json`
