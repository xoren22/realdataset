# Ridge Features Comparison

Generated from `experiments/comparing_feature_sets_ridge.py`.

This experiment compares four ridge-regression feature sets under the same
sampling-assisted AP-held-out protocol. Each result is averaged over 10 random
sparse-support draws. For every draw, each setup gets 5 support points sampled
uniformly without replacement; only non-support query points are scored.

All RMSE/MAE values are in dBm.

## Protocol

- Outer split: leave-one-AP-location-out, 16 folds.
- Random support draws: seeds `1000..1009`.
- Support count: 5 points per setup per draw.
- Ridge model: `StandardScaler` + `Ridge(alpha=0.01, solver="svd")`.
- Train loss: mean fold train RMSE, then averaged over the 10 support draws.
- Validation loss: pooled query RMSE per support draw, then mean/std over the 10 draws.
- R101 cache: `runs/comparing_feature_sets_ridge/r101_global_feature_cache/`.

The R101 cache stores:

- `r101_pl_pred`: decoded R101 prediction sampled at each measurement point.
- `encoder2048`: global pooled final ResNet101 encoder feature vector. It is
  `model.unet.encoder(x)[-1]`, shape `[batch, 2048, 8, 8]`, adaptive-average
  pooled to `[batch, 2048]`. These are global map/setup features, not local
  point decoder features.

## Feature Sets

`SPARSE_COLS` in the repo has 13 columns, but `support_count` is constant in
this protocol, so this experiment drops it and uses the 12 non-constant sparse
features:

`ap_x_m, ap_y_m, mx_m, my_m, distance_m, log_distance_m, same_room, support_mean_rssi, support_std_rssi, nearest_support_dist_m, nearest_support_rssi, idw_support_rssi`

The compared ridge models are:

| model | feature count | features |
|---|---:|---|
| `sparse12_ridge` | 12 | sparse engineered features only |
| `sparse12_plus_r101_pl_ridge` | 13 | sparse12 + `r101_pl_pred` |
| `sparse12_plus_encoder2048_ridge` | 2060 | sparse12 + encoder2048 |
| `sparse12_plus_r101_pl_encoder2048_ridge` | 2061 | sparse12 + `r101_pl_pred` + encoder2048 |
| `r101_pl_encoder2048_ridge` | 2049 | `r101_pl_pred` + encoder2048 |

## Overall Results

Mean and standard deviation are across the 10 random sparse-support draws.

| model | features | train RMSE mean | train RMSE std | val RMSE mean | val RMSE std | val MAE mean |
|---|---:|---:|---:|---:|---:|---:|
| `sparse12_plus_r101_pl_ridge` | 13 | 5.197 | 0.129 | 5.885 | 0.253 | 4.540 |
| `sparse12_plus_r101_pl_encoder2048_ridge` | 2061 | 4.816 | 0.114 | 5.931 | 0.261 | 4.627 |
| `sparse12_plus_encoder2048_ridge` | 2060 | 5.125 | 0.094 | 6.310 | 0.288 | 4.919 |
| `sparse12_ridge` | 12 | 5.551 | 0.114 | 6.351 | 0.224 | 4.907 |
| `r101_pl_encoder2048_ridge` | 2049 | 6.439 | 0.268 | 7.141 | 0.372 | 5.657 |

## Per-Draw Validation RMSE

| support seed | sparse12 | sparse12 + r101_pl | sparse12 + encoder2048 | sparse12 + r101_pl + encoder2048 | r101_pl + encoder2048 |
|---:|---:|---:|---:|---:|---:|
| 1000 | 6.128 | 5.836 | 6.120 | 5.857 | 7.085 |
| 1001 | 6.762 | 6.134 | 6.720 | 6.161 | 7.125 |
| 1002 | 6.679 | 6.439 | 6.203 | 6.012 | 7.211 |
| 1003 | 6.527 | 6.002 | 6.405 | 6.065 | 7.587 |
| 1004 | 6.179 | 5.676 | 6.761 | 6.465 | 7.772 |
| 1005 | 6.186 | 5.896 | 5.787 | 5.541 | 6.573 |
| 1006 | 6.318 | 5.851 | 6.217 | 5.858 | 7.130 |
| 1007 | 6.214 | 5.755 | 6.139 | 5.676 | 6.741 |
| 1008 | 6.227 | 5.618 | 6.380 | 5.782 | 6.818 |
| 1009 | 6.291 | 5.642 | 6.364 | 5.891 | 7.364 |

## Interpretation

The cleanest ridge result is `sparse12_plus_r101_pl_ridge`: adding only the
decoded R101 prediction to the 12 sparse engineered features improves validation
RMSE from `6.351 ± 0.224` to `5.885 ± 0.253`. That is a real gain for a linear
head, and it says `r101_pl_pred` contains useful large-scale propagation signal
that is not fully captured by the sparse engineered features.

Adding the 2048 global pooled encoder features lowers train RMSE further
(`5.197 -> 4.816`) but does not improve validation (`5.885 -> 5.931`). This is
the clearest sign of extra fit capacity without robust transfer benefit under
random sparse supports. The encoder vector is not useless, but in this ridge
setup it mostly fits setup-level variation that does not generalize better than
the simpler `sparse12 + r101_pl_pred` head.

The added `sparse12 + encoder2048` check does not change that conclusion. It
gets `6.310 ± 0.288` validation RMSE, essentially tied with sparse12 alone
(`6.351 ± 0.224`) and much worse than sparse12 plus the decoded R101 prediction
(`5.885 ± 0.253`). So the encoder features by themselves are not the useful
R101 contribution in this ridge setup.

The R101-only-ish ridge set, `r101_pl_encoder2048_ridge`, is much worse
(`7.141 ± 0.372` validation RMSE). So the R101 features do not replace the real
sparse-support features. The best ridge behavior comes from using R101 as an
auxiliary propagation prior on top of real sparse interpolation/geometry
features.

## One-Feature R101 Calibration

I also fit ridge models using only `r101_pl_pred`, equivalent in raw space to:

`RSSI_hat = a * r101_pl_pred + b`

Using the same 10 random support draws and alpha sweep, the best alpha is
approximately `10`, though alpha barely changes the result from `0.01` to `10`.
The fitted raw-space calibration is:

`RSSI_hat = -0.323 * r101_pl_pred - 21.28`

with:

| feature set | best alpha | a mean | a std | b mean | b std | val RMSE mean | val RMSE std |
|---|---:|---:|---:|---:|---:|---:|---:|
| `r101_pl_pred` only | 10 | -0.323 | 0.011 | -21.28 | 0.86 | 7.167 | 0.271 |

This explains how `r101_pl_pred` behaves inside the multifeature ridge heads.
By itself, the decoded prediction needs a steep negative calibration slope but
still only reaches about `7.17 dB` RMSE. When sparse features are present, the
raw coefficient on `r101_pl_pred` shrinks to about `-0.10`, because the sparse
distance/support features already explain much of the same large-scale pathloss
trend.

## Conclusion

R101 does give an advantage in this ridge setting, but the advantage comes from
the decoded prediction feature, not from the 2048 encoder features. Adding
`r101_pl_pred` to the 12 sparse engineered features improves validation RMSE by
about `0.47 dB` at fixed `alpha=0.01` and about `0.44 dB` after alpha tuning,
with the improvement present across all 10 random support draws.

The 2048 global encoder features do not show a reliable advantage. Added without
`r101_pl_pred`, they barely change sparse-only validation performance
(`6.310 ± 0.288` vs `6.351 ± 0.224` at fixed alpha; `6.262 ± 0.280` vs
`6.238 ± 0.207` after tuning). Added on top of `sparse12 + r101_pl_pred`, they
lower train RMSE but are slightly worse on validation (`5.891 ± 0.259` vs
`5.799 ± 0.188` after tuning).

So the useful R101 contribution is a compact, decoded propagation prior that
helps the sparse real-data features. The pooled encoder vector mostly adds
capacity and does not provide dependable extra generalization under this
random-support benchmark.

## Alpha Tuning

I also swept ridge alpha separately for each feature set over:

`0.01, 0.1, 1, 10, 30, 100, 300, 1000, 3000, 10000`

This sweep used the same 10 random support draws and 16 AP-held-out folds. The
selected alpha is the alpha with the lowest mean validation RMSE across those
outer-CV runs; this is an alpha-sensitivity comparison, not nested inner-CV
model selection.

| model | features | best alpha | train RMSE mean | train RMSE std | val RMSE mean | val RMSE std | val MAE mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sparse12_plus_r101_pl_ridge` | 13 | 100 | 5.252 | 0.119 | 5.799 | 0.188 | 4.419 |
| `sparse12_plus_r101_pl_encoder2048_ridge` | 2061 | 100 | 4.872 | 0.110 | 5.891 | 0.259 | 4.529 |
| `sparse12_ridge` | 12 | 100 | 5.608 | 0.115 | 6.238 | 0.207 | 4.812 |
| `sparse12_plus_encoder2048_ridge` | 2060 | 100 | 5.196 | 0.100 | 6.262 | 0.280 | 4.847 |
| `r101_pl_encoder2048_ridge` | 2049 | 30 | 6.452 | 0.267 | 7.127 | 0.365 | 5.516 |

Tuning alpha improves the earlier fixed-`0.01` ridge results, especially by
regularizing the low-dimensional sparse heads more strongly. The ranking does
not materially change. The best model remains `sparse12 + r101_pl`, now at
`5.799 ± 0.188` RMSE. Adding encoder2048 to that model is slightly worse on
average (`+0.092 ± 0.304` dB paired across support draws), so there is still no
reliable validation advantage from the encoder vector.

## Artifacts

- `experiments/comparing_feature_sets_ridge.py`
- `experiments/tune_ridge_feature_set_alpha.py`
- `runs/comparing_feature_sets_ridge/comparing_feature_sets_ridge_overall_summary.csv`
- `runs/comparing_feature_sets_ridge/comparing_feature_sets_ridge_by_seed_summary.csv`
- `runs/comparing_feature_sets_ridge/comparing_feature_sets_ridge_fold_metrics.csv`
- `runs/comparing_feature_sets_ridge/comparing_feature_sets_ridge_predictions.csv`
- `runs/comparing_feature_sets_ridge/missing_sparse12_plus_encoder2048_overall_summary.csv`
- `runs/comparing_feature_sets_ridge/missing_sparse12_plus_encoder2048_by_seed_summary.csv`
- `runs/comparing_feature_sets_ridge/manifest.json`
- `runs/comparing_feature_sets_ridge/r101_global_feature_cache/`
- `runs/ridge_feature_set_alpha_tuning/ridge_feature_set_alpha_best_by_model.csv`
- `runs/ridge_feature_set_alpha_tuning/ridge_feature_set_alpha_overall_by_alpha.csv`
