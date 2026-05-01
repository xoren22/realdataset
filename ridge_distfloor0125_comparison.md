# R101 distance-floor ridge comparison

Generated from `experiments/comparing_feature_sets_ridge_distfloor0125.py`.

This reruns the 12 vs 13 ridge comparison after changing the R101 real-data
input preprocessing so the distance channel is floored at `0.125 m` before
log-normalization:

`distance_for_r101 = max(distance_m, 0.125)`.

The rerun uses the same support tables as the previous cached comparison for
seeds `1000..1009`, so the before/after comparison isolates the R101 input
preprocessing change. The plain `sparse12_ridge` model is unchanged because it
does not consume R101 predictions.

## Fixed alpha, alpha = 0.01

| model | old val RMSE | new val RMSE | delta |
|---|---:|---:|---:|
| `sparse12_ridge` | 6.350909 | 6.350909 | 0.000000 |
| `sparse12_plus_r101_pl_ridge` | 5.884951 | 5.877212 | -0.007739 |

The 13-feature gain over sparse12 changed from `0.465958 dB` to `0.473697 dB`.

## Tuned alpha

| model | old alpha | new alpha | old val RMSE | new val RMSE | delta |
|---|---:|---:|---:|---:|---:|
| `sparse12_ridge` | 100 | 100 | 6.237637 | 6.237637 | 0.000000 |
| `sparse12_plus_r101_pl_ridge` | 100 | 100 | 5.798903 | 5.793332 | -0.005571 |

The tuned 13-feature gain over sparse12 changed from `0.438733 dB` to
`0.444304 dB`.

## R101 prediction change

Across the ten cached support seeds, the new clipped-distance R101 prediction
feature differs from the previous cached feature by:

| statistic | value |
|---|---:|
| mean signed difference | +0.510874 dB |
| mean absolute difference | 0.569962 dB |
| RMSE difference | 0.893795 dB |
| mean max absolute difference per seed | 4.708511 dB |
| fraction changed by more than 0.1 dB | 0.738109 |

By distance bin, the fixed-alpha 13-feature ridge RMSE changed only slightly:

| AP distance bin | old RMSE | new RMSE | delta |
|---|---:|---:|---:|
| `<=0.5 m` | 6.548700 | 6.540769 | -0.007931 |
| `0.5-1 m` | 6.630039 | 6.622830 | -0.007209 |
| `1-2 m` | 5.373284 | 5.373317 | +0.000033 |
| `2-4 m` | 4.646265 | 4.640155 | -0.006111 |
| `4-8 m` | 4.463358 | 4.455847 | -0.007511 |
| `>8 m` | 7.959466 | 7.948104 | -0.011363 |

## Conclusion

The preprocessing change does affect the raw R101 prediction feature, but it
does not materially change the ridge feature-set result on this benchmark. The
best model is still `sparse12 + r101_pl_pred`, with essentially the same alpha
and about a `0.44 dB` validation RMSE improvement over sparse12 alone after
alpha tuning.

Outputs:

- `runs/comparing_feature_sets_ridge_distfloor0125/r101_distfloor0125_feature_cache/`
- `runs/comparing_feature_sets_ridge_distfloor0125/ridge_distfloor0125_fixed_alpha_overall_summary.csv`
- `runs/comparing_feature_sets_ridge_distfloor0125/ridge_distfloor0125_alpha_best_by_model.csv`
- `runs/comparing_feature_sets_ridge_distfloor0125/ridge_distfloor0125_alpha_overall_by_alpha.csv`
