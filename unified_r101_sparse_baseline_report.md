# Unified RSSI Baseline and r101 Transfer Report

Generated from the experiments in this conversation. All reported RMSE values
are in dBm and are computed as:

`RMSE = sqrt(mean((prediction - target)^2))`

## Evaluation Design

The dataset contains 1,027 observed RSSI measurements. The upstream H5 grid has
1,060 nonzero labels because 33 target values are filled/imputed by the toolbox;
those imputed H5 targets were excluded from all train/validation scoring.

All final experiments use leave-one-AP-location-out cross validation with 16
outer folds. This split is important because some AP locations appear in more
than one setup; grouping by AP point prevents the same AP location from leaking
between train and validation. For sampling-assisted experiments, 5 observed RSSI
points per setup are revealed as sparse context and are not scored as query
points. The remaining 927 non-support query points are scored.

![Observed RSSI target distribution](figs/baseline_target_distribution.png)

![Observed RSSI vs distance](figs/baseline_rssi_vs_distance.png)

## Baselines Without Sparse Context

These first baselines use only ordinary engineered geometry/room features:
`ap_x_m, ap_y_m, mx_m, my_m, log_distance_m, same_room, room_meas, room_ap`.
They do not get the 5 sparse measurements from the target setup.

| model | pooled RMSE | MAE | mean fold RMSE | fold RMSE std | corr(y, yhat) |
|---|---:|---:|---:|---:|---:|
| rf | 5.093 | 3.908 | 4.950 | 1.429 | 0.925 |
| knn | 5.293 | 4.093 | 5.078 | 1.646 | 0.921 |
| hgb | 5.673 | 4.358 | 5.513 | 1.830 | 0.907 |
| ridge | 6.166 | 4.834 | 6.146 | 1.469 | 0.887 |
| logdistance | 8.213 | 6.533 | 8.070 | 1.281 | 0.788 |
| mean | 13.442 | 10.726 | 13.109 | 3.434 | -0.352 |

![No-sparse baseline RMSE](figs/baseline_cv_rmse.png)

![No-sparse best prediction scatter](figs/baseline_pred_vs_true.png)

**Conclusion.** Without sparse target-setup measurements, the constrained random
forest is the strongest ordinary baseline at 5.09 dBm. This is a useful lower
bar for judging whether the 5 sparse samples and r101 features are actually
adding signal.

## r101 Input Construction

The MLSP r101 checkpoint expects 11 input channels:

`reflectance, transmittance, distance, antenna_gain, freq_sin_1, freq_cos_1, freq_sin_2, freq_cos_2, mask, floor_plan, sparse`

For this real dataset we can construct only part of that input directly:

| channel | construction in this dataset |
|---|---|
| reflectance/transmittance | zero-filled by default; random wall imputation was tested separately and disabled |
| distance | Euclidean distance from AP to every image pixel, normalized using MLSP stats |
| antenna_gain | unavailable, zero-filled |
| frequency Fourier channels | generated from setup WiFi channel/frequency |
| mask | 1 on unknown pixels, 0 at the 5 sparse support pixels |
| floor_plan | inferred wall-only mask from the known grid layout |
| sparse | 5 RSSI support points converted to pathloss proxy and resized nearest-neighbor to 256x256 |

The floor-plan channel was corrected to be wall-only: 1 means wall and 0 means
free/non-wall. The earlier occupancy-style floor plan was wrong because it made
whole rooms 1. We used the 1px wall variant from then on. A 2px wall variant was
briefly tested and was slightly worse for the r101 ridge feature head.

![Wall floor-plan channel](figs/sampling_floor_plan_channel_1px.png)

Sparse RSSI support values are converted to a pathloss proxy with:

`PL_proxy = C_fold - RSSI`

where:

`C_fold = 110.146 + mean(training support RSSI)`

This is a fold-safe normalization alignment, not a physical EIRP estimate. The
constant is recomputed inside each AP-held-out fold using only training AP
support points. The held-out AP fold is not used to estimate `C_fold`.

![Sparse channel before and after r101 featurization](figs/sparse_channel_before_after_setup1_1px.png)

![Selected r101 input channels](figs/sampling_r101_channels_setup1_1px.png)

Important caveat: all r101 feature-extraction experiments below still include
the 5 sparse support pixels in the r101 sparse channel. When a model is called
"r101-only" below, that means the downstream tabular model only sees r101
outputs; it does not mean r101 was run without sparse measurements.

## Sampling-Assisted Results

These experiments use the 5 sparse support measurements per setup. Query
measurements are excluded from the sparse input and are the only scored points.

| model | feature set | pooled RMSE | MAE | mean fold RMSE | fold RMSE std | corr(y, yhat) |
|---|---|---:|---:|---:|---:|---:|
| r101_feature_ridge | sparse engineered features + `r101_pl_pred` + 2048 encoder features | 4.593 | 3.545 | 4.498 | 0.574 | 0.937 |
| sparse_rf | sparse engineered features only | 4.662 | 3.614 | 4.608 | 0.942 | 0.936 |
| r101_pl_rf | sparse engineered features + `r101_pl_pred` | 4.668 | 3.595 | 4.625 | 0.774 | 0.935 |
| r101_top10_rf | sparse engineered features + `r101_pl_pred` + top 10 encoder features | 4.829 | 3.706 | 4.802 | 0.873 | 0.931 |
| sparse_ridge | sparse engineered features only | 4.891 | 3.813 | 4.786 | 0.837 | 0.928 |
| r101_only_ridge_a1 | `r101_pl_pred` + 2048 encoder features only | 6.332 | 5.154 | 6.327 | 0.959 | 0.877 |
| r101_only_ridge_a0_1 | same, alpha 0.1 | 6.332 | 5.155 | 6.327 | 0.961 | 0.877 |
| r101_only_ridge_a0_01 | same, alpha 0.01 | 6.332 | 5.155 | 6.327 | 0.961 | 0.877 |
| r101_only_ridge_a10 | same, alpha 10 | 6.334 | 5.153 | 6.330 | 0.944 | 0.876 |
| r101_only_ridge_a100 | same, alpha 100 | 6.489 | 5.207 | 6.485 | 0.851 | 0.874 |
| r101_pl_train_cal | direct r101 decoded pathloss prediction, linearly calibrated on outer-train | 6.947 | 5.719 | 6.844 | 0.859 | 0.849 |
| sparse_idw | inverse-distance interpolation from 5 sparse RSSI samples | 7.671 | 5.777 | 7.463 | 1.529 | 0.815 |
| r101_pl_support_cal | per-setup calibration using only 5 sparse points | 15.525 | 12.615 | 15.049 | 3.235 | 0.871 |

![Sampling-assisted RMSE](figs/sampling_assisted_rmse.png)

![Best sampling-assisted prediction scatter](figs/sampling_assisted_best_scatter.png)

![Direct r101 calibrated prediction scatter](figs/r101_raw_calibrated_y_true_vs_y_hat_1px.png)

**Conclusion.** The best model is `r101_feature_ridge` at 4.59 dBm. It improves
over the no-sparse RF baseline by about 0.50 dBm and over sparse RF by about
0.07 dBm. The improvement over sparse RF is small but consistent with the
correlation and feature-importance analysis: r101 adds a smooth correction, but
much of its signal overlaps with distance and sparse-support interpolation.

### One-feature r101 decoded-map calibration

A later random-support check fit ridge models using only the decoded
`r101_pl_pred` scalar:

`RSSI_hat = a * r101_pl_pred + b`

Across 10 random support draws and the same AP-held-out folds, the best alpha
was about `10`, though alphas from `0.01` to `10` were almost identical. The
average raw-space fit was:

`RSSI_hat = -0.323 * r101_pl_pred - 21.28`

with validation RMSE `7.167 ± 0.271` dBm over random support draws. This is
worse than the fixed-support `r101_pl_train_cal` row above because the newer
number averages over random sparse support choices and uses only one scalar
feature. It is still useful diagnostically: when sparse features are absent,
the model leans heavily on `r101_pl_pred`; when sparse features are present, the
raw coefficient on `r101_pl_pred` shrinks to about `-0.10` because sparse
distance/support features explain much of the same large-scale pathloss trend.

### Paired significance of adding `r101_pl_pred`

A later feature-set comparison repeated the sparse random-support protocol over
10 independent support draws and compared the 12 non-constant sparse engineered
features against those same 12 features plus `r101_pl_pred`. The comparison is
paired by support draw, so it directly measures whether the 13th feature gives
a consistent gain rather than just comparing overlapping error bars.

| model family | comparison | mean RMSE gain | std of paired gain | paired t-test p-value |
|---|---|---:|---:|---:|
| tuned SVM-style linear regressor | `sparse12` -> `sparse12 + r101_pl_pred` | 0.506 | 0.152 | 2.3e-6 |
| tuned ridge | `sparse12` -> `sparse12 + r101_pl_pred` | 0.439 | 0.119 | 9.7e-7 |

For the tuned SVM-style comparison, all 10 support draws improved when adding
`r101_pl_pred`. This makes the decoded r101 prediction's contribution
statistically clear under random sparse sampling, even though the absolute gain
is modest at roughly half a dB.

## Ridge Alpha Choice

We initially swept ridge alphas with AP-grouped inner CV. That showed small
alphas were best for `r101_feature_ridge`; large alphas sharply underfit. The
main script now fixes `alpha = 0.01` and does not search alphas during the main
experiment.

Earlier alpha grid:

`0.01, 0.1, 1, 10, 30, 100, 300, 1000, 3000, 10000`

![Archived r101 ridge alpha sweep](figs/alpha_tuning_r101_feature_ridge.png)

![Archived sparse ridge alpha sweep](figs/alpha_tuning_sparse_ridge.png)

**Conclusion.** The original hand-set `alpha=100` was too strong for the 2049+
dimensional r101 ridge head. With fixed `alpha=0.01`, `r101_feature_ridge`
improved from about 4.74 dBm to 4.59 dBm.

## Correlation and Residual Analysis

Raw correlation with the target is high for several models:

| model | corr(y, yhat) | RMSE | bias |
|---|---:|---:|---:|
| r101_feature_ridge | 0.937 | 4.593 | 0.051 |
| sparse_rf | 0.936 | 4.662 | -0.172 |
| r101_pl_rf | 0.935 | 4.668 | -0.124 |
| r101_top10_rf | 0.931 | 4.829 | -0.288 |
| sparse_ridge | 0.928 | 4.891 | -0.166 |
| r101_only_ridge_a1 | 0.877 | 6.332 | -0.248 |
| r101_pl_train_cal | 0.849 | 6.947 | 0.029 |
| sparse_idw | 0.815 | 7.671 | 0.857 |

Residual correlations explain why r101 does not dominate sparse RF despite
being clearly useful. After sparse RF, the remaining residual has almost no
linear correlation with the r101 predictions:

| residual | candidate predictor | corr(candidate, residual) |
|---|---|---:|
| y - sparse_rf | r101_pl_train_cal | -0.045 |
| y - sparse_rf | r101_feature_ridge | -0.015 |
| y - sparse_rf | r101_only_ridge_a1 | 0.006 |
| y - sparse_rf | r101_pl_rf | -0.069 |
| y - sparse_rf | r101_top10_rf | -0.067 |

**Conclusion.** r101 is not weak: direct calibrated r101 has 0.849 correlation
with the target, and r101-only ridge reaches 0.877. The issue is conditional
information. Sparse support features and geometry already explain much of the
same large-scale variance, so r101 only has a small residual contribution after
those features are present.

## Feature Importance

### sparse_rf

The constrained sparse RF relies mostly on distance and local sparse support.

| feature | mean importance | std |
|---|---:|---:|
| log_distance_m | 0.343 | 0.018 |
| distance_m | 0.339 | 0.019 |
| nearest_support_rssi | 0.089 | 0.022 |
| same_room | 0.086 | 0.006 |
| idw_support_rssi | 0.058 | 0.036 |
| my_m | 0.029 | 0.002 |
| nearest_support_dist_m | 0.016 | 0.002 |
| mx_m | 0.014 | 0.003 |

![sparse_rf feature importance](figs/feature_importance_sparse_rf.png)

### sparse_ridge

For sparse ridge, the strongest standardized coefficient is the IDW sparse RSSI
estimate, followed by distance, room, and support statistics.

| feature | mean abs coef | std |
|---|---:|---:|
| idw_support_rssi | 8.203 | 0.308 |
| log_distance_m | 3.036 | 0.211 |
| same_room | 2.775 | 0.120 |
| support_mean_rssi | 2.733 | 0.255 |
| support_std_rssi | 2.169 | 0.176 |
| distance_m | 1.883 | 0.307 |
| my_m | 0.885 | 0.108 |
| mx_m | 0.691 | 0.108 |

![sparse_ridge feature importance](figs/feature_importance_sparse_ridge.png)

### r101_feature_ridge

The r101 ridge head uses both engineered sparse features and r101 outputs. The
largest non-encoder coefficients are:

| group | feature | mean abs coef | std |
|---|---|---:|---:|
| sparse | idw_support_rssi | 6.060 | 0.220 |
| r101_pl_pred | r101_pl_pred | 4.154 | 0.262 |
| sparse | same_room | 2.537 | 0.124 |
| sparse | log_distance_m | 2.261 | 0.218 |
| sparse | nearest_support_dist_m | 0.911 | 0.083 |
| sparse | distance_m | 0.857 | 0.256 |
| sparse | my_m | 0.399 | 0.118 |
| sparse | mx_m | 0.310 | 0.110 |

Coefficient mass by group:

| group | total abs coef mean | std |
|---|---:|---:|
| sparse | 13.659 | 0.385 |
| r101_encoder | 10.237 | 0.301 |
| r101_pl_pred | 4.154 | 0.262 |

Top individual encoder coefficients are tiny, e.g. `r101_f1336` has mean abs
coefficient 0.028. The encoder contribution is spread over many weak
directions, not a few dominant coordinates.

![r101_feature_ridge non-encoder importance](figs/feature_importance_r101_ridge.png)

**Conclusion.** r101 features are definitely used, especially `r101_pl_pred`,
but the sparse/interpolation block remains the largest source of signal. The
encoder helps ridge as a distributed linear correction; it is not naturally
suited to small-sample tree splits on individual encoder coordinates.

## Tree-Based r101 Heads

We tried two RF heads using r101-derived features:

| model | features | RMSE |
|---|---|---:|
| r101_pl_rf | sparse engineered features + `r101_pl_pred` | 4.668 |
| r101_top10_rf | sparse engineered features + `r101_pl_pred` + top 10 encoder features selected by fixed-ridge weights | 4.829 |

The raw 2048-feature tree idea was also tried earlier and was worse:
`r101_feature_rf` was about 5.88 dBm and `r101_feature_extra_trees` was about
6.98 dBm. Reducing to the top 10 ridge-weighted encoder coordinates improved
the RF substantially, but adding those encoder coordinates still hurt compared
with using `r101_pl_pred` alone.

**Conclusion.** The tree-based head does not find useful split structure in the
encoder coordinates at this sample size. The r101 signal is better consumed by
ridge than by RF.

## Reflectance/Transmittance Imputation

Because the real dataset lacks RF reflectance/transmittance channels, we tested
a random wall-value imputation idea. For each draw, one positive reflectance and
one positive transmittance scalar were sampled from the MLSP normalization
distributions, written only on wall pixels, normalized, and passed through r101.
Predictions/features were averaged over 8 draws. The sampler explicitly
rejected non-positive raw r/t values.

Archived result from that run:

| model | zero-filled r/t RMSE | random r/t imputation RMSE |
|---|---:|---:|
| r101_feature_ridge | 4.741 | 4.965 |
| r101_pl_train_cal | 6.947 | 7.542 |
| r101_pl_support_cal | 15.525 | 16.321 |

Sparse-only models were unchanged by construction. Imputation was slower and
worse, so it remains implemented but disabled by default.

**Conclusion.** Knowing wall locations is useful, but randomly guessing material
parameters from broad synthetic statistics did not transfer. Zero-filled r/t is
the better default for this real dataset.

## What Sparse Measurements Changed

The 5 sparse target-setup measurements are the most important experimental
shift:

| comparison | RMSE |
|---|---:|
| no-sparse RF | 5.093 |
| sparse_rf with 5 support points | 4.662 |
| sparse_ridge with 5 support points | 4.891 |
| r101_pl_train_cal with sparse channel in r101 input | 6.947 |
| r101_only_ridge with sparse channel in r101 input | 6.332 |
| r101_feature_ridge with sparse support + geometry + r101 | 4.593 |

The sparse channel helps in two ways:

1. Directly, through engineered support features such as `idw_support_rssi`,
   `nearest_support_rssi`, and `support_mean_rssi`.
2. Indirectly, through the r101 sparse measurement channel, which lets the
   frozen network condition its decoded map and encoder representation on real
   measurements from the target setup.

**Conclusion.** Sparse measurements are not a minor detail. They turn the task
from AP-location generalization into sampling-assisted map reconstruction, and
they explain most of the improvement over the original no-sparse baselines.

## Overall Assessment

The strongest final model is `r101_feature_ridge`:

`13 sparse/geometry features + r101_pl_pred + 2048 r101 encoder features`

with fixed `alpha=0.01`, evaluated by AP-held-out outer CV. It reaches 4.59 dBm
RMSE. This is the best result we obtained, but the margin over sparse RF is
small: 4.59 versus 4.66 dBm.

The experiments support these conclusions:

1. The sparse support points are highly valuable. They improve RF from 5.09 dBm
   without sparse context to 4.66 dBm with 5 support points.
2. The r101 checkpoint is useful. Direct decoded-map calibration has 0.849
   target correlation, and network-only ridge reaches 6.33 dBm.
3. r101 is most useful as a linear correction combined with sparse/geometry
   features. That combination reaches 4.59 dBm.
4. r101 does not give much residual information after sparse RF; residual
   correlations are near zero.
5. Tree heads are not the right consumer for the encoder features at this data
   size. The encoder signal is distributed across many small linear directions.
6. Wall-only 1px floor-plan construction is the correct floor channel; random
   reflectance/transmittance imputation did not help.

Final recommendation: use `r101_feature_ridge` as the best baseline, report
`sparse_rf` as the strongest simple sparse-aware baseline, and describe r101 as
a modest but real transfer signal whose main value appears when combined with
the sparse support/geometry features through a strongly regularized linear head.

## Artifacts

- `experiments/baselines.py`
- `experiments/sampling_assisted_r101.py`
- `runs/baselines_summary.csv`
- `runs/baselines_oof_predictions.csv`
- `runs/sampling_assisted_r101_summary.csv`
- `runs/sampling_assisted_r101_predictions.csv`
- `runs/sampling_assisted_r101_correlations.csv`
- `runs/sampling_assisted_r101_residual_correlations.csv`
- `runs/feature_importance_*.csv`
- `figs/baseline_*.png`
- `figs/sampling_*.png`
- `figs/feature_importance_*.png`
- `figs/alpha_tuning_*.png`
