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


# Sampling-assisted r101 transfer on RSSI data

Auto-generated by `experiments/sampling_assisted_r101.py`.

- **task**: sampling-assisted RSSI map reconstruction, not zero-shot AP generalization
- **sparse context**: 5 observed RSSI points per setup are revealed as input
- **floor-plan channel**: wall-only mask, 1px thick
- **active r/t channel variant**: `zero_rt`
- **evaluation**: only non-support query points are scored
- **outer split**: leave-one-AP-location-out, k = 16
- **query predictions per model**: 927
- **checkpoint**: `/mnt/weka/kpetrosyan/synthetic_pretraining_checkpoints/r101_bs181_lr3e-03_wr0.1_dr0.1_wd1e-03/2026-04-14_09-58-12.473943/every/step_00350000_every.ckpt`

## Input construction

The r101 model expects 11 MLSP channels. This dataset provides distance,
frequency, mask, a layout-derived wall floor plan, and sparse measurements. It
lacks measured RF reflectance, RF transmittance, and antenna pattern. The active
`zero_rt` variant zero-fills reflectance/transmittance. The script keeps an
`impute_rt` implementation for later experiments, but it is disabled by default
because it did not improve RMSE and made runs much slower.

Sparse RSSI support values are converted to a pathloss proxy with:

`PL_proxy = C_fold - RSSI`, where
`C_fold = 110.146 + mean(training support RSSI)`.

`C_fold` is recomputed inside each AP-held-out fold using only support points
from training AP locations, then applied to both train and held-out sparse
channels in that fold. This is not a physical EIRP estimate; it is a fold-safe
scale alignment for the pretrained sparse-channel normalization.

![Wall floor-plan channel](figs/sampling_floor_plan_channel_1px.png)

![Sparse pathloss channel](figs/sampling_sparse_channel_setup1.png)

![Selected r101 input channels](figs/sampling_r101_channels_setup1_1px.png)

## Results

| wall_thickness_px | rt_mode | rt_impute_n | model | pooled_rmse | pooled_mae | mean_fold_rmse | std_fold_rmse | mean_fold_r2 | n_predictions |
|---|---|---|---|---|---|---|---|---|---|
| 1 | zero_rt | 1 | r101_feature_ridge | 4.593 | 3.545 | 4.498 | 0.574 | 0.836 | 927 |
| 1 | zero_rt | 1 | sparse_rf | 4.662 | 3.614 | 4.608 | 0.942 | 0.811 | 927 |
| 1 | zero_rt | 1 | r101_pl_rf | 4.668 | 3.595 | 4.625 | 0.774 | 0.815 | 927 |
| 1 | zero_rt | 1 | r101_top10_rf | 4.829 | 3.706 | 4.802 | 0.873 | 0.799 | 927 |
| 1 | zero_rt | 1 | sparse_ridge | 4.891 | 3.813 | 4.786 | 0.837 | 0.822 | 927 |
| 1 | zero_rt | 1 | r101_only_ridge_a1 | 6.332 | 5.154 | 6.327 | 0.959 | 0.643 | 927 |
| 1 | zero_rt | 1 | r101_only_ridge_a0_1 | 6.332 | 5.155 | 6.327 | 0.961 | 0.642 | 927 |
| 1 | zero_rt | 1 | r101_only_ridge_a0_01 | 6.332 | 5.155 | 6.327 | 0.961 | 0.642 | 927 |
| 1 | zero_rt | 1 | r101_only_ridge_a10 | 6.334 | 5.153 | 6.330 | 0.944 | 0.645 | 927 |
| 1 | zero_rt | 1 | r101_only_ridge_a100 | 6.489 | 5.207 | 6.485 | 0.851 | 0.649 | 927 |
| 1 | zero_rt | 1 | r101_pl_train_cal | 6.947 | 5.719 | 6.844 | 0.859 | 0.601 | 927 |
| 1 | zero_rt | 1 | sparse_idw | 7.671 | 5.777 | 7.463 | 1.529 | 0.578 | 927 |
| 1 | zero_rt | 1 | r101_pl_support_cal | 15.525 | 12.615 | 15.049 | 3.235 | -0.949 | 927 |

Best model: **r101_feature_ridge** with pooled query RMSE **4.59 dBm**.

![Sampling-assisted RMSE](figs/sampling_assisted_rmse.png)

![Best model query predictions](figs/sampling_assisted_best_scatter.png)

## Ridge alpha

`sparse_ridge` and `r101_feature_ridge` use fixed `alpha = 0.01`. This
was chosen from the earlier AP-grouped inner-CV sweep; the script no longer
searches alpha during the main experiment. The earlier candidate grid was:

`0.01, 0.1, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0`

Active alpha records:

| model | alpha | selected_folds |
|---|---|---|
| r101_feature_ridge | 0.010 | 16 |
| r101_only_ridge_a0_01 | 0.010 | 16 |
| r101_only_ridge_a0_1 | 0.100 | 16 |
| r101_only_ridge_a1 | 1.000 | 16 |
| r101_only_ridge_a10 | 10.000 | 16 |
| r101_only_ridge_a100 | 100.000 | 16 |
| sparse_ridge | 0.010 | 16 |

## Feature importance

These importances are averaged across the same AP-held-out folds. Ridge
coefficients are absolute coefficients after feature standardization, so they
are comparable within a fitted ridge head. The r101 encoder group is reported
both as total coefficient mass and as individual top encoder dimensions because
its signal is spread across 2,048 pooled features.

### sparse_rf

| feature | mean | std |
|---|---|---|
| log_distance_m | 0.343 | 0.018 |
| distance_m | 0.339 | 0.019 |
| nearest_support_rssi | 0.089 | 0.022 |
| same_room | 0.086 | 0.006 |
| idw_support_rssi | 0.058 | 0.036 |
| my_m | 0.029 | 0.002 |
| nearest_support_dist_m | 0.016 | 0.001 |
| mx_m | 0.014 | 0.003 |

![sparse_rf feature importance](figs/feature_importance_sparse_rf.png)

### sparse_ridge

| feature | mean | std |
|---|---|---|
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

Non-encoder coefficients:

| group | feature | mean | std |
|---|---|---|---|
| sparse | idw_support_rssi | 6.060 | 0.220 |
| r101_pl_pred | r101_pl_pred | 4.154 | 0.262 |
| sparse | same_room | 2.537 | 0.124 |
| sparse | log_distance_m | 2.261 | 0.218 |
| sparse | nearest_support_dist_m | 0.911 | 0.082 |
| sparse | distance_m | 0.857 | 0.256 |
| sparse | my_m | 0.399 | 0.118 |
| sparse | mx_m | 0.310 | 0.110 |
| sparse | nearest_support_rssi | 0.299 | 0.178 |
| sparse | support_std_rssi | 0.016 | 0.002 |

Group coefficient mass:

| group | mean | std |
|---|---|---|
| sparse | 13.659 | 0.385 |
| r101_encoder | 10.237 | 0.301 |
| r101_pl_pred | 4.154 | 0.262 |

Top encoder coefficients:

| feature | mean | std |
|---|---|---|
| r101_f1336 | 0.028 | 0.003 |
| r101_f724 | 0.020 | 0.002 |
| r101_f214 | 0.020 | 0.003 |
| r101_f906 | 0.019 | 0.002 |
| r101_f135 | 0.019 | 0.002 |
| r101_f1825 | 0.018 | 0.002 |
| r101_f1206 | 0.018 | 0.003 |
| r101_f1587 | 0.018 | 0.002 |

![r101_feature_ridge feature importance](figs/feature_importance_r101_ridge.png)

## Conclusion

The r101 checkpoint is useful as a frozen feature extractor once the ridge
regularization is tuned inside the outer folds: `r101_feature_ridge` gets
4.59 dBm RMSE versus
4.66 dBm for `sparse_rf` and
4.89 dBm for `sparse_ridge`. Directly calibrating
the r101 pathloss map is much worse (`r101_pl_train_cal` =
6.95 dBm), which suggests that the synthetic pathloss
decoder does not transfer cleanly to this RSSI dataset when reflectance and
transmittance are unavailable.

The tree ablations did not help: `r101_pl_rf` gets 4.67
dBm using sparse features plus `r101_pl_pred`, and `r101_top10_rf` gets
4.83 dBm after adding the 10 encoder coordinates with
the largest absolute fixed-ridge weights per outer fold. This suggests the
useful r101 signal transfers as a smooth, mostly additive correction that ridge
can exploit, while the decoded map and selected individual encoder coordinates
are not strong RF split variables.

The pure-network ridge ablation, using only `r101_pl_pred` plus the 2,048
encoder features and no engineered sparse/geometry features, reaches
6.33 dBm with `r101_only_ridge_a1`. This confirms
that r101 alone carries substantial signal, but the best performance still
comes from combining r101 with the sparse support and geometry features.

The useful signal in this dataset is mostly captured by sparse RSSI support
statistics plus geometry. The feature-importance tables support that diagnosis:
`r101_feature_ridge` uses `r101_pl_pred`, but its largest non-encoder weights
are still the sparse/interpolation and geometry features. The pooled encoder
features carry signal as many small coefficients; that is enough to improve a
tuned linear head, but not enough for the raw-feature tree ensembles to beat the
sparse random forest.

## Artifacts

- `runs/sampling_assisted_r101_samples.csv`
- `runs/sampling_assisted_r101_folds.csv`
- `runs/sampling_assisted_r101_predictions.csv`
- `runs/sampling_assisted_r101_summary.csv`
- `runs/sampling_assisted_r101_config.json`
- `figs/sampling_*.png`


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


# R101 Concrete Follow-up Checks

This report follows the five prioritized checks from `r101_error_analyzes.md`. The comparisons are intentionally limited to `sparse_rf` and the best existing R101-based model, `r101_feature_ridge`, except check 5 where the R101 ridge head is rerun with point-local features in place of global pooled features.

## 1. Remove Points Closest To The AP

**Goal.** Test whether AP-adjacent labels are a special error source and whether removing them changes the `sparse_rf` vs `r101_feature_ridge` comparison.

**How.** I reused the existing out-of-fold query predictions, removed query points by true AP distance threshold, and recomputed pooled RMSE for only `sparse_rf` and `r101_feature_ridge`. Support points were already excluded by the original prediction table.

**Results.**

| exclude_distance | kept_n | removed_n | sparse_rf_rmse | sparse_rf_delta | r101_feature_ridge_rmse | r101_feature_ridge_delta | r101_minus_sparse |
|---|---|---|---|---|---|---|---|
| none | 927 | 0 | 4.662 | 0.000 | 4.593 | 0.000 | -0.068 |
| >0m | 910 | 17 | 4.633 | -0.028 | 4.557 | -0.036 | -0.077 |
| >0.5m | 909 | 18 | 4.634 | -0.028 | 4.559 | -0.034 | -0.074 |
| >1m | 857 | 70 | 4.498 | -0.164 | 4.387 | -0.206 | -0.111 |
| >1.5m | 816 | 111 | 4.440 | -0.221 | 4.326 | -0.267 | -0.114 |
| >2m | 777 | 150 | 4.465 | -0.196 | 4.303 | -0.291 | -0.162 |
| >3m | 655 | 272 | 4.446 | -0.216 | 4.301 | -0.293 | -0.145 |

Worst query errors among points within 1 m of the H5 AP label:

| setup | point | holdout_ap_point | distance_m | y_true | sparse_rf | r101_feature_ridge | sparse_abs | r101_abs |
|---|---|---|---|---|---|---|---|---|
| 6 | 35 | 26 | 1.00 | -53.00 | -40.80 | -33.80 | 12.20 | 19.20 |
| 2 | 2 | 3 | 1.00 | -19.00 | -35.72 | -34.45 | 16.72 | 15.45 |
| 6 | 25 | 26 | 1.00 | -46.00 | -36.47 | -32.70 | 9.53 | 13.30 |
| 2 | 4 | 3 | 1.00 | -48.00 | -35.31 | -35.51 | 12.69 | 12.49 |
| 16 | 30 | 30 | 0.00 | -44.00 | -25.16 | -31.73 | 18.84 | 12.27 |
| 16 | 29 | 30 | 1.00 | -26.00 | -37.86 | -37.94 | 11.86 | 11.94 |
| 1 | 2 | 1 | 1.00 | -40.00 | -30.86 | -28.31 | 9.14 | 11.69 |
| 3 | 5 | 5 | 0.00 | -22.00 | -29.13 | -33.63 | 7.13 | 11.63 |
| 20 | 47 | 47 | 0.00 | -24.00 | -32.63 | -33.42 | 8.63 | 9.42 |
| 6 | 23 | 26 | 1.00 | -41.00 | -36.13 | -32.19 | 4.87 | 8.81 |

The near-AP points are disproportionately bad, especially for R101. Removing only the exact AP point leaves the ranking unchanged, but removing points within 1 m cuts R101 RMSE much more than sparse RF and increases the R101 edge from about 0.07 dB to about 0.24 dB. This supports the suspicion that AP-adjacent label/pathology effects are hiding some of the useful R101 signal.

## 2. Audit Setup 16 And Setup 2 Mapping

**Goal.** Audit setup 16 and setup 2 point mapping/AP labels, focusing on the suspicious AP-adjacent values.

**How.** I compared `RSSI_raw_data.csv`, the H5 `ap_locations`, the H5 `indices` grid, and the repo's sequential raw-row-to-point mapping. I also parsed the raw `Distance` column, which appears to be the phone's reported WiFi distance estimate, and compared it to geometric distance from the H5 AP point.

**Results.**

Summary:

| setup | h5_ap_point | zero_geom_point | ap_rssi | ap_raw_distance | strongest_point | strongest_rssi | strongest_geom_distance | raw_nearest_point | raw_nearest_distance | raw_nearest_rssi | csv_matches_h5_grid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 3 | 3 | -33.000 | 0.400 | 2 | -19.000 | 1.000 | 2 | 0.100 | -19.000 | True |
| 16 | 30 | 30 | -44.000 | 1.500 | 29 | -26.000 | 1.000 | 29 | 0.200 | -26.000 | True |

AP-neighborhood rows:

| setup | point | h5_ap_point | geom_distance_m | raw_distance_m | rssi |
|---|---|---|---|---|---|
| 2 | 1 | 3 | 2.000 | 0.800 | -38.000 |
| 2 | 2 | 3 | 1.000 | 0.100 | -19.000 |
| 2 | 3 | 3 | 0.000 | 0.400 | -33.000 |
| 2 | 4 | 3 | 1.000 | 2.500 | -48.000 |
| 2 | 5 | 3 | 2.000 | 2.000 | -46.000 |
| 2 | 6 | 3 | 2.500 | 2.800 | -49.000 |
| 2 | 7 | 3 | 3.162 | 2.000 | -46.000 |
| 2 | 8 | 3 | 2.236 | 1.200 | -42.000 |
| 16 | 26 | 30 | 4.000 | 2.200 | -47.000 |
| 16 | 27 | 30 | 3.000 | 1.400 | -43.000 |
| 16 | 28 | 30 | 2.000 | 0.700 | -37.000 |
| 16 | 29 | 30 | 1.000 | 0.200 | -26.000 |
| 16 | 30 | 30 | 0.000 | 1.500 | -44.000 |
| 16 | 31 | 30 | 1.000 | 1.500 | -44.000 |
| 16 | 32 | 30 | 1.414 | 1.900 | -46.000 |
| 16 | 33 | 30 | 2.236 | 1.400 | -43.000 |
| 16 | 34 | 30 | 3.162 | 2.700 | -49.000 |
| 16 | 35 | 30 | 4.123 | 2.700 | -49.000 |

The raw CSV values and H5 grid values agree, so this is not a CSV-to-H5 transcription error. The suspicious part is the AP label itself. In setup 16, H5 says AP point 30, but point 29 is both the strongest label (-26 dBm) and the raw-nearest point (~0.2 m); point 30 is -44 dBm and raw distance ~1.5 m. Setup 2 has the same pattern: H5 says AP point 3, but point 2 is strongest (-19 dBm) and raw-nearest (~0.1 m), while point 3 is -33 dBm and raw distance ~0.4 m. This makes setup 16 and setup 2 look like one-grid-point AP label errors or at least AP placement uncertainty at exactly the scale that hurts deterministic pathloss models.

## 3. Random Support Draws Instead Of Fixed Linspace

**Goal.** Replace fixed linspace support points with random support draws to see whether R101 only helps for particular support geometries.

**How.** I drew 10 random five-point support sets per setup (`seeds 1000..1009`). For each draw I rebuilt sparse features, rebuilt the R101 sparse input channel, reran frozen R101 inference, and evaluated only `sparse_rf` and `r101_feature_ridge` with the same leave-one-AP-location-out split.

**Results.**

Per random draw:

| scheme | r101_feature_ridge | sparse_rf | r101_minus_sparse |
|---|---|---|---|
| random_1000 | 5.857 | 4.938 | 0.920 |
| random_1001 | 6.161 | 5.012 | 1.149 |
| random_1002 | 6.012 | 5.044 | 0.968 |
| random_1003 | 6.065 | 5.292 | 0.774 |
| random_1004 | 6.465 | 4.845 | 1.620 |
| random_1005 | 5.541 | 6.153 | -0.612 |
| random_1006 | 5.858 | 5.231 | 0.627 |
| random_1007 | 5.676 | 5.133 | 0.544 |
| random_1008 | 5.782 | 5.041 | 0.741 |
| random_1009 | 5.891 | 5.330 | 0.560 |

Aggregate over random support draws:

| stat | sparse_rf | r101_feature_ridge | r101_minus_sparse |
|---|---|---|---|
| mean | 5.202 | 5.931 | 0.729 |
| std | 0.368 | 0.261 | 0.571 |
| min | 4.845 | 5.541 | -0.612 |
| max | 6.153 | 6.465 | 1.620 |

Random support geometry matters a lot. The fixed linspace support result from the earlier report was `sparse_rf=4.662` and `r101_feature_ridge=4.593`. Under random supports, both models are worse on average, and the R101 ridge head is not consistently better than sparse RF. That means the previous small R101 edge is support-layout dependent rather than a robust dominance result.

## 4. Repeat-AP Noise Floor Beside Model RMSE

**Goal.** Put model RMSE next to a repeat-AP label noise floor.

**How.** For AP locations that were measured in multiple setups, I compared the RSSI labels point-by-point between setup pairs with the same H5 AP point. This is not a model prediction; it is the disagreement between two real measurements that nominally share the same AP location and geometry.

**Results.**

Repeat-AP pairs:

| ap_point | setup_a | setup_b | overlap_n | repeat_rmse | repeat_mae | bias_a_minus_b | max_abs_diff |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 13 | 50 | 4.499 | 3.600 | 0.680 | 13.000 |
| 18 | 4 | 14 | 52 | 5.628 | 4.635 | 2.019 | 14.000 |
| 26 | 6 | 15 | 49 | 4.257 | 3.306 | 0.204 | 12.000 |
| 30 | 5 | 16 | 48 | 6.149 | 4.521 | 1.729 | 19.000 |

Model RMSE beside repeat-AP disagreement:

| comparison | n | rmse | mae | bias_pred_minus_true |
|---|---|---|---|---|
| sparse_rf | 927 | 4.662 | 3.614 | -0.172 |
| r101_feature_ridge | 927 | 4.593 | 3.545 | 0.051 |
| repeat_ap_pairs | 199 | 5.191 | 4.020 | nan |

The repeated-AP disagreement is 5.19 dB pooled across repeated pairs, which is larger than both model RMSEs on the sampling-assisted benchmark. That does not mean the models are below physical noise; the model test set and repeat-pair comparison are not identical tasks. It does mean that same-AP repeatability is bad enough to explain why pushing far below 4.6 dB is difficult without setup-specific calibration or better labels.

## 5. Point-Local Decoder/Encoder Features

**Goal.** Replace global pooled encoder features with point-local decoder/encoder features sampled at the query pixel.

**How.** I kept the fixed linspace supports and same outer AP split. For the R101 ridge head I used sparse engineered features, `r101_pl_pred`, the 2,048-channel local bottleneck vector sampled at the query pixel's corresponding 8x8 encoder cell, and the 16-channel local decoder vector sampled at the exact 256x256 query pixel. This replaces the old globally pooled 2,048 encoder features. I also refit `sparse_rf` in the same loop as a sanity check.

**Results.**

| model | query_n | pooled_rmse | pooled_mae | mean_fold_rmse | std_fold_rmse | source |
|---|---|---|---|---|---|---|
| r101_feature_ridge | 927 | 4.593 | 3.545 | 4.498 | 0.574 | cached_global_or_sparse |
| sparse_rf | 927 | 4.662 | 3.614 | 4.608 | 0.942 | cached_global_or_sparse |
| r101_pointlocal_ridge | 927 | 9.395 | 5.464 | 7.823 | 4.153 | pointlocal_check |
| sparse_rf_refit_fixed_support | 927 | 4.662 | 3.614 | 4.608 | 0.942 | pointlocal_check |

The point-local replacement hurts badly: `r101_pointlocal_ridge` is around 9.4 dB RMSE, far worse than both `sparse_rf` and the cached global-pooled `r101_feature_ridge`. The local decoder/bottleneck activations do not transfer as a useful point descriptor in this dataset with this linear head. The earlier global pooled features seem to help mostly as setup/map-level calibration features, not as local spatial descriptors.

## 6. Real Distance Scale Versus Synthetic Normalization

**Goal.** Quantify whether the real geometry is out-of-scale for the synthetic distance normalization, and whether model errors vary across that normalized distance range.

**How.** I recomputed the R101 distance-channel normalization used by `build_input_tensor`: `z = (ln(distance_m + 1e-6) - 3.539) / 0.797`. The synthetic distance mean corresponds to `exp(3.539) = 34.44 m`. I summarized both observed measurement points and all 256x256 pixels generated for each real setup, then recomputed cached `sparse_rf` and `r101_feature_ridge` RMSE by normalized-distance bin.

**Results.**

Distribution relative to synthetic normalization:

| population | n | distance_min | distance_p25 | distance_median | distance_p75 | distance_max | z_min | z_p25 | z_median | z_p75 | z_max | frac_below_synth_mean | frac_below_minus2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| observed_points | 1027 | 0.000 | 3.000 | 4.472 | 8.602 | 16.651 | -21.783 | -3.063 | -2.562 | -1.741 | -0.912 | 1.000 | 0.691 |
| all_256x256_pixels_by_setup | 1310720 | 0.007 | 5.343 | 8.261 | 10.923 | 19.759 | nan | nan | -1.812 | nan | nan | 1.000 | 0.404 |

Query RMSE by synthetic-normalized distance bin:

| synthetic_z_bin | query_n | distance_median | sparse_rf_rmse | r101_feature_ridge_rmse | r101_minus_sparse |
|---|---|---|---|---|---|
| <-4 | 111 | 1.000 | 6.045 | 6.214 | 0.169 |
| -4..-3 | 163 | 2.236 | 4.395 | 4.404 | 0.009 |
| -3..-2 | 368 | 4.472 | 3.716 | 3.590 | -0.126 |
| -2..-1 | 283 | 10.735 | 5.264 | 5.095 | -0.169 |
| -1..0 | 2 | 16.008 | 2.884 | 3.136 | 0.252 |

The real geometry is indeed far left of the synthetic distance distribution. Observed points have median normalized distance around -2.56, and every observed point is below the synthetic mean. The full image grids are also mostly below the synthetic mean; averaging over setups, about 100.0% of pixels are below it. This reinforces that R101 is being run in an extrapolative low-distance regime, not merely with missing reflectance/transmittance. The bin table does not show a simple monotonic failure pattern, but it does show that the entire evaluated benchmark lives in a narrow, low-z slice of the synthetic normalization.

## Conclusion

**Detailed analysis.** The new checks make the problem look less like "R101 is failing to learn radio propagation" and more like "this benchmark gives R101 very little stable residual signal after sparse real measurements, and several labels violate the AP geometry the model is asked to use."

The strongest new evidence is the AP-label audit. Setup 16 and setup 2 both have H5 AP labels that are adjacent to, but not at, the strongest/raw-nearest point. Setup 16 says AP point 30, yet point 29 is -26 dBm and raw-nearest while point 30 is -44 dBm. Setup 2 says AP point 3, yet point 2 is -19 dBm and raw-nearest while point 3 is -33 dBm. A deterministic geometry/pathloss model is structurally punished by that kind of one-point AP ambiguity. Sparse RF is partly protected because it uses real support RSSI statistics and can learn dataset quirks directly from real labels.

Removing near-AP query points clarifies the effect. With all queries, R101 ridge only beats sparse RF by about 0.07 dB. After excluding points within 1 m of the H5 AP label, the R101 edge grows to 0.111 dB. That is not huge, but it is the direction expected if the worst AP-adjacent labels are suppressing R101's geometry signal.

The random-support experiment weakens the case that R101 is robustly useful under the current sampling-assisted protocol. Across 10 random support draws, mean RMSE was 5.202 for sparse RF and 5.931 for R101 ridge, with mean R101-minus-sparse of 0.729. So the fixed linspace support result was not a broad win; it was a small win under one support geometry. The sparse support points dominate the conditional problem, and changing them changes both models enough to swamp the old 0.07 dB headline.

The repeat-AP noise floor stays central. The pooled repeat-AP disagreement is 5.191 dB, larger than the model RMSEs reported for the sampling-assisted query benchmark. Since the repeat-pair task is not identical to model evaluation, I would not treat 5.19 dB as a hard lower bound. But it is clear evidence that setup-level measurement variation is on the same scale as the residuals we are trying to explain.

The point-local feature check is the most negative result for R101 as a local field model here. Replacing global pooled encoder features with local bottleneck plus decoder features produced 9.395 dB RMSE. That says the useful part of the frozen R101 transfer is not a clean local descriptor at the receiver point. In this real-data setup, the global pooled features are more plausibly acting as map/setup calibration variables, while `r101_pl_pred` supplies a smooth pathloss-like trend that overlaps heavily with distance and sparse interpolation.

The distance-normalization check adds another structural reason for weak transfer. Observed real distances have median 4.472 m and median synthetic-normalized z-score -2.562; all observed points are below the synthetic mean distance. Even all generated real image pixels are mostly below the synthetic mean, with mean fraction below the synthetic mean of 100.0%. So the frozen R101 is operating in a short-range regime that is far from the synthetic normalization center, in addition to missing material channels and AP-label uncertainty.

My updated opinion: R101 is useful, but not in the way we hoped. It contains real propagation signal, yet the current dataset/protocol makes that signal marginal after five real support measurements. The best existing R101 model is worth keeping as an auxiliary feature head, but I would not present it as materially superior to sparse RF on this benchmark. Before investing in more R101 ablations, I would fix or model AP placement uncertainty, evaluate support-draw distributions instead of a single linspace support set, rescale or fine-tune the distance channel for this real geometry, and separate "global setup calibration" features from truly local R101 field features. If those changes still leave R101 at parity, then the practical value of this pretrained checkpoint for this real dataset is limited to small regularizing corrections rather than strong reconstruction gains.



Question : read the repo and the unified_r101_sparse_baseline_report.md to familiarize yourself with what I am doing here, analyze why r101 isn't giving
  significantly better results then a dumb sparse_rf baseline. relevant repos to read about the training of the model is ~/mlsp_wair_d and syntehtic data generation is
  implemented in ~/synth_gen. I want you to dig much deeper the surface level conclusions like maybe synthetic isn't close to this data or maybe we're missing reflectance
  and transmittance in inputs. Never in a million years should have the sparse_rf given such a close, in practice identical, results to a resnet 101 model train on 125M
  synthetic samples with 1/3 of them not having reflectance or trasnmittance and having to rely only on the floor plan and sparse measureemnts just like in our case.
  Analyze where the model predicts incorrectly, weather you see intuitively why the labels were what they were and what the remaining residual of the predictions and the
  ground truths for both the sparse_rf and the best version of r101-ridge are and why they are like that.""" you asnwered to me with

Answer: I dug through realdataset, experiments/sampling_assisted_r101.py, the MLSP dataloader/featurizer, and the synthetic generator. My read is:

  The sparse RF is not actually a dumb baseline in this setup. It is a real-data supervised conditional interpolator with 5 target-map measurements, trained on the same
  physical floor/grid and same measurement campaign. The R101 is a frozen synthetic pathloss model whose point-specific contribution here is mostly one scalar,
  r101_pl_pred; the 2048 encoder features are globally pooled once per setup, so they cannot describe “this particular query point is weird” except through setup-level
  calibration.

  The main reasons it ends up close:

  1. The real labels have a 4-6 dB repeatability floor. Same AP point, same grid, different setup gives pointwise disagreement already comparable to final RMSE:

  | repeated AP | setups | pointwise RMSE diff |
  |---|---:|---:|
  | AP 1 | 1 vs 13 | 4.50 dB |
  | AP 18 | 4 vs 14 | 5.63 dB |
  | AP 26 | 6 vs 15 | 4.26 dB |
  | AP 30 | 5 vs 16 | 6.15 dB |

  Example: setup 5 and 16 both use AP point 30. At point 30 itself, setup 5 is -25 dBm, setup 16 is -44 dBm; at point 45 they are -61 vs -73. No frozen geometry model can
  infer that without enough real support points.

  2. The worst residuals are label/measurement discontinuities, not missing “deep” geometry. Setup 6 point 35 is only 1 m from AP point 26 but is -53 dBm; nearest support
     at the AP is -27 dBm. Sparse RF predicts -40.8, R101-ridge predicts -33.8, so R101 is worse because the label violates the smooth physical field it learned. Setup 16
     has AP point 30 at -44 dBm while adjacent point 29 is -26 dBm; both models miss that.
  3. The real geometry is badly out-of-scale for the synthetic distance normalization. Synthetic normalization has d_log_mean=3.539, i.e. exp(mean)=34.4 m. Real
     measurement distances are mostly tiny: median 4.47 m, 75th percentile 8.60 m, max 16.65 m. In normalized distance units, real measurement points have median z-score
     -2.56; even all 256x256 real pixels are mostly below the synthetic mean. So R101 is being run far left of its distance distribution, not just with missing material
     channels.
  4. R101 is useful, but its useful signal overlaps sparse RF. Raw r101_pl_pred has strong correlation with true RSSI/pathloss (corr ~= -0.855 with RSSI), so the network
     is not dead. But after sparse RF, the residual has almost no relation to R101. In my merge, corr(r101_pl_pred, true - sparse_rf) is only 0.038; the report’s
     calibrated residual table says the same thing qualitatively. R101 helps 461 query points and hurts 466; the RMSE gain comes from a few large corrections, not broad
     dominance.

  Residual summary for the two models:

  | model | RMSE | MAE | bias pred-true | p95 abs err | max abs err |
  |---|---:|---:|---:|---:|---:|
  | sparse_rf | 4.662 | 3.614 | -0.172 | 9.183 | 25.382 |
  | r101_feature_ridge | 4.593 | 3.545 | 0.051 | 9.044 | 19.202 |

  Pattern: both compress extremes. For very weak labels < -80 dBm, both predict too strong by about +5 dB. For very strong labels > -30 dBm, both predict too weak, and
  R101-ridge is worse there. R101 improves the -70..-60 dBm bin a lot (5.47 -> 3.72 RMSE) and improves office points (4.60 -> 4.28), but hurts elevator points (4.67 ->
  5.37) and near/AP-adjacent anomalies.

  The biggest conceptual miss in the current R101 transfer experiment is not just “synthetic mismatch” or “missing r/t”. It is that the frozen model predicts a
  deterministic physical pathloss surface, while the scoring labels contain setup-specific measurement offsets, AP/receiver quirks, possible point mapping anomalies, and
  local discontinuities of roughly the same magnitude as the final RMSE. Sparse RF wins more than it “should” because it is allowed to learn those quirks from real labels
  and use 5 real samples from the held-out setup.

  Concrete next checks I would prioritize:

  1. Remove the points closest to the AP and observ how the RMSE changes
  2. Audit setup 16 and setup 2 point mapping/AP labels. Setup 16 AP point 30 being -44 while point 29 is -26 is suspicious.
  3. Train/evaluate with random support draws, not fixed linspace supports, to see whether R101 only helps particular support geometries.
  4. Report a repeat-AP noise floor next to model RMSE; it explains why 4.6 dB may already be near the limit for this benchmark.
  5. Replace global pooled encoder features with point-local decoder/encoder features sampled at the query pixel.
  6. The real geometry is badly out-of-scale for the synthetic distance normalization. Synthetic normalization has d_log_mean=3.539, i.e. exp(mean)=34.4 m. Real measurement distances are mostly tiny: median 4.47 m, 75th percentile 8.60 m, max 16.65 m. In normalized distance units, real measurement points have median z-score -2.56; even all 256x256 real pixels are mostly below the synthetic mean. So R101 is being run far left of its distance distribution, not just with missing material. Address this by mittigating this effect in the level of preprocessing of these samples and see weather it helps get better results.


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


