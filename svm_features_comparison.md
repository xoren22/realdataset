# SVM Features Comparison

Generated from `experiments/comparing_feature_sets_svm.py`.

This repeats the ridge feature-set comparison with linear epsilon-insensitive
SVM-style regression trained by SGD. It uses the same 10 random sparse-support draws, the same
leave-one-AP-location-out folds, and the cached R101 decoded prediction plus
global encoder features from:

`runs/comparing_feature_sets_ridge/r101_global_feature_cache/`

No R101 inference is run by this experiment.

All RMSE/MAE values are in dBm.

## Protocol

- Model: `StandardScaler` + `SGDRegressor(loss="epsilon_insensitive", penalty="l2")`.
- Kernel: linear.
- Epsilon: `0.1` dB.
- Regularization sweep:
  - low-dimensional feature sets: `alpha = 1e-05, 3e-05, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1`
  - encoder feature sets: `alpha = 0.001, 0.003, 0.01, 0.03, 0.1`
- Support draws: seeds `1000..1009`, 5 uniformly sampled support points per setup.
- Evaluation: query points only; support points are excluded from train/validation scoring.
- Split: leave-one-AP-location-out, 16 folds per support draw.

`alpha` is L2 regularization strength: larger `alpha` means stronger regularization.

## Feature Sets

| model | features |
|---|---|
| `sparse12_svm` | 12 non-constant sparse engineered features |
| `sparse12_plus_r101_pl_svm` | sparse12 + `r101_pl_pred` |
| `sparse12_plus_encoder2048_svm` | sparse12 + global pooled R101 encoder2048 |
| `sparse12_plus_r101_pl_encoder2048_svm` | sparse12 + `r101_pl_pred` + encoder2048 |
| `r101_pl_encoder2048_svm` | `r101_pl_pred` + encoder2048 |

## Best Alpha Results

Mean and standard deviation are across the 10 random support draws.

| model | feature_count | alpha | train_RMSE | val_RMSE | val_mae_mean |
|---|---|---|---|---|---|
| sparse12_plus_r101_pl_svm | 13 | 0.010 | 5.266 ± 0.117 | 5.802 ± 0.191 | 4.371 |
| sparse12_plus_r101_pl_encoder2048_svm | 2061 | 0.010 | 5.374 ± 0.127 | 6.158 ± 0.285 | 4.686 |
| sparse12_svm | 12 | 0.010 | 5.664 ± 0.116 | 6.307 ± 0.244 | 4.793 |
| sparse12_plus_encoder2048_svm | 2060 | 0.010 | 5.720 ± 0.146 | 6.563 ± 0.352 | 5.025 |
| r101_pl_encoder2048_svm | 2049 | 0.003 | 7.264 ± 0.258 | 7.614 ± 0.599 | 5.826 |

## Per-Draw Validation RMSE

Each column uses that model's selected best alpha above.

| support seed | sparse12 | sparse12 + r101_pl | sparse12 + encoder2048 | sparse12 + r101_pl + encoder2048 | r101_pl + encoder2048 |
|---:|---:|---:|---:|---:|---:|
| 1000 | 5.958 | 5.656 | 6.204 | 6.108 | 7.352 |
| 1001 | 6.625 | 5.993 | 6.864 | 6.429 | 7.652 |
| 1002 | 6.565 | 6.195 | 6.599 | 6.282 | 7.904 |
| 1003 | 6.594 | 5.987 | 6.762 | 6.309 | 7.406 |
| 1004 | 6.164 | 5.649 | 6.845 | 6.589 | 8.862 |
| 1005 | 6.033 | 5.682 | 5.940 | 5.660 | 6.723 |
| 1006 | 6.294 | 5.786 | 6.252 | 5.901 | 7.436 |
| 1007 | 6.090 | 5.722 | 6.350 | 5.845 | 7.097 |
| 1008 | 6.298 | 5.628 | 6.826 | 6.166 | 7.491 |
| 1009 | 6.454 | 5.720 | 6.986 | 6.291 | 8.216 |

## Alpha Sensitivity

These are means across the 10 support draws. Validation RMSE is pooled per draw,
then averaged across draws.

| model | alpha | train RMSE | val RMSE |
|---|---:|---:|---:|
| `sparse12_svm` | 1e-05 | 5.616 | 6.348 |
| `sparse12_svm` | 3e-05 | 5.598 | 6.335 |
| `sparse12_svm` | 0.0001 | 5.595 | 6.328 |
| `sparse12_svm` | 0.0003 | 5.596 | 6.321 |
| `sparse12_svm` | 0.001 | 5.605 | 6.357 |
| `sparse12_svm` | 0.003 | 5.620 | 6.336 |
| `sparse12_svm` | 0.01 | 5.664 | 6.307 |
| `sparse12_svm` | 0.03 | 5.793 | 6.318 |
| `sparse12_svm` | 0.1 | 6.773 | 7.113 |
| `sparse12_plus_r101_pl_svm` | 1e-05 | 5.268 | 5.856 |
| `sparse12_plus_r101_pl_svm` | 3e-05 | 5.243 | 5.874 |
| `sparse12_plus_r101_pl_svm` | 0.0001 | 5.231 | 5.855 |
| `sparse12_plus_r101_pl_svm` | 0.0003 | 5.228 | 5.849 |
| `sparse12_plus_r101_pl_svm` | 0.001 | 5.230 | 5.844 |
| `sparse12_plus_r101_pl_svm` | 0.003 | 5.239 | 5.825 |
| `sparse12_plus_r101_pl_svm` | 0.01 | 5.266 | 5.802 |
| `sparse12_plus_r101_pl_svm` | 0.03 | 5.367 | 5.844 |
| `sparse12_plus_r101_pl_svm` | 0.1 | 6.219 | 6.549 |
| `sparse12_plus_encoder2048_svm` | 0.001 | 7.811 | 7.934 |
| `sparse12_plus_encoder2048_svm` | 0.003 | 6.045 | 6.805 |
| `sparse12_plus_encoder2048_svm` | 0.01 | 5.720 | 6.563 |
| `sparse12_plus_encoder2048_svm` | 0.03 | 6.171 | 6.984 |
| `sparse12_plus_encoder2048_svm` | 0.1 | 9.956 | 10.416 |
| `sparse12_plus_r101_pl_encoder2048_svm` | 0.001 | 7.598 | 7.534 |
| `sparse12_plus_r101_pl_encoder2048_svm` | 0.003 | 5.716 | 6.485 |
| `sparse12_plus_r101_pl_encoder2048_svm` | 0.01 | 5.374 | 6.158 |
| `sparse12_plus_r101_pl_encoder2048_svm` | 0.03 | 5.687 | 6.508 |
| `sparse12_plus_r101_pl_encoder2048_svm` | 0.1 | 9.389 | 9.868 |
| `r101_pl_encoder2048_svm` | 0.001 | 8.625 | 8.527 |
| `r101_pl_encoder2048_svm` | 0.003 | 7.264 | 7.614 |
| `r101_pl_encoder2048_svm` | 0.01 | 7.209 | 7.672 |
| `r101_pl_encoder2048_svm` | 0.03 | 8.361 | 8.778 |
| `r101_pl_encoder2048_svm` | 0.1 | 13.278 | 13.484 |

## Interpretation

The SVM results tell the same main story as ridge. The decoded R101 prediction
is useful when added to sparse features: `sparse12 + r101_pl` is the best SVM
feature set. The 2048 encoder vector does not provide a reliable improvement:
adding it to `sparse12 + r101_pl` lowers train error slightly but worsens
validation, and adding it to sparse12 without `r101_pl_pred` is worse than
sparse12 alone.

Compared with tuned ridge, the best SVM is slightly worse:

- best tuned ridge: `sparse12 + r101_pl`, RMSE `5.799 ± 0.188`
- best tuned SVM: `sparse12 + r101_pl`, RMSE `5.802 ± 0.191`

So replacing ridge with linear SVM does not change the conclusion: R101's
useful contribution is the decoded prediction scalar, while encoder2048 mostly
adds capacity without dependable validation gain.

## Artifacts

- `experiments/comparing_feature_sets_svm.py`
- `runs/comparing_feature_sets_svm/svm_feature_set_alpha_fold_metrics.csv`
- `runs/comparing_feature_sets_svm/svm_feature_set_alpha_by_seed_alpha_summary.csv`
- `runs/comparing_feature_sets_svm/svm_feature_set_alpha_overall_by_alpha.csv`
- `runs/comparing_feature_sets_svm/svm_feature_set_alpha_best_by_model.csv`
- `runs/comparing_feature_sets_svm/manifest.json`
