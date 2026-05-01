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

