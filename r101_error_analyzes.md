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
