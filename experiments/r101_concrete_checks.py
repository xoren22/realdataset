#!/usr/bin/env python
"""Focused follow-up checks for the r101 vs sparse_rf error analysis."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.base import clone

from experiments.baselines import load_samples
from experiments.sampling_assisted_r101 import (
    IMAGE_SIZE,
    RIDGE_ALPHA,
    SPARSE_COLS,
    SUPPORT_COUNT,
    WALL_THICKNESS_PX,
    add_sparse_features,
    build_input_tensor,
    c_from_support,
    fit_ridge,
    folds,
    forward_r101_batch,
    load_encoder_unet_model,
    point_grid_locations,
    point_to_pixel,
    ridge_feature_cols,
    setup_frequencies,
    support_table,
    train_models,
)
from src.paths import REPO_ROOT, RSSI_CSV, WIFI_H5, require_data


OUT_DIR = REPO_ROOT / "runs" / "r101_concrete_checks"
REPORT = REPO_ROOT / "r101_concrete_checks_report.md"
MODEL_NAMES = ("sparse_rf", "r101_feature_ridge")
RANDOM_SUPPORT_SEEDS = tuple(range(1000, 1010))


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_pred - y_true)))


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    if df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(map(str, df.columns)) + " |"]
    lines.append("|" + "|".join("---" for _ in df.columns) + "|")
    for row in df.itertuples(index=False):
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.{digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def append_section(title: str, body: str, reset: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if reset or not REPORT.exists():
        REPORT.write_text(
            "# R101 Concrete Follow-up Checks\n\n"
            "This report follows the five prioritized checks from "
            "`r101_error_analyzes.md`. The comparisons are intentionally "
            "limited to `sparse_rf` and the best existing R101-based model, "
            "`r101_feature_ridge`, except check 5 where the R101 ridge head is "
            "rerun with point-local features in place of global pooled features. "
            "A sixth distance-normalization check was added afterward because "
            "the real geometry may be far outside the synthetic distance scale.\n\n"
        )
    with REPORT.open("a") as f:
        f.write(f"## {title}\n\n{body.strip()}\n\n")


def query_predictions_with_features() -> pd.DataFrame:
    cached = REPO_ROOT / "runs" / "deep_residual_analysis" / "query_predictions_with_features.csv"
    if cached.exists():
        return pd.read_csv(cached)

    preds = pd.read_csv(REPO_ROOT / "runs" / "sampling_assisted_r101_predictions.csv")
    preds = preds[preds["model"].isin(MODEL_NAMES)]
    wide = preds.pivot_table(index=["fold", "sample_id", "setup", "point", "holdout_ap_point"], columns="model", values="y_pred").reset_index()
    truth = preds.drop_duplicates("sample_id")[["sample_id", "y_true"]]
    samples = pd.read_csv(REPO_ROOT / "runs" / "sampling_assisted_r101_samples.csv")
    keep = [
        "sample_id",
        "ap_point",
        "ap_x_m",
        "ap_y_m",
        "mx_m",
        "my_m",
        "room_meas",
        "room_ap",
        "same_room",
        "distance_m",
        "log_distance_m",
        "is_support",
        "support_mean_rssi",
        "support_std_rssi",
        "nearest_support_dist_m",
        "nearest_support_rssi",
        "idw_support_rssi",
        "r101_pl_pred",
        "sparse_pl_at_point",
    ]
    features = samples.drop_duplicates("sample_id")[keep]
    return wide.merge(truth, on="sample_id").merge(features, on="sample_id")


def check_near_ap() -> None:
    df = query_predictions_with_features()
    thresholds = [None, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    baseline = {}
    rows = []
    for threshold in thresholds:
        if threshold is None:
            kept = df.copy()
            label = "none"
        else:
            kept = df[df["distance_m"] > threshold].copy()
            label = f">{threshold:g}m"
        row = {
            "exclude_distance": label,
            "kept_n": int(len(kept)),
            "removed_n": int(len(df) - len(kept)),
        }
        for model in MODEL_NAMES:
            score = rmse(kept["y_true"], kept[model])
            if threshold is None:
                baseline[model] = score
            row[f"{model}_rmse"] = score
            row[f"{model}_delta"] = score - baseline[model]
        row["r101_minus_sparse"] = row["r101_feature_ridge_rmse"] - row["sparse_rf_rmse"]
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "check1_near_ap_rmse.csv", index=False)

    near = df[df["distance_m"] <= 1.0].copy()
    near["sparse_abs"] = np.abs(near["sparse_rf"] - near["y_true"])
    near["r101_abs"] = np.abs(near["r101_feature_ridge"] - near["y_true"])
    top = near.sort_values(["r101_abs", "sparse_abs"], ascending=False).head(10)
    top = top[
        [
            "setup",
            "point",
            "holdout_ap_point",
            "distance_m",
            "y_true",
            "sparse_rf",
            "r101_feature_ridge",
            "sparse_abs",
            "r101_abs",
        ]
    ]
    top.to_csv(OUT_DIR / "check1_near_ap_top_errors.csv", index=False)

    body = f"""
**Goal.** Test whether AP-adjacent labels are a special error source and whether removing them changes the `sparse_rf` vs `r101_feature_ridge` comparison.

**How.** I reused the existing out-of-fold query predictions, removed query points by true AP distance threshold, and recomputed pooled RMSE for only `sparse_rf` and `r101_feature_ridge`. Support points were already excluded by the original prediction table.

**Results.**

{md_table(out)}

Worst query errors among points within 1 m of the H5 AP label:

{md_table(top, digits=2)}

The near-AP points are disproportionately bad, especially for R101. Removing only the exact AP point leaves the ranking unchanged, but removing points within 1 m cuts R101 RMSE much more than sparse RF and increases the R101 edge from about 0.07 dB to about 0.24 dB. This supports the suspicion that AP-adjacent label/pathology effects are hiding some of the useful R101 signal.
"""
    append_section("1. Remove Points Closest To The AP", body, reset=True)


def parse_distance_m(value: object) -> float:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else np.nan


def check_setup_mapping() -> None:
    samples = load_samples()
    raw = pd.read_csv(RSSI_CSV)
    raw["setup"] = raw["Setup"].astype(int)
    raw["rssi"] = raw["Strength"].map(lambda x: float(re.search(r"-?\d+", str(x)).group()))
    raw["raw_distance_m"] = raw["Distance"].map(parse_distance_m)

    audit_rows = []
    neighborhood_rows = []
    with h5py.File(WIFI_H5, "r") as f:
        h5_ap = f["ap_locations"][:].astype(int)
        h5_data = f["data"][:]
        h5_indices = f["indices"][:]
        for setup in (2, 16):
            setup_samples = samples[samples["setup"] == setup].copy()
            setup_raw = raw[raw["setup"] == setup].reset_index(drop=True)
            setup_samples = setup_samples.merge(
                setup_raw[["rssi", "raw_distance_m"]].rename(columns={"rssi": "raw_rssi"}),
                left_on="source_row",
                right_index=True,
                how="left",
            )
            h5_match = True
            for row in setup_samples.itertuples(index=False):
                loc = np.argwhere(h5_indices[setup - 1] == row.point)
                if loc.shape != (1, 2):
                    h5_match = False
                    continue
                rr, cc = loc[0]
                if float(h5_data[setup - 1, rr, cc]) != float(row.rssi):
                    h5_match = False
            strongest = setup_samples.loc[setup_samples["rssi"].idxmax()]
            raw_nearest = setup_samples.loc[setup_samples["raw_distance_m"].idxmin()]
            ap_sample = setup_samples[setup_samples["point"] == int(h5_ap[setup - 1])].iloc[0]
            zero_dist = setup_samples.loc[setup_samples["distance_m"].idxmin()]
            audit_rows.append(
                {
                    "setup": setup,
                    "h5_ap_point": int(h5_ap[setup - 1]),
                    "zero_geom_point": int(zero_dist.point),
                    "ap_rssi": float(ap_sample.rssi),
                    "ap_raw_distance": float(ap_sample.raw_distance_m),
                    "strongest_point": int(strongest.point),
                    "strongest_rssi": float(strongest.rssi),
                    "strongest_geom_distance": float(strongest.distance_m),
                    "raw_nearest_point": int(raw_nearest.point),
                    "raw_nearest_distance": float(raw_nearest.raw_distance_m),
                    "raw_nearest_rssi": float(raw_nearest.rssi),
                    "csv_matches_h5_grid": bool(h5_match),
                }
            )

            focus_points = sorted(set(range(max(1, int(h5_ap[setup - 1]) - 4), min(53, int(h5_ap[setup - 1]) + 5) + 1)))
            for row in setup_samples[setup_samples["point"].isin(focus_points)].itertuples(index=False):
                neighborhood_rows.append(
                    {
                        "setup": setup,
                        "point": int(row.point),
                        "h5_ap_point": int(h5_ap[setup - 1]),
                        "geom_distance_m": float(row.distance_m),
                        "raw_distance_m": float(row.raw_distance_m),
                        "rssi": float(row.rssi),
                    }
                )

    audit = pd.DataFrame(audit_rows)
    neighborhood = pd.DataFrame(neighborhood_rows)
    audit.to_csv(OUT_DIR / "check2_setup2_16_mapping_audit.csv", index=False)
    neighborhood.to_csv(OUT_DIR / "check2_setup2_16_ap_neighborhood.csv", index=False)

    body = f"""
**Goal.** Audit setup 16 and setup 2 point mapping/AP labels, focusing on the suspicious AP-adjacent values.

**How.** I compared `RSSI_raw_data.csv`, the H5 `ap_locations`, the H5 `indices` grid, and the repo's sequential raw-row-to-point mapping. I also parsed the raw `Distance` column, which appears to be the phone's reported WiFi distance estimate, and compared it to geometric distance from the H5 AP point.

**Results.**

Summary:

{md_table(audit)}

AP-neighborhood rows:

{md_table(neighborhood)}

The raw CSV values and H5 grid values agree, so this is not a CSV-to-H5 transcription error. The suspicious part is the AP label itself. In setup 16, H5 says AP point 30, but point 29 is both the strongest label (-26 dBm) and the raw-nearest point (~0.2 m); point 30 is -44 dBm and raw distance ~1.5 m. Setup 2 has the same pattern: H5 says AP point 3, but point 2 is strongest (-19 dBm) and raw-nearest (~0.1 m), while point 3 is -33 dBm and raw distance ~0.4 m. This makes setup 16 and setup 2 look like one-grid-point AP label errors or at least AP placement uncertainty at exactly the scale that hurts deterministic pathloss models.
"""
    append_section("2. Audit Setup 16 And Setup 2 Mapping", body)


def random_support_table(samples: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for setup, group in samples.groupby("setup"):
        support_ids = set(rng.choice(group["sample_id"].to_numpy(dtype=int), size=SUPPORT_COUNT, replace=False))
        for row in group.itertuples(index=False):
            rows.append(
                {
                    "sample_id": int(row.sample_id),
                    "setup": int(setup),
                    "point": int(row.point),
                    "is_support": int(row.sample_id in support_ids),
                }
            )
    return pd.DataFrame(rows)


def extract_global_r101_features_for_fold(
    model: torch.nn.Module,
    samples: pd.DataFrame,
    support_samples: pd.DataFrame,
    c_dbm: float,
    fold: int,
    holdout_ap_point: int,
) -> pd.DataFrame:
    point_locations = point_grid_locations()
    freqs = setup_frequencies()
    tensors = []
    setup_ids = []
    sparse_grids = {}
    for setup in sorted(samples["setup"].unique()):
        setup_support = support_samples[(support_samples["setup"] == setup) & (support_samples["is_support"] == 1)]
        tensor, _, sparse_grid = build_input_tensor(
            setup=int(setup),
            samples=samples,
            setup_support=setup_support,
            point_locations=point_locations,
            freq_mhz=freqs[int(setup)],
            wall_thickness_px=WALL_THICKNESS_PX,
            c_dbm=c_dbm,
        )
        tensors.append(tensor)
        setup_ids.append(int(setup))
        sparse_grids[int(setup)] = sparse_grid

    pred_pl_batch, pooled_batch = forward_r101_batch(model, tensors)
    rows = []
    for batch_idx, setup in enumerate(setup_ids):
        sparse_grid = sparse_grids[setup]
        setup_rows = samples[samples["setup"] == setup]
        pooled = pooled_batch[batch_idx]
        for row in setup_rows.itertuples(index=False):
            grid_row, grid_col = point_locations[int(row.point)]
            pix_y, pix_x = point_to_pixel(grid_row, grid_col)
            item = {
                "sample_id": int(row.sample_id),
                "setup": int(setup),
                "point": int(row.point),
                "r101_pl_pred": float(pred_pl_batch[batch_idx, pix_y, pix_x]),
                "fold": int(fold),
                "holdout_ap_point": int(holdout_ap_point),
                "c_fold_dbm": float(c_dbm),
                "sparse_pl_at_point": float(sparse_grid[grid_row, grid_col]),
                "wall_thickness_px": WALL_THICKNESS_PX,
                "rt_mode": "zero_rt",
                "rt_impute_n": 1,
            }
            item.update({f"r101_f{i}": float(value) for i, value in enumerate(pooled)})
            rows.append(item)
    return pd.DataFrame(rows)


def evaluate_support_scheme(
    samples: pd.DataFrame,
    supports: pd.DataFrame,
    model: torch.nn.Module,
    scheme: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sparse_data = add_sparse_features(samples, supports)
    support_samples = samples.merge(supports, on=["sample_id", "setup", "point"])
    pred_rows = []
    fold_rows = []
    for fold_row in folds(samples).itertuples(index=False):
        train_support = support_samples[
            (support_samples["ap_point"] != fold_row.holdout_ap_point) & (support_samples["is_support"] == 1)
        ]
        c_fold = c_from_support(train_support)
        r101_features = extract_global_r101_features_for_fold(
            model=model,
            samples=samples,
            support_samples=support_samples,
            c_dbm=c_fold,
            fold=int(fold_row.fold),
            holdout_ap_point=int(fold_row.holdout_ap_point),
        )
        data = sparse_data.merge(r101_features, on=["sample_id", "setup", "point"], how="left")
        is_val = data["ap_point"] == fold_row.holdout_ap_point
        train = data[(~is_val) & (data["is_support"] == 0)].copy()
        val = data[is_val & (data["is_support"] == 0)].copy()

        sparse_model = clone(train_models()["sparse_rf"])
        sparse_model.fit(train[SPARSE_COLS], train["rssi"])
        sparse_pred = sparse_model.predict(val[SPARSE_COLS])
        for row, pred in zip(val.itertuples(index=False), sparse_pred):
            pred_rows.append(
                {
                    "scheme": scheme,
                    "model": "sparse_rf",
                    "fold": int(fold_row.fold),
                    "sample_id": int(row.sample_id),
                    "setup": int(row.setup),
                    "point": int(row.point),
                    "y_true": float(row.rssi),
                    "y_pred": float(pred),
                    "error": float(pred - row.rssi),
                }
            )

        cols = ridge_feature_cols("r101_feature_ridge", train)
        ridge = fit_ridge(train, cols, RIDGE_ALPHA)
        r101_pred = ridge.predict(val[cols])
        for row, pred in zip(val.itertuples(index=False), r101_pred):
            pred_rows.append(
                {
                    "scheme": scheme,
                    "model": "r101_feature_ridge",
                    "fold": int(fold_row.fold),
                    "sample_id": int(row.sample_id),
                    "setup": int(row.setup),
                    "point": int(row.point),
                    "y_true": float(row.rssi),
                    "y_pred": float(pred),
                    "error": float(pred - row.rssi),
                }
            )
        fold_rows.append({"scheme": scheme, "fold": int(fold_row.fold), "query_n": int(len(val))})
    return pd.DataFrame(pred_rows), pd.DataFrame(fold_rows)


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scheme, model), group in predictions.groupby(["scheme", "model"]):
        fold_rmse = group.groupby("fold").apply(lambda g: rmse(g["y_true"], g["y_pred"]), include_groups=False)
        rows.append(
            {
                "scheme": scheme,
                "model": model,
                "query_n": int(len(group)),
                "pooled_rmse": rmse(group["y_true"], group["y_pred"]),
                "pooled_mae": mae(group["y_true"], group["y_pred"]),
                "mean_fold_rmse": float(fold_rmse.mean()),
                "std_fold_rmse": float(fold_rmse.std()),
            }
        )
    return pd.DataFrame(rows).sort_values(["scheme", "model"]).reset_index(drop=True)


def check_random_supports() -> None:
    require_data()
    torch.set_num_threads(8)
    samples = load_samples()
    model = load_encoder_unet_model()

    all_predictions = []
    for seed in RANDOM_SUPPORT_SEEDS:
        supports = random_support_table(samples, seed)
        predictions, _folds = evaluate_support_scheme(samples, supports, model, scheme=f"random_{seed}")
        all_predictions.append(predictions)
        print(f"finished random support seed {seed}", flush=True)

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(OUT_DIR / "check3_random_support_predictions.csv", index=False)
    summary = summarize_predictions(predictions)

    paired = summary.pivot(index="scheme", columns="model", values="pooled_rmse").reset_index()
    paired["r101_minus_sparse"] = paired["r101_feature_ridge"] - paired["sparse_rf"]
    aggregate = (
        paired[["sparse_rf", "r101_feature_ridge", "r101_minus_sparse"]]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"index": "stat"})
    )
    summary.to_csv(OUT_DIR / "check3_random_support_summary.csv", index=False)
    paired.to_csv(OUT_DIR / "check3_random_support_paired_rmse.csv", index=False)
    aggregate.to_csv(OUT_DIR / "check3_random_support_aggregate.csv", index=False)

    body = f"""
**Goal.** Replace fixed linspace support points with random support draws to see whether R101 only helps for particular support geometries.

**How.** I drew 10 random five-point support sets per setup (`seeds {RANDOM_SUPPORT_SEEDS[0]}..{RANDOM_SUPPORT_SEEDS[-1]}`). For each draw I rebuilt sparse features, rebuilt the R101 sparse input channel, reran frozen R101 inference, and evaluated only `sparse_rf` and `r101_feature_ridge` with the same leave-one-AP-location-out split.

**Results.**

Per random draw:

{md_table(paired)}

Aggregate over random support draws:

{md_table(aggregate)}

Random support geometry matters a lot. The fixed linspace support result from the earlier report was `sparse_rf=4.662` and `r101_feature_ridge=4.593`. Under random supports, both models are worse on average, and the R101 ridge head is not consistently better than sparse RF. That means the previous small R101 edge is support-layout dependent rather than a robust dominance result.
"""
    append_section("3. Random Support Draws Instead Of Fixed Linspace", body)


def check_repeat_ap_noise_floor() -> None:
    samples = load_samples()
    pred = query_predictions_with_features()
    model_rows = []
    for model in MODEL_NAMES:
        model_rows.append(
            {
                "comparison": model,
                "n": int(len(pred)),
                "rmse": rmse(pred["y_true"], pred[model]),
                "mae": mae(pred["y_true"], pred[model]),
                "bias_pred_minus_true": float((pred[model] - pred["y_true"]).mean()),
            }
        )

    repeat_rows = []
    for ap_point, group in samples.groupby("ap_point"):
        setups = sorted(group["setup"].unique())
        if len(setups) < 2:
            continue
        for i, setup_a in enumerate(setups):
            for setup_b in setups[i + 1 :]:
                a = samples[samples["setup"] == setup_a][["point", "rssi"]].rename(columns={"rssi": "rssi_a"})
                b = samples[samples["setup"] == setup_b][["point", "rssi"]].rename(columns={"rssi": "rssi_b"})
                merged = a.merge(b, on="point")
                diff = merged["rssi_a"] - merged["rssi_b"]
                repeat_rows.append(
                    {
                        "ap_point": int(ap_point),
                        "setup_a": int(setup_a),
                        "setup_b": int(setup_b),
                        "overlap_n": int(len(merged)),
                        "repeat_rmse": float(np.sqrt(np.mean(diff**2))),
                        "repeat_mae": float(np.mean(np.abs(diff))),
                        "bias_a_minus_b": float(diff.mean()),
                        "max_abs_diff": float(np.max(np.abs(diff))),
                    }
                )

    repeat = pd.DataFrame(repeat_rows)
    repeat_summary = pd.DataFrame(
        [
            {
                "comparison": "repeat_ap_pairs",
                "n": int(repeat["overlap_n"].sum()),
                "rmse": float(np.sqrt(np.average(repeat["repeat_rmse"] ** 2, weights=repeat["overlap_n"]))),
                "mae": float(np.average(repeat["repeat_mae"], weights=repeat["overlap_n"])),
                "bias_pred_minus_true": float("nan"),
            }
        ]
    )
    model_summary = pd.DataFrame(model_rows)
    repeat.to_csv(OUT_DIR / "check4_repeat_ap_pairs.csv", index=False)
    pd.concat([model_summary, repeat_summary], ignore_index=True).to_csv(OUT_DIR / "check4_noise_floor_vs_models.csv", index=False)

    body = f"""
**Goal.** Put model RMSE next to a repeat-AP label noise floor.

**How.** For AP locations that were measured in multiple setups, I compared the RSSI labels point-by-point between setup pairs with the same H5 AP point. This is not a model prediction; it is the disagreement between two real measurements that nominally share the same AP location and geometry.

**Results.**

Repeat-AP pairs:

{md_table(repeat)}

Model RMSE beside repeat-AP disagreement:

{md_table(pd.concat([model_summary, repeat_summary], ignore_index=True))}

The repeated-AP disagreement is 5.19 dB pooled across repeated pairs, which is larger than both model RMSEs on the sampling-assisted benchmark. That does not mean the models are below physical noise; the model test set and repeat-pair comparison are not identical tasks. It does mean that same-AP repeatability is bad enough to explain why pushing far below 4.6 dB is difficult without setup-specific calibration or better labels.
"""
    append_section("4. Repeat-AP Noise Floor Beside Model RMSE", body)


def sample_feature_map(feature: torch.Tensor, pixel_y: int, pixel_x: int) -> np.ndarray:
    _, _, height, width = feature.shape
    yy = int(np.clip(np.floor((pixel_y + 0.5) * height / IMAGE_SIZE), 0, height - 1))
    xx = int(np.clip(np.floor((pixel_x + 0.5) * width / IMAGE_SIZE), 0, width - 1))
    return feature[0, :, yy, xx].detach().cpu().float().numpy()


def extract_pointlocal_features_for_fold(
    model: torch.nn.Module,
    samples: pd.DataFrame,
    support_samples: pd.DataFrame,
    c_dbm: float,
    fold: int,
    holdout_ap_point: int,
) -> pd.DataFrame:
    point_locations = point_grid_locations()
    freqs = setup_frequencies()
    rows = []
    for setup in sorted(samples["setup"].unique()):
        setup_support = support_samples[(support_samples["setup"] == setup) & (support_samples["is_support"] == 1)]
        tensor, _, sparse_grid = build_input_tensor(
            setup=int(setup),
            samples=samples,
            setup_support=setup_support,
            point_locations=point_locations,
            freq_mhz=freqs[int(setup)],
            wall_thickness_px=WALL_THICKNESS_PX,
            c_dbm=c_dbm,
        )
        x = torch.from_numpy(tensor[None, ...])
        with torch.inference_mode():
            features = model.unet.encoder(x)
            decoded = model.unet.decoder(features)
            pred = model.unet.segmentation_head(decoded).squeeze(1).float() * 160.0

        for row in samples[samples["setup"] == setup].itertuples(index=False):
            grid_row, grid_col = point_locations[int(row.point)]
            pix_y, pix_x = point_to_pixel(grid_row, grid_col)
            bottleneck = sample_feature_map(features[-1], pix_y, pix_x)
            decoder = sample_feature_map(decoded, pix_y, pix_x)
            item = {
                "sample_id": int(row.sample_id),
                "setup": int(setup),
                "point": int(row.point),
                "r101_pl_pred": float(pred[0, pix_y, pix_x]),
                "fold": int(fold),
                "holdout_ap_point": int(holdout_ap_point),
                "c_fold_dbm": float(c_dbm),
                "sparse_pl_at_point": float(sparse_grid[grid_row, grid_col]),
                "wall_thickness_px": WALL_THICKNESS_PX,
                "rt_mode": "zero_rt",
                "rt_impute_n": 1,
            }
            item.update({f"r101_local_f{i}": float(value) for i, value in enumerate(bottleneck)})
            item.update({f"r101_dec_f{i}": float(value) for i, value in enumerate(decoder)})
            rows.append(item)
    return pd.DataFrame(rows)


def check_pointlocal_features() -> None:
    require_data()
    torch.set_num_threads(8)
    samples = load_samples()
    supports = support_table(samples, SUPPORT_COUNT)
    sparse_data = add_sparse_features(samples, supports)
    support_samples = samples.merge(supports, on=["sample_id", "setup", "point"])
    model = load_encoder_unet_model()
    pred_rows = []

    for fold_row in folds(samples).itertuples(index=False):
        train_support = support_samples[
            (support_samples["ap_point"] != fold_row.holdout_ap_point) & (support_samples["is_support"] == 1)
        ]
        c_fold = c_from_support(train_support)
        local_features = extract_pointlocal_features_for_fold(
            model=model,
            samples=samples,
            support_samples=support_samples,
            c_dbm=c_fold,
            fold=int(fold_row.fold),
            holdout_ap_point=int(fold_row.holdout_ap_point),
        )
        data = sparse_data.merge(local_features, on=["sample_id", "setup", "point"], how="left")
        is_val = data["ap_point"] == fold_row.holdout_ap_point
        train = data[(~is_val) & (data["is_support"] == 0)].copy()
        val = data[is_val & (data["is_support"] == 0)].copy()

        sparse_model = clone(train_models()["sparse_rf"])
        sparse_model.fit(train[SPARSE_COLS], train["rssi"])
        sparse_pred = sparse_model.predict(val[SPARSE_COLS])
        for row, pred in zip(val.itertuples(index=False), sparse_pred):
            pred_rows.append(
                {
                    "model": "sparse_rf_refit_fixed_support",
                    "fold": int(fold_row.fold),
                    "sample_id": int(row.sample_id),
                    "setup": int(row.setup),
                    "point": int(row.point),
                    "y_true": float(row.rssi),
                    "y_pred": float(pred),
                    "error": float(pred - row.rssi),
                }
            )

        local_cols = SPARSE_COLS + ["r101_pl_pred"] + [
            c for c in train.columns if c.startswith("r101_local_f") or c.startswith("r101_dec_f")
        ]
        ridge = fit_ridge(train, local_cols, RIDGE_ALPHA)
        local_pred = ridge.predict(val[local_cols])
        for row, pred in zip(val.itertuples(index=False), local_pred):
            pred_rows.append(
                {
                    "model": "r101_pointlocal_ridge",
                    "fold": int(fold_row.fold),
                    "sample_id": int(row.sample_id),
                    "setup": int(row.setup),
                    "point": int(row.point),
                    "y_true": float(row.rssi),
                    "y_pred": float(pred),
                    "error": float(pred - row.rssi),
                }
            )
        print(f"finished point-local fold {int(fold_row.fold)}", flush=True)

    predictions = pd.DataFrame(pred_rows)
    predictions.to_csv(OUT_DIR / "check5_pointlocal_predictions.csv", index=False)
    summary = summarize_predictions(predictions.assign(scheme="fixed"))
    prior = pd.read_csv(REPO_ROOT / "runs" / "sampling_assisted_r101_summary.csv")
    prior = prior[prior["model"].isin(MODEL_NAMES)][
        ["model", "n_predictions", "pooled_rmse", "pooled_mae", "mean_fold_rmse", "std_fold_rmse"]
    ].rename(columns={"n_predictions": "query_n"})
    combined = pd.concat(
        [
            prior.assign(source="cached_global_or_sparse"),
            summary.drop(columns=["scheme"]).assign(source="pointlocal_check"),
        ],
        ignore_index=True,
    )
    combined.to_csv(OUT_DIR / "check5_pointlocal_summary.csv", index=False)

    body = f"""
**Goal.** Replace global pooled encoder features with point-local decoder/encoder features sampled at the query pixel.

**How.** I kept the fixed linspace supports and same outer AP split. For the R101 ridge head I used sparse engineered features, `r101_pl_pred`, the 2,048-channel local bottleneck vector sampled at the query pixel's corresponding 8x8 encoder cell, and the 16-channel local decoder vector sampled at the exact 256x256 query pixel. This replaces the old globally pooled 2,048 encoder features. I also refit `sparse_rf` in the same loop as a sanity check.

**Results.**

{md_table(combined)}

The point-local replacement hurts badly: `r101_pointlocal_ridge` is around {float(combined.loc[combined['model'] == 'r101_pointlocal_ridge', 'pooled_rmse'].iloc[0]):.1f} dB RMSE, far worse than both `sparse_rf` and the cached global-pooled `r101_feature_ridge`. The local decoder/bottleneck activations do not transfer as a useful point descriptor in this dataset with this linear head. The earlier global pooled features seem to help mostly as setup/map-level calibration features, not as local spatial descriptors.
"""
    append_section("5. Point-Local Decoder/Encoder Features", body)


def check_distance_normalization() -> None:
    from experiments.sampling_assisted_r101 import NORM, CELL_M, GRID_H, GRID_W

    samples = load_samples()
    point_dist = samples[["setup", "point", "distance_m", "rssi"]].copy()
    point_dist["ln_distance"] = np.log(point_dist["distance_m"] + 1e-6)
    point_dist["synthetic_z"] = (point_dist["ln_distance"] - NORM["d_log_mean"]) / NORM["d_log_std"]

    pixel_rows = []
    for setup, group in samples.groupby("setup"):
        ap = group.iloc[0]
        ys = ((np.arange(IMAGE_SIZE, dtype=np.float32) + 0.5) * GRID_H / IMAGE_SIZE) * CELL_M
        xs = ((np.arange(IMAGE_SIZE, dtype=np.float32) + 0.5) * GRID_W / IMAGE_SIZE) * CELL_M
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        distance = np.hypot(xx - float(ap.ap_x_m), yy - float(ap.ap_y_m)).astype(np.float32)
        z = (np.log(distance + 1e-6) - NORM["d_log_mean"]) / NORM["d_log_std"]
        pixel_rows.append(
            {
                "setup": int(setup),
                "pixel_n": int(z.size),
                "pixel_distance_min": float(distance.min()),
                "pixel_distance_p25": float(np.percentile(distance, 25)),
                "pixel_distance_median": float(np.median(distance)),
                "pixel_distance_p75": float(np.percentile(distance, 75)),
                "pixel_distance_max": float(distance.max()),
                "pixel_z_median": float(np.median(z)),
                "pixel_z_p95": float(np.percentile(z, 95)),
                "pixel_frac_below_synth_mean": float(np.mean(z < 0.0)),
                "pixel_frac_below_minus2": float(np.mean(z < -2.0)),
            }
        )

    point_summary = pd.DataFrame(
        [
            {
                "population": "observed_points",
                "n": int(len(point_dist)),
                "distance_min": float(point_dist["distance_m"].min()),
                "distance_p25": float(point_dist["distance_m"].quantile(0.25)),
                "distance_median": float(point_dist["distance_m"].median()),
                "distance_p75": float(point_dist["distance_m"].quantile(0.75)),
                "distance_max": float(point_dist["distance_m"].max()),
                "z_min": float(point_dist["synthetic_z"].min()),
                "z_p25": float(point_dist["synthetic_z"].quantile(0.25)),
                "z_median": float(point_dist["synthetic_z"].median()),
                "z_p75": float(point_dist["synthetic_z"].quantile(0.75)),
                "z_max": float(point_dist["synthetic_z"].max()),
                "frac_below_synth_mean": float((point_dist["synthetic_z"] < 0).mean()),
                "frac_below_minus2": float((point_dist["synthetic_z"] < -2).mean()),
            }
        ]
    )
    pixel_by_setup = pd.DataFrame(pixel_rows)
    pixel_summary = pd.DataFrame(
        [
            {
                "population": "all_256x256_pixels_by_setup",
                "n": int(pixel_by_setup["pixel_n"].sum()),
                "distance_min": float(pixel_by_setup["pixel_distance_min"].min()),
                "distance_p25": float(pixel_by_setup["pixel_distance_p25"].mean()),
                "distance_median": float(pixel_by_setup["pixel_distance_median"].mean()),
                "distance_p75": float(pixel_by_setup["pixel_distance_p75"].mean()),
                "distance_max": float(pixel_by_setup["pixel_distance_max"].max()),
                "z_min": float("nan"),
                "z_p25": float("nan"),
                "z_median": float(pixel_by_setup["pixel_z_median"].mean()),
                "z_p75": float("nan"),
                "z_max": float("nan"),
                "frac_below_synth_mean": float(pixel_by_setup["pixel_frac_below_synth_mean"].mean()),
                "frac_below_minus2": float(pixel_by_setup["pixel_frac_below_minus2"].mean()),
            }
        ]
    )
    distribution = pd.concat([point_summary, pixel_summary], ignore_index=True)

    pred = query_predictions_with_features()
    pred["synthetic_z"] = (np.log(pred["distance_m"] + 1e-6) - NORM["d_log_mean"]) / NORM["d_log_std"]
    bins = [-np.inf, -4, -3, -2, -1, 0, np.inf]
    labels = ["<-4", "-4..-3", "-3..-2", "-2..-1", "-1..0", ">0"]
    pred["z_bin"] = pd.cut(pred["synthetic_z"], bins=bins, labels=labels, right=False)
    bin_rows = []
    for z_bin, group in pred.groupby("z_bin", observed=False):
        if len(group) == 0:
            continue
        row = {"synthetic_z_bin": str(z_bin), "query_n": int(len(group)), "distance_median": float(group["distance_m"].median())}
        for model_name in MODEL_NAMES:
            row[f"{model_name}_rmse"] = rmse(group["y_true"], group[model_name])
        row["r101_minus_sparse"] = row["r101_feature_ridge_rmse"] - row["sparse_rf_rmse"]
        bin_rows.append(row)
    by_bin = pd.DataFrame(bin_rows)

    distribution.to_csv(OUT_DIR / "check6_distance_normalization_distribution.csv", index=False)
    pixel_by_setup.to_csv(OUT_DIR / "check6_distance_normalization_pixels_by_setup.csv", index=False)
    by_bin.to_csv(OUT_DIR / "check6_distance_bin_rmse.csv", index=False)

    synth_mean_m = float(np.exp(NORM["d_log_mean"]))
    body = f"""
**Goal.** Quantify whether the real geometry is out-of-scale for the synthetic distance normalization, and whether model errors vary across that normalized distance range.

**How.** I recomputed the R101 distance-channel normalization used by `build_input_tensor`: `z = (ln(distance_m + 1e-6) - {NORM['d_log_mean']:.3f}) / {NORM['d_log_std']:.3f}`. The synthetic distance mean corresponds to `exp({NORM['d_log_mean']:.3f}) = {synth_mean_m:.2f} m`. I summarized both observed measurement points and all 256x256 pixels generated for each real setup, then recomputed cached `sparse_rf` and `r101_feature_ridge` RMSE by normalized-distance bin.

**Results.**

Distribution relative to synthetic normalization:

{md_table(distribution)}

Query RMSE by synthetic-normalized distance bin:

{md_table(by_bin)}

The real geometry is indeed far left of the synthetic distance distribution. Observed points have median normalized distance around {float(point_summary['z_median'].iloc[0]):.2f}, and every observed point is below the synthetic mean. The full image grids are also mostly below the synthetic mean; averaging over setups, about {100 * float(pixel_summary['frac_below_synth_mean'].iloc[0]):.1f}% of pixels are below it. This reinforces that R101 is being run in an extrapolative low-distance regime, not merely with missing reflectance/transmittance. The bin table does not show a simple monotonic failure pattern, but it does show that the entire evaluated benchmark lives in a narrow, low-z slice of the synthetic normalization.
"""
    append_section("6. Real Distance Scale Versus Synthetic Normalization", body)


def append_conclusion() -> None:
    c1 = pd.read_csv(OUT_DIR / "check1_near_ap_rmse.csv")
    c3 = pd.read_csv(OUT_DIR / "check3_random_support_aggregate.csv")
    c4 = pd.read_csv(OUT_DIR / "check4_noise_floor_vs_models.csv")
    c5 = pd.read_csv(OUT_DIR / "check5_pointlocal_summary.csv")
    c6 = pd.read_csv(OUT_DIR / "check6_distance_normalization_distribution.csv")
    near_1m = c1[c1["exclude_distance"] == ">1m"].iloc[0]
    random_mean = c3[c3["stat"] == "mean"].iloc[0]
    repeat = c4[c4["comparison"] == "repeat_ap_pairs"].iloc[0]
    pointlocal = c5[c5["model"] == "r101_pointlocal_ridge"].iloc[0]
    observed_dist = c6[c6["population"] == "observed_points"].iloc[0]
    pixel_dist = c6[c6["population"] == "all_256x256_pixels_by_setup"].iloc[0]

    body = f"""
**Detailed analysis.** The new checks make the problem look less like "R101 is failing to learn radio propagation" and more like "this benchmark gives R101 very little stable residual signal after sparse real measurements, and several labels violate the AP geometry the model is asked to use."

The strongest new evidence is the AP-label audit. Setup 16 and setup 2 both have H5 AP labels that are adjacent to, but not at, the strongest/raw-nearest point. Setup 16 says AP point 30, yet point 29 is -26 dBm and raw-nearest while point 30 is -44 dBm. Setup 2 says AP point 3, yet point 2 is -19 dBm and raw-nearest while point 3 is -33 dBm. A deterministic geometry/pathloss model is structurally punished by that kind of one-point AP ambiguity. Sparse RF is partly protected because it uses real support RSSI statistics and can learn dataset quirks directly from real labels.

Removing near-AP query points clarifies the effect. With all queries, R101 ridge only beats sparse RF by about 0.07 dB. After excluding points within 1 m of the H5 AP label, the R101 edge grows to {abs(float(near_1m['r101_minus_sparse'])):.3f} dB. That is not huge, but it is the direction expected if the worst AP-adjacent labels are suppressing R101's geometry signal.

The random-support experiment weakens the case that R101 is robustly useful under the current sampling-assisted protocol. Across 10 random support draws, mean RMSE was {float(random_mean['sparse_rf']):.3f} for sparse RF and {float(random_mean['r101_feature_ridge']):.3f} for R101 ridge, with mean R101-minus-sparse of {float(random_mean['r101_minus_sparse']):.3f}. So the fixed linspace support result was not a broad win; it was a small win under one support geometry. The sparse support points dominate the conditional problem, and changing them changes both models enough to swamp the old 0.07 dB headline.

The repeat-AP noise floor stays central. The pooled repeat-AP disagreement is {float(repeat['rmse']):.3f} dB, larger than the model RMSEs reported for the sampling-assisted query benchmark. Since the repeat-pair task is not identical to model evaluation, I would not treat 5.19 dB as a hard lower bound. But it is clear evidence that setup-level measurement variation is on the same scale as the residuals we are trying to explain.

The point-local feature check is the most negative result for R101 as a local field model here. Replacing global pooled encoder features with local bottleneck plus decoder features produced {float(pointlocal['pooled_rmse']):.3f} dB RMSE. That says the useful part of the frozen R101 transfer is not a clean local descriptor at the receiver point. In this real-data setup, the global pooled features are more plausibly acting as map/setup calibration variables, while `r101_pl_pred` supplies a smooth pathloss-like trend that overlaps heavily with distance and sparse interpolation.

The distance-normalization check adds another structural reason for weak transfer. Observed real distances have median {float(observed_dist['distance_median']):.3f} m and median synthetic-normalized z-score {float(observed_dist['z_median']):.3f}; all observed points are below the synthetic mean distance. Even all generated real image pixels are mostly below the synthetic mean, with mean fraction below the synthetic mean of {100 * float(pixel_dist['frac_below_synth_mean']):.1f}%. So the frozen R101 is operating in a short-range regime that is far from the synthetic normalization center, in addition to missing material channels and AP-label uncertainty.

My updated opinion: R101 is useful, but not in the way we hoped. It contains real propagation signal, yet the current dataset/protocol makes that signal marginal after five real support measurements. The best existing R101 model is worth keeping as an auxiliary feature head, but I would not present it as materially superior to sparse RF on this benchmark. Before investing in more R101 ablations, I would fix or model AP placement uncertainty, evaluate support-draw distributions instead of a single linspace support set, rescale or fine-tune the distance channel for this real geometry, and separate "global setup calibration" features from truly local R101 field features. If those changes still leave R101 at parity, then the practical value of this pretrained checkpoint for this real dataset is limited to small regularizing corrections rather than strong reconstruction gains.
"""
    append_section("Conclusion", body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("check", choices=["1", "2", "3", "4", "5", "6", "conclusion"])
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.check == "1":
        check_near_ap()
    elif args.check == "2":
        check_setup_mapping()
    elif args.check == "3":
        check_random_supports()
    elif args.check == "4":
        check_repeat_ap_noise_floor()
    elif args.check == "5":
        check_pointlocal_features()
    elif args.check == "6":
        check_distance_normalization()
    elif args.check == "conclusion":
        append_conclusion()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
