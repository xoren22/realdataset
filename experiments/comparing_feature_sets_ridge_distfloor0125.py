#!/usr/bin/env python
"""Ridge 12-vs-13 feature comparison with R101 distance input floored at 0.125 m."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.baselines import load_samples, rmse
from experiments.comparing_feature_sets_ridge import (
    SEEDS,
    SPARSE_12_COLS,
    random_support_table,
)
from experiments.sampling_assisted_r101 import (
    GRID_H,
    GRID_W,
    IMAGE_SIZE,
    CELL_M,
    CHANNELS,
    MLSP_SPARSE_MEAN,
    MLSP_SPARSE_STD,
    NORM,
    OUT_NORM_DB,
    RIDGE_ALPHA,
    RIDGE_ALPHA_SWEEP,
    SUPPORT_COUNT,
    WALL_THICKNESS_PX,
    add_sparse_features,
    c_from_support,
    folds,
    frequency_fourier,
    load_encoder_unet_model,
    point_grid_locations,
    point_to_pixel,
    resize_nearest,
    rssi_to_pathloss_proxy,
    setup_frequencies,
    wall_floor_plan_image,
)
from src.paths import REPO_ROOT, require_data


MIN_DISTANCE_M = 0.125
OUT_DIR = REPO_ROOT / "runs" / "comparing_feature_sets_ridge_distfloor0125"
CACHE_DIR = OUT_DIR / "r101_distfloor0125_feature_cache"

FEATURE_SETS = {
    "sparse12_ridge": "sparse12",
    "sparse12_plus_r101_pl_ridge": "sparse12_pl",
}


def seed_cache_path(seed: int) -> Path:
    return CACHE_DIR / f"seed_{seed}_r101_pl_distfloor0125.npz"


def seed_support_path(seed: int) -> Path:
    return CACHE_DIR / f"seed_{seed}_support_table.csv"


def source_support_path(seed: int) -> Path:
    return (
        REPO_ROOT
        / "runs"
        / "comparing_feature_sets_ridge"
        / "r101_global_feature_cache"
        / f"seed_{seed}_support_table.csv"
    )


def support_table(samples: pd.DataFrame, seed: int) -> pd.DataFrame:
    path = seed_support_path(seed)
    if path.exists():
        return pd.read_csv(path)
    old_path = source_support_path(seed)
    if old_path.exists():
        return pd.read_csv(old_path)
    return random_support_table(samples, seed)


def build_input_tensor_distfloor(
    setup: int,
    samples: pd.DataFrame,
    setup_support: pd.DataFrame,
    point_locations: dict[int, tuple[int, int]],
    freq_mhz: float,
    wall_thickness_px: int,
    c_dbm: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    setup_all = samples[samples["setup"] == setup]
    floor_hi = wall_floor_plan_image(wall_thickness_px)
    mask_hi = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    sparse = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for row in setup_support.itertuples(index=False):
        grid_row, grid_col = point_locations[int(row.point)]
        sparse[grid_row, grid_col] = rssi_to_pathloss_proxy(np.array([row.rssi]), c_dbm)[0]
    sparse_hi = resize_nearest(sparse)
    mask_hi[sparse_hi != 0] = 0.0

    ap = setup_all.iloc[0]
    ys = ((np.arange(IMAGE_SIZE, dtype=np.float32) + 0.5) * GRID_H / IMAGE_SIZE) * CELL_M
    xs = ((np.arange(IMAGE_SIZE, dtype=np.float32) + 0.5) * GRID_W / IMAGE_SIZE) * CELL_M
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    distance = np.hypot(xx - float(ap.ap_x_m), yy - float(ap.ap_y_m)).astype(np.float32)
    distance = np.maximum(distance, MIN_DISTANCE_M).astype(np.float32)
    f1, f2, f3, f4 = frequency_fourier(freq_mhz)

    tensor = np.zeros((len(CHANNELS), IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    tensor[2] = (np.log(distance + 1e-6) - NORM["d_log_mean"]) / NORM["d_log_std"]
    tensor[4] = f1
    tensor[5] = f2
    tensor[6] = f3
    tensor[7] = f4
    tensor[8] = mask_hi
    tensor[9] = floor_hi
    sparse_norm = sparse_hi.copy()
    sparse_mask = sparse_norm != 0
    sparse_norm[sparse_mask] = (sparse_norm[sparse_mask] - MLSP_SPARSE_MEAN) / MLSP_SPARSE_STD
    tensor[10] = sparse_norm
    return tensor, c_dbm, sparse


def forward_r101_batch(model: torch.nn.Module, tensors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    x = torch.from_numpy(np.stack(tensors, axis=0))
    with torch.inference_mode():
        features = model.unet.encoder(x)
        decoded = model.unet.decoder(features)
        pred = model.unet.segmentation_head(decoded).squeeze(1)
        pooled = F.adaptive_avg_pool2d(features[-1], output_size=1).flatten(1).float()
    return (pred.float().numpy() * OUT_NORM_DB), pooled.numpy()


def build_seed_cache(seed: int, samples: pd.DataFrame, model: torch.nn.Module | None) -> None:
    cache_path = seed_cache_path(seed)
    support_path = seed_support_path(seed)
    if cache_path.exists() and support_path.exists():
        return
    if model is None:
        raise ValueError("model is required when cache is missing")

    supports = support_table(samples, seed)
    support_samples = samples.merge(supports, on=["sample_id", "setup", "point"])
    point_locations = point_grid_locations()
    freqs = setup_frequencies()
    fold_defs = folds(samples)
    setup_ids = np.array(sorted(samples["setup"].unique()), dtype=np.int16)
    sample_ids = samples.sort_values("sample_id")["sample_id"].to_numpy(dtype=np.int32)
    sample_setups = samples.sort_values("sample_id")["setup"].to_numpy(dtype=np.int16)

    pl_by_fold_sample = np.zeros((len(fold_defs), len(sample_ids)), dtype=np.float32)
    encoder_by_fold_setup = np.zeros((len(fold_defs), len(setup_ids), 2048), dtype=np.float32)
    c_by_fold = np.zeros(len(fold_defs), dtype=np.float32)

    for fold_row in fold_defs.itertuples(index=False):
        train_support = support_samples[
            (support_samples["ap_point"] != fold_row.holdout_ap_point)
            & (support_samples["is_support"] == 1)
        ]
        c_fold = c_from_support(train_support)
        c_by_fold[int(fold_row.fold)] = c_fold

        tensors = []
        for setup in setup_ids:
            setup_support = support_samples[
                (support_samples["setup"] == int(setup)) & (support_samples["is_support"] == 1)
            ]
            tensor, _, _sparse_grid = build_input_tensor_distfloor(
                setup=int(setup),
                samples=samples,
                setup_support=setup_support,
                point_locations=point_locations,
                freq_mhz=freqs[int(setup)],
                wall_thickness_px=WALL_THICKNESS_PX,
                c_dbm=c_fold,
            )
            tensors.append(tensor)

        pred_pl_batch, pooled_batch = forward_r101_batch(model, tensors)
        encoder_by_fold_setup[int(fold_row.fold)] = pooled_batch.astype(np.float32)

        for setup_idx, setup in enumerate(setup_ids):
            setup_rows = samples[samples["setup"] == int(setup)]
            for row in setup_rows.itertuples(index=False):
                grid_row, grid_col = point_locations[int(row.point)]
                pix_y, pix_x = point_to_pixel(grid_row, grid_col)
                pl_by_fold_sample[int(fold_row.fold), int(row.sample_id)] = pred_pl_batch[
                    setup_idx, pix_y, pix_x
                ]

        print(f"cached distfloor seed {seed}, fold {int(fold_row.fold)}", flush=True)

    supports.to_csv(support_path, index=False)
    np.savez_compressed(
        cache_path,
        support_seed=np.array([seed], dtype=np.int32),
        sample_ids=sample_ids,
        sample_setups=sample_setups,
        setup_ids=setup_ids,
        pl_by_fold_sample=pl_by_fold_sample,
        encoder_by_fold_setup=encoder_by_fold_setup,
        c_by_fold=c_by_fold,
        min_distance_m=np.array([MIN_DISTANCE_M], dtype=np.float32),
    )


def matrices_for_fold(
    seed: int, samples: pd.DataFrame, fold: int
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    supports = pd.read_csv(seed_support_path(seed))
    sparse_data = add_sparse_features(samples, supports).sort_values("sample_id").reset_index(drop=True)
    cache = np.load(seed_cache_path(seed))
    pl_by_fold_sample = cache["pl_by_fold_sample"]

    fold_data = sparse_data.copy()
    fold_data["r101_pl_pred"] = pl_by_fold_sample[fold, fold_data["sample_id"].to_numpy(dtype=int)]
    sparse12_x = fold_data[SPARSE_12_COLS].to_numpy(dtype=np.float32)
    pl_x = fold_data[["r101_pl_pred"]].to_numpy(dtype=np.float32)
    y = fold_data["rssi"].to_numpy(dtype=np.float32)
    matrices = {
        "sparse12": sparse12_x,
        "sparse12_pl": np.concatenate([sparse12_x, pl_x], axis=1),
    }
    return fold_data, matrices, y, fold_data["ap_point"].to_numpy(), fold_data["is_support"].to_numpy()


def ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    alpha: float,
    solver: str = "lsqr",
) -> tuple[np.ndarray, np.ndarray]:
    kwargs = {"alpha": float(alpha), "solver": solver}
    if solver == "lsqr":
        kwargs.update({"tol": 1e-6, "max_iter": 10000})
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(**kwargs))])
    model.fit(train_x, train_y)
    return model.predict(train_x), model.predict(val_x)


def summarize_fixed(metrics: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows = []
    for (seed, model), pred_group in predictions.groupby(["support_seed", "model"]):
        metric_group = metrics[(metrics["support_seed"] == seed) & (metrics["model"] == model)]
        seed_rows.append(
            {
                "support_seed": int(seed),
                "model": model,
                "feature_count": int(pred_group["feature_count"].iloc[0]),
                "query_n": int(len(pred_group)),
                "train_rmse_mean_over_16_folds": float(metric_group["train_rmse"].mean()),
                "train_rmse_std_over_16_folds": float(metric_group["train_rmse"].std()),
                "val_pooled_rmse": rmse(pred_group["y_true"], pred_group["y_pred"]),
                "val_pooled_mae": float(np.mean(np.abs(pred_group["error"]))),
                "val_mean_fold_rmse": float(metric_group["val_rmse"].mean()),
                "val_std_fold_rmse": float(metric_group["val_rmse"].std()),
            }
        )
    by_seed = pd.DataFrame(seed_rows).sort_values(["model", "support_seed"]).reset_index(drop=True)

    overall_rows = []
    for model, group in by_seed.groupby("model"):
        overall_rows.append(
            {
                "model": model,
                "feature_count": int(group["feature_count"].iloc[0]),
                "support_draws": int(len(group)),
                "train_rmse_mean": float(group["train_rmse_mean_over_16_folds"].mean()),
                "train_rmse_std_across_draws": float(group["train_rmse_mean_over_16_folds"].std()),
                "val_rmse_mean": float(group["val_pooled_rmse"].mean()),
                "val_rmse_std_across_draws": float(group["val_pooled_rmse"].std()),
                "val_mae_mean": float(group["val_pooled_mae"].mean()),
                "val_mean_fold_rmse_mean": float(group["val_mean_fold_rmse"].mean()),
                "val_mean_fold_rmse_std_across_draws": float(group["val_mean_fold_rmse"].std()),
            }
        )
    overall = pd.DataFrame(overall_rows).sort_values("val_rmse_mean").reset_index(drop=True)
    return by_seed, overall


def summarize_alpha(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_alpha_rows = []
    for (seed, model, alpha), group in metrics.groupby(["support_seed", "model", "alpha"]):
        seed_alpha_rows.append(
            {
                "support_seed": int(seed),
                "model": str(model),
                "alpha": float(alpha),
                "feature_count": int(group["feature_count"].iloc[0]),
                "train_rmse_mean_over_16_folds": float(group["train_rmse"].mean()),
                "train_rmse_std_over_16_folds": float(group["train_rmse"].std()),
                "val_mean_fold_rmse": float(group["val_rmse"].mean()),
                "val_std_fold_rmse": float(group["val_rmse"].std()),
                "val_pooled_rmse": float(np.sqrt(np.average(group["val_rmse"] ** 2, weights=group["val_n"]))),
                "val_mae_mean_over_16_folds": float(group["val_mae"].mean()),
            }
        )
    by_seed_alpha = pd.DataFrame(seed_alpha_rows)

    overall_rows = []
    for (model, alpha), group in by_seed_alpha.groupby(["model", "alpha"]):
        overall_rows.append(
            {
                "model": str(model),
                "alpha": float(alpha),
                "feature_count": int(group["feature_count"].iloc[0]),
                "support_draws": int(len(group)),
                "train_rmse_mean": float(group["train_rmse_mean_over_16_folds"].mean()),
                "train_rmse_std_across_draws": float(group["train_rmse_mean_over_16_folds"].std()),
                "val_rmse_mean": float(group["val_pooled_rmse"].mean()),
                "val_rmse_std_across_draws": float(group["val_pooled_rmse"].std()),
                "val_mean_fold_rmse_mean": float(group["val_mean_fold_rmse"].mean()),
                "val_mean_fold_rmse_std_across_draws": float(group["val_mean_fold_rmse"].std()),
                "val_mae_mean": float(group["val_mae_mean_over_16_folds"].mean()),
            }
        )
    overall = pd.DataFrame(overall_rows).sort_values(["model", "alpha"]).reset_index(drop=True)
    best = (
        overall.sort_values(["model", "val_rmse_mean"])
        .groupby("model", as_index=False)
        .first()
        .sort_values("val_rmse_mean")
        .reset_index(drop=True)
    )
    return by_seed_alpha, overall, best


def main() -> int:
    require_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    samples = load_samples()

    model = None
    for seed in SEEDS:
        if not (seed_cache_path(seed).exists() and seed_support_path(seed).exists()):
            if model is None:
                model = load_encoder_unet_model()
            build_seed_cache(seed, samples, model)
            print(f"built R101 distfloor cache for seed {seed}", flush=True)
        else:
            print(f"loaded R101 distfloor cache for seed {seed}", flush=True)

    fixed_metric_rows = []
    fixed_pred_rows = []
    alpha_metric_rows = []
    fold_defs = list(folds(samples).itertuples(index=False))
    for seed in SEEDS:
        for fold_row in fold_defs:
            fold = int(fold_row.fold)
            fold_data, matrices, y, ap_points, is_support = matrices_for_fold(seed, samples, fold)
            is_val = ap_points == int(fold_row.holdout_ap_point)
            is_query = is_support == 0
            train_mask = (~is_val) & is_query
            val_mask = is_val & is_query
            val_rows = fold_data.loc[val_mask, ["sample_id", "setup", "point", "rssi"]].copy()

            for model_name, matrix_key in FEATURE_SETS.items():
                x = matrices[matrix_key]
                train_pred, val_pred = ridge_predict(
                    x[train_mask], y[train_mask], x[val_mask], RIDGE_ALPHA, solver="svd"
                )
                train_err = train_pred - y[train_mask]
                val_err = val_pred - y[val_mask]
                fixed_metric_rows.append(
                    {
                        "support_seed": int(seed),
                        "fold": fold,
                        "holdout_ap_point": int(fold_row.holdout_ap_point),
                        "model": model_name,
                        "feature_count": int(x.shape[1]),
                        "train_n": int(train_mask.sum()),
                        "val_n": int(val_mask.sum()),
                        "train_rmse": rmse(y[train_mask], train_pred),
                        "train_mae": float(np.mean(np.abs(train_err))),
                        "train_bias": float(np.mean(train_err)),
                        "val_rmse": rmse(y[val_mask], val_pred),
                        "val_mae": float(np.mean(np.abs(val_err))),
                        "val_bias": float(np.mean(val_err)),
                    }
                )
                for row, y_pred, err in zip(val_rows.itertuples(index=False), val_pred, val_err):
                    fixed_pred_rows.append(
                        {
                            "support_seed": int(seed),
                            "fold": fold,
                            "holdout_ap_point": int(fold_row.holdout_ap_point),
                            "model": model_name,
                            "feature_count": int(x.shape[1]),
                            "sample_id": int(row.sample_id),
                            "setup": int(row.setup),
                            "point": int(row.point),
                            "y_true": float(row.rssi),
                            "y_pred": float(y_pred),
                            "error": float(err),
                        }
                    )

                for alpha in RIDGE_ALPHA_SWEEP:
                    train_pred, val_pred = ridge_predict(x[train_mask], y[train_mask], x[val_mask], alpha)
                    train_err = train_pred - y[train_mask]
                    val_err = val_pred - y[val_mask]
                    alpha_metric_rows.append(
                        {
                            "support_seed": int(seed),
                            "fold": fold,
                            "holdout_ap_point": int(fold_row.holdout_ap_point),
                            "model": model_name,
                            "feature_count": int(x.shape[1]),
                            "alpha": float(alpha),
                            "train_n": int(train_mask.sum()),
                            "val_n": int(val_mask.sum()),
                            "train_rmse": rmse(y[train_mask], train_pred),
                            "train_mae": float(np.mean(np.abs(train_err))),
                            "train_bias": float(np.mean(train_err)),
                            "val_rmse": rmse(y[val_mask], val_pred),
                            "val_mae": float(np.mean(np.abs(val_err))),
                            "val_bias": float(np.mean(val_err)),
                        }
                    )
        print(f"evaluated distfloor ridge feature sets for seed {seed}", flush=True)

    fixed_metrics = pd.DataFrame(fixed_metric_rows)
    fixed_predictions = pd.DataFrame(fixed_pred_rows)
    fixed_by_seed, fixed_overall = summarize_fixed(fixed_metrics, fixed_predictions)
    alpha_metrics = pd.DataFrame(alpha_metric_rows)
    alpha_by_seed, alpha_overall, alpha_best = summarize_alpha(alpha_metrics)

    fixed_metrics.to_csv(OUT_DIR / "ridge_distfloor0125_fixed_alpha_fold_metrics.csv", index=False)
    fixed_predictions.to_csv(OUT_DIR / "ridge_distfloor0125_fixed_alpha_predictions.csv", index=False)
    fixed_by_seed.to_csv(OUT_DIR / "ridge_distfloor0125_fixed_alpha_by_seed_summary.csv", index=False)
    fixed_overall.to_csv(OUT_DIR / "ridge_distfloor0125_fixed_alpha_overall_summary.csv", index=False)
    alpha_metrics.to_csv(OUT_DIR / "ridge_distfloor0125_alpha_fold_metrics.csv", index=False)
    alpha_by_seed.to_csv(OUT_DIR / "ridge_distfloor0125_alpha_by_seed_summary.csv", index=False)
    alpha_overall.to_csv(OUT_DIR / "ridge_distfloor0125_alpha_overall_by_alpha.csv", index=False)
    alpha_best.to_csv(OUT_DIR / "ridge_distfloor0125_alpha_best_by_model.csv", index=False)
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "experiment": "comparing_feature_sets_ridge_distfloor0125",
                "support_seeds": list(SEEDS),
                "support_sampling": (
                    "same support tables as runs/comparing_feature_sets_ridge when present; "
                    "otherwise uniform without replacement, 5 points per setup per seed"
                ),
                "min_distance_m": MIN_DISTANCE_M,
                "ridge_fixed_alpha": RIDGE_ALPHA,
                "ridge_alpha_sweep": list(RIDGE_ALPHA_SWEEP),
                "sparse_12_features": SPARSE_12_COLS,
                "feature_sets": {
                    "sparse12_ridge": "12 sparse engineered features, support_count dropped because constant",
                    "sparse12_plus_r101_pl_ridge": "12 sparse features + r101_pl_pred from R101 input with distance floored at 0.125 m",
                },
                "cache_dir": str(CACHE_DIR),
            },
            indent=2,
        )
    )

    print("\nFixed alpha summary:")
    print(fixed_overall.to_string(index=False))
    print("\nBest tuned alpha summary:")
    print(alpha_best.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
