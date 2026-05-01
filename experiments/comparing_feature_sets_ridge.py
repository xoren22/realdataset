#!/usr/bin/env python
"""Compare ridge regression feature sets over random sparse support draws."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.baselines import load_samples, rmse
from experiments.sampling_assisted_r101 import (
    RIDGE_ALPHA,
    SPARSE_COLS,
    SUPPORT_COUNT,
    WALL_THICKNESS_PX,
    add_sparse_features,
    build_input_tensor,
    c_from_support,
    folds,
    forward_r101_batch,
    load_encoder_unet_model,
    point_grid_locations,
    point_to_pixel,
    setup_frequencies,
)
from src.paths import REPO_ROOT, require_data


SEEDS = tuple(range(1000, 1010))
OUT_DIR = REPO_ROOT / "runs" / "comparing_feature_sets_ridge"
CACHE_DIR = OUT_DIR / "r101_global_feature_cache"
SPARSE_12_COLS = [col for col in SPARSE_COLS if col != "support_count"]


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


def seed_cache_path(seed: int) -> Path:
    return CACHE_DIR / f"seed_{seed}_r101_pl_and_encoder.npz"


def seed_support_path(seed: int) -> Path:
    return CACHE_DIR / f"seed_{seed}_support_table.csv"


def build_seed_cache(seed: int, samples: pd.DataFrame, model: torch.nn.Module | None) -> None:
    cache_path = seed_cache_path(seed)
    support_path = seed_support_path(seed)
    if cache_path.exists() and support_path.exists():
        return
    if model is None:
        raise ValueError("model is required when cache is missing")

    supports = random_support_table(samples, seed)
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
            (support_samples["ap_point"] != fold_row.holdout_ap_point) & (support_samples["is_support"] == 1)
        ]
        c_fold = c_from_support(train_support)
        c_by_fold[int(fold_row.fold)] = c_fold

        tensors = []
        for setup in setup_ids:
            setup_support = support_samples[(support_samples["setup"] == int(setup)) & (support_samples["is_support"] == 1)]
            tensor, _, _sparse_grid = build_input_tensor(
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
                pl_by_fold_sample[int(fold_row.fold), int(row.sample_id)] = pred_pl_batch[setup_idx, pix_y, pix_x]

        print(f"cached seed {seed}, fold {int(fold_row.fold)}", flush=True)

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
    )


def fit_predict_ridge(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=RIDGE_ALPHA, solver="svd"))])
    model.fit(train_x, train_y)
    return model.predict(train_x), model.predict(val_x)


def evaluate_seed(seed: int, samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    supports = pd.read_csv(seed_support_path(seed))
    sparse_data = add_sparse_features(samples, supports)
    sparse_data = sparse_data.sort_values("sample_id").reset_index(drop=True)
    cache = np.load(seed_cache_path(seed))
    pl_by_fold_sample = cache["pl_by_fold_sample"]
    encoder_by_fold_setup = cache["encoder_by_fold_setup"]
    setup_ids = cache["setup_ids"].astype(int).tolist()
    setup_to_idx = {setup: idx for idx, setup in enumerate(setup_ids)}

    pred_rows = []
    metric_rows = []
    feature_sets = {
        "sparse12_ridge": "sparse12",
        "sparse12_plus_r101_pl_ridge": "sparse12_pl",
        "r101_pl_encoder2048_ridge": "pl_encoder",
        "sparse12_plus_encoder2048_ridge": "sparse12_encoder",
        "sparse12_plus_r101_pl_encoder2048_ridge": "sparse12_pl_encoder",
    }

    for fold_row in folds(samples).itertuples(index=False):
        fold = int(fold_row.fold)
        fold_data = sparse_data.copy()
        fold_data["r101_pl_pred"] = pl_by_fold_sample[fold, fold_data["sample_id"].to_numpy(dtype=int)]
        setup_index = fold_data["setup"].map(setup_to_idx).to_numpy(dtype=int)
        encoder = encoder_by_fold_setup[fold, setup_index, :]
        sparse12_x = fold_data[SPARSE_12_COLS].to_numpy(dtype=np.float32)
        pl_x = fold_data[["r101_pl_pred"]].to_numpy(dtype=np.float32)
        y = fold_data["rssi"].to_numpy(dtype=np.float32)
        is_val = fold_data["ap_point"].to_numpy() == int(fold_row.holdout_ap_point)
        is_query = fold_data["is_support"].to_numpy() == 0
        train_mask = (~is_val) & is_query
        val_mask = is_val & is_query

        matrices = {
            "sparse12": sparse12_x,
            "sparse12_pl": np.concatenate([sparse12_x, pl_x], axis=1),
            "pl_encoder": np.concatenate([pl_x, encoder], axis=1),
            "sparse12_encoder": np.concatenate([sparse12_x, encoder], axis=1),
            "sparse12_pl_encoder": np.concatenate([sparse12_x, pl_x, encoder], axis=1),
        }

        for model_name, matrix_key in feature_sets.items():
            x = matrices[matrix_key]
            train_pred, val_pred = fit_predict_ridge(x[train_mask], y[train_mask], x[val_mask])
            train_err = train_pred - y[train_mask]
            val_err = val_pred - y[val_mask]
            metric_rows.append(
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

            val_rows = fold_data.loc[val_mask, ["sample_id", "setup", "point", "rssi"]].copy()
            for row, y_pred, err in zip(val_rows.itertuples(index=False), val_pred, val_err):
                pred_rows.append(
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
    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows)


def summarize(metrics: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def main() -> int:
    require_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    samples = load_samples()

    model = None
    for seed in SEEDS:
        if not (seed_cache_path(seed).exists() and seed_support_path(seed).exists()):
            if model is None:
                model = load_encoder_unet_model()
            build_seed_cache(seed, samples, model)
            print(f"built R101 global feature cache for seed {seed}", flush=True)
        else:
            print(f"loaded R101 global feature cache for seed {seed}", flush=True)

    metric_frames = []
    prediction_frames = []
    for seed in SEEDS:
        metrics, predictions = evaluate_seed(seed, samples)
        metric_frames.append(metrics)
        prediction_frames.append(predictions)
        print(f"evaluated ridge feature sets for seed {seed}", flush=True)

    metrics = pd.concat(metric_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    by_seed, overall = summarize(metrics, predictions)

    metrics.to_csv(OUT_DIR / "comparing_feature_sets_ridge_fold_metrics.csv", index=False)
    predictions.to_csv(OUT_DIR / "comparing_feature_sets_ridge_predictions.csv", index=False)
    by_seed.to_csv(OUT_DIR / "comparing_feature_sets_ridge_by_seed_summary.csv", index=False)
    overall.to_csv(OUT_DIR / "comparing_feature_sets_ridge_overall_summary.csv", index=False)
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "experiment": "comparing_feature_sets_ridge",
                "support_seeds": list(SEEDS),
                "support_sampling": "uniform without replacement, 5 points per setup per seed",
                "ridge_alpha": RIDGE_ALPHA,
                "sparse_12_features": SPARSE_12_COLS,
                "feature_sets": {
                    "sparse12_ridge": "12 sparse engineered features, support_count dropped because constant",
                    "sparse12_plus_r101_pl_ridge": "12 sparse features + r101_pl_pred",
                    "r101_pl_encoder2048_ridge": "r101_pl_pred + 2048 global pooled R101 encoder features",
                    "sparse12_plus_encoder2048_ridge": "12 sparse features + 2048 global pooled R101 encoder features",
                    "sparse12_plus_r101_pl_encoder2048_ridge": "12 sparse features + r101_pl_pred + 2048 global pooled R101 encoder features",
                },
                "cache_dir": str(CACHE_DIR),
            },
            indent=2,
        )
    )

    print("\nOverall mean/std across 10 random support draws:")
    print(overall.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
