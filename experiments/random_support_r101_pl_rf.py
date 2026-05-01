#!/usr/bin/env python
"""Cache random-support r101_pl_pred features and evaluate r101_pl_rf vs sparse_rf."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone

from experiments.baselines import load_samples, rmse
from experiments.sampling_assisted_r101 import (
    RIDGE_ALPHA,
    SPARSE_COLS,
    SUPPORT_COUNT,
    WALL_THICKNESS_PX,
    add_sparse_features,
    build_input_tensor,
    c_from_support,
    fit_ridge,
    folds,
    load_encoder_unet_model,
    point_grid_locations,
    point_to_pixel,
    r101_tree_model,
    setup_frequencies,
    support_table,
    train_models,
)
from src.paths import REPO_ROOT, require_data


SEEDS = tuple(range(1000, 1010))
OUT_DIR = REPO_ROOT / "runs" / "random_support_r101_pl_rf"
CACHE_DIR = OUT_DIR / "r101_pl_pred_cache"


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


def cache_path(seed: int) -> Path:
    return CACHE_DIR / f"seed_{seed}_fold_features.csv.gz"


def extract_r101_pl_for_fold(
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

    x = torch.from_numpy(np.stack(tensors, axis=0))
    with torch.inference_mode():
        features = model.unet.encoder(x)
        decoded = model.unet.decoder(features)
        pred_pl = model.unet.segmentation_head(decoded).squeeze(1).float().numpy() * 160.0

    rows = []
    for batch_idx, setup in enumerate(setup_ids):
        setup_rows = samples[samples["setup"] == setup]
        sparse_grid = sparse_grids[setup]
        for row in setup_rows.itertuples(index=False):
            grid_row, grid_col = point_locations[int(row.point)]
            pix_y, pix_x = point_to_pixel(grid_row, grid_col)
            rows.append(
                {
                    "fold": int(fold),
                    "holdout_ap_point": int(holdout_ap_point),
                    "sample_id": int(row.sample_id),
                    "setup": int(setup),
                    "point": int(row.point),
                    "r101_pl_pred": float(pred_pl[batch_idx, pix_y, pix_x]),
                    "c_fold_dbm": float(c_dbm),
                    "sparse_pl_at_point": float(sparse_grid[grid_row, grid_col]),
                }
            )
    return pd.DataFrame(rows)


def build_seed_cache(seed: int, samples: pd.DataFrame, model: torch.nn.Module) -> pd.DataFrame:
    path = cache_path(seed)
    if path.exists():
        return pd.read_csv(path)

    supports = random_support_table(samples, seed)
    sparse_data = add_sparse_features(samples, supports)
    support_samples = samples.merge(supports, on=["sample_id", "setup", "point"])

    fold_frames = []
    for fold_row in folds(samples).itertuples(index=False):
        train_support = support_samples[
            (support_samples["ap_point"] != fold_row.holdout_ap_point) & (support_samples["is_support"] == 1)
        ]
        c_fold = c_from_support(train_support)
        r101_pl = extract_r101_pl_for_fold(
            model=model,
            samples=samples,
            support_samples=support_samples,
            c_dbm=c_fold,
            fold=int(fold_row.fold),
            holdout_ap_point=int(fold_row.holdout_ap_point),
        )
        data = sparse_data.merge(r101_pl, on=["sample_id", "setup", "point"], how="left")
        data["support_seed"] = int(seed)
        fold_frames.append(data)

    out = pd.concat(fold_frames, ignore_index=True)
    out.to_csv(path, index=False)
    return out


def evaluate_seed(seed: int, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_rows = []
    fold_rows = []
    for fold_row in folds(data.drop_duplicates("sample_id")).itertuples(index=False):
        fold_data = data[data["fold"] == fold_row.fold].copy()
        is_val = fold_data["ap_point"] == fold_row.holdout_ap_point
        train = fold_data[(~is_val) & (fold_data["is_support"] == 0)].copy()
        val = fold_data[is_val & (fold_data["is_support"] == 0)].copy()

        r101_cols = SPARSE_COLS + ["r101_pl_pred"]

        train_const = float((train["rssi"] + train["r101_pl_pred"]).mean())
        const_train_pred = train_const - val["r101_pl_pred"].to_numpy()

        const_support_series = pd.Series(index=val.index, dtype=float)
        for setup, setup_val in val.groupby("setup", sort=False):
            support = fold_data[(fold_data["setup"] == setup) & (fold_data["is_support"] == 1)]
            support_const = float((support["rssi"] + support["r101_pl_pred"]).mean())
            const_support_series.loc[setup_val.index] = support_const - setup_val["r101_pl_pred"]
        const_support_pred = const_support_series.loc[val.index].to_numpy()

        sparse_model = clone(train_models()["sparse_rf"])
        sparse_model.fit(train[SPARSE_COLS], train["rssi"])
        sparse_pred = sparse_model.predict(val[SPARSE_COLS])

        r101_model = r101_tree_model()
        r101_model.fit(train[r101_cols].to_numpy(dtype=np.float32), train["rssi"])
        r101_pred = r101_model.predict(val[r101_cols].to_numpy(dtype=np.float32))

        r101_ridge = fit_ridge(train, r101_cols, RIDGE_ALPHA)
        r101_ridge_pred = r101_ridge.predict(val[r101_cols])

        for model_name, pred in [
            ("r101_pl_const_train", const_train_pred),
            ("r101_pl_const_support", const_support_pred),
            ("sparse_rf", sparse_pred),
            ("r101_pl_rf", r101_pred),
            ("r101_pl_ridge", r101_ridge_pred),
        ]:
            errors = pred - val["rssi"].to_numpy()
            for row, y_pred, err in zip(val.itertuples(index=False), pred, errors):
                pred_rows.append(
                    {
                        "support_seed": int(seed),
                        "model": model_name,
                        "fold": int(fold_row.fold),
                        "holdout_ap_point": int(fold_row.holdout_ap_point),
                        "sample_id": int(row.sample_id),
                        "setup": int(row.setup),
                        "point": int(row.point),
                        "y_true": float(row.rssi),
                        "y_pred": float(y_pred),
                        "error": float(err),
                    }
                )
            fold_rows.append(
                {
                    "support_seed": int(seed),
                    "model": model_name,
                    "fold": int(fold_row.fold),
                    "holdout_ap_point": int(fold_row.holdout_ap_point),
                    "query_n": int(len(val)),
                    "rmse": rmse(val["rssi"].to_numpy(), pred),
                    "mae": float(np.mean(np.abs(errors))),
                    "bias_pred_minus_true": float(np.mean(errors)),
                }
            )
    return pd.DataFrame(pred_rows), pd.DataFrame(fold_rows)


def summarize(predictions: pd.DataFrame, fold_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows = []
    for (seed, model), group in predictions.groupby(["support_seed", "model"]):
        folds_for_model = fold_metrics[(fold_metrics["support_seed"] == seed) & (fold_metrics["model"] == model)]
        seed_rows.append(
            {
                "support_seed": int(seed),
                "model": model,
                "query_n": int(len(group)),
                "pooled_rmse": rmse(group["y_true"], group["y_pred"]),
                "pooled_mae": float(np.mean(np.abs(group["error"]))),
                "bias_pred_minus_true": float(group["error"].mean()),
                "mean_fold_rmse": float(folds_for_model["rmse"].mean()),
                "std_fold_rmse": float(folds_for_model["rmse"].std()),
            }
        )
    by_seed = pd.DataFrame(seed_rows).sort_values(["support_seed", "model"]).reset_index(drop=True)

    overall_rows = []
    for model, group in predictions.groupby("model"):
        folds_for_model = fold_metrics[fold_metrics["model"] == model]
        overall_rows.append(
            {
                "model": model,
                "support_draws": int(group["support_seed"].nunique()),
                "total_predictions": int(len(group)),
                "mean_seed_pooled_rmse": float(by_seed[by_seed["model"] == model]["pooled_rmse"].mean()),
                "std_seed_pooled_rmse": float(by_seed[by_seed["model"] == model]["pooled_rmse"].std()),
                "pooled_all_draws_rmse": rmse(group["y_true"], group["y_pred"]),
                "pooled_all_draws_mae": float(np.mean(np.abs(group["error"]))),
                "mean_fold_rmse_over_160_folds": float(folds_for_model["rmse"].mean()),
                "std_fold_rmse_over_160_folds": float(folds_for_model["rmse"].std()),
            }
        )
    overall = pd.DataFrame(overall_rows).sort_values("model").reset_index(drop=True)
    return by_seed, overall


def main() -> int:
    require_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)

    samples = load_samples()
    model = None
    all_predictions = []
    all_fold_metrics = []
    for seed in SEEDS:
        if cache_path(seed).exists():
            data = pd.read_csv(cache_path(seed))
            print(f"loaded cached r101_pl_pred features for seed {seed}", flush=True)
        else:
            if model is None:
                model = load_encoder_unet_model()
            data = build_seed_cache(seed, samples, model)
            print(f"built cached r101_pl_pred features for seed {seed}", flush=True)
        predictions, fold_metrics = evaluate_seed(seed, data)
        all_predictions.append(predictions)
        all_fold_metrics.append(fold_metrics)
        print(f"evaluated seed {seed}", flush=True)

    predictions = pd.concat(all_predictions, ignore_index=True)
    fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)
    by_seed, overall = summarize(predictions, fold_metrics)

    predictions.to_csv(OUT_DIR / "random_support_r101_pl_rf_predictions.csv", index=False)
    fold_metrics.to_csv(OUT_DIR / "random_support_r101_pl_rf_16fold_metrics.csv", index=False)
    by_seed.to_csv(OUT_DIR / "random_support_r101_pl_rf_by_seed_summary.csv", index=False)
    overall.to_csv(OUT_DIR / "random_support_r101_pl_rf_overall_summary.csv", index=False)
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "support_seeds": list(SEEDS),
                "support_sampling": "uniform without replacement, 5 points per setup per seed",
                "feature_sets": {
                    "r101_pl_const_train": ["r101_pl_pred"],
                    "r101_pl_const_support": ["r101_pl_pred"],
                    "sparse_rf": SPARSE_COLS,
                    "r101_pl_rf": SPARSE_COLS + ["r101_pl_pred"],
                    "r101_pl_ridge": SPARSE_COLS + ["r101_pl_pred"],
                },
                "r101_cache_dir": str(CACHE_DIR),
                "r101_cache_files": [str(cache_path(seed)) for seed in SEEDS],
                "models": {
                    "sparse_rf": "train_models()['sparse_rf']",
                    "r101_pl_rf": "r101_tree_model() on raw SPARSE_COLS + r101_pl_pred",
                },
                "ridge_alpha_unused_here": RIDGE_ALPHA,
            },
            indent=2,
        )
    )

    print("\nOverall:")
    print(overall.to_string(index=False))
    print("\nBy seed:")
    print(by_seed.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
