#!/usr/bin/env python
"""Alpha sweep for ridge feature-set comparisons using cached R101 features."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.baselines import load_samples, rmse
from experiments.comparing_feature_sets_ridge import CACHE_DIR, SEEDS, SPARSE_12_COLS, seed_cache_path, seed_support_path
from experiments.sampling_assisted_r101 import RIDGE_ALPHA_SWEEP, add_sparse_features, folds
from src.paths import REPO_ROOT, require_data


OUT_DIR = REPO_ROOT / "runs" / "ridge_feature_set_alpha_tuning"
ALPHAS = RIDGE_ALPHA_SWEEP

FEATURE_SETS = {
    "sparse12_ridge": "sparse12",
    "sparse12_plus_r101_pl_ridge": "sparse12_pl",
    "sparse12_plus_encoder2048_ridge": "sparse12_encoder",
    "sparse12_plus_r101_pl_encoder2048_ridge": "sparse12_pl_encoder",
    "r101_pl_encoder2048_ridge": "pl_encoder",
}


def ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "ridge",
                Ridge(
                    alpha=float(alpha),
                    solver="lsqr",
                    tol=1e-6,
                    max_iter=10000,
                ),
            ),
        ]
    )
    model.fit(train_x, train_y)
    return model.predict(train_x), model.predict(val_x)


def matrices_for_fold(seed: int, samples: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    supports = pd.read_csv(seed_support_path(seed))
    sparse_data = add_sparse_features(samples, supports).sort_values("sample_id").reset_index(drop=True)
    cache = np.load(seed_cache_path(seed))
    pl_by_fold_sample = cache["pl_by_fold_sample"]
    encoder_by_fold_setup = cache["encoder_by_fold_setup"]
    setup_ids = cache["setup_ids"].astype(int).tolist()
    setup_to_idx = {setup: idx for idx, setup in enumerate(setup_ids)}

    fold_data = sparse_data.copy()
    fold_data["r101_pl_pred"] = pl_by_fold_sample[fold, fold_data["sample_id"].to_numpy(dtype=int)]
    setup_index = fold_data["setup"].map(setup_to_idx).to_numpy(dtype=int)
    encoder = encoder_by_fold_setup[fold, setup_index, :]
    sparse12_x = fold_data[SPARSE_12_COLS].to_numpy(dtype=np.float32)
    pl_x = fold_data[["r101_pl_pred"]].to_numpy(dtype=np.float32)
    y = fold_data["rssi"].to_numpy(dtype=np.float32)
    matrices = {
        "sparse12": sparse12_x,
        "sparse12_pl": np.concatenate([sparse12_x, pl_x], axis=1),
        "sparse12_encoder": np.concatenate([sparse12_x, encoder], axis=1),
        "sparse12_pl_encoder": np.concatenate([sparse12_x, pl_x, encoder], axis=1),
        "pl_encoder": np.concatenate([pl_x, encoder], axis=1),
    }
    return fold_data, matrices, y, fold_data["ap_point"].to_numpy(), fold_data["is_support"].to_numpy()


def summarize(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return by_seed_alpha, overall


def main() -> int:
    require_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = load_samples()
    missing = [path for seed in SEEDS for path in (seed_cache_path(seed), seed_support_path(seed)) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing cached R101 feature files: {missing[:3]}")

    metric_rows = []
    fold_defs = list(folds(samples).itertuples(index=False))
    for seed in SEEDS:
        for fold_row in fold_defs:
            fold = int(fold_row.fold)
            fold_data, matrices, y, ap_points, is_support = matrices_for_fold(seed, samples, fold)
            is_val = ap_points == int(fold_row.holdout_ap_point)
            is_query = is_support == 0
            train_mask = (~is_val) & is_query
            val_mask = is_val & is_query
            for model_name, matrix_key in FEATURE_SETS.items():
                x = matrices[matrix_key]
                for alpha in ALPHAS:
                    train_pred, val_pred = ridge_predict(x[train_mask], y[train_mask], x[val_mask], alpha)
                    train_err = train_pred - y[train_mask]
                    val_err = val_pred - y[val_mask]
                    metric_rows.append(
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
        print(f"finished alpha sweep seed {seed}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    by_seed_alpha, overall_alpha = summarize(metrics)
    best = (
        overall_alpha.sort_values(["model", "val_rmse_mean"])
        .groupby("model", as_index=False)
        .first()
        .sort_values("val_rmse_mean")
        .reset_index(drop=True)
    )

    metrics.to_csv(OUT_DIR / "ridge_feature_set_alpha_fold_metrics.csv", index=False)
    by_seed_alpha.to_csv(OUT_DIR / "ridge_feature_set_alpha_by_seed_alpha_summary.csv", index=False)
    overall_alpha.to_csv(OUT_DIR / "ridge_feature_set_alpha_overall_by_alpha.csv", index=False)
    best.to_csv(OUT_DIR / "ridge_feature_set_alpha_best_by_model.csv", index=False)
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "experiment": "ridge_feature_set_alpha_tuning",
                "support_seeds": list(SEEDS),
                "alphas": list(ALPHAS),
                "feature_sets": FEATURE_SETS,
                "cache_dir": str(CACHE_DIR),
                "note": "Alpha selected by lowest mean validation RMSE across the same 10x16 outer CV runs; not nested inner CV.",
            },
            indent=2,
        )
    )

    print("\nBest alpha by model:")
    print(best.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
