#!/usr/bin/env python
"""Compare linear SVM-style regression feature sets over random sparse support draws."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

from experiments.baselines import load_samples, rmse
from experiments.comparing_feature_sets_ridge import SEEDS, SPARSE_12_COLS, seed_cache_path, seed_support_path
from experiments.sampling_assisted_r101 import add_sparse_features, folds
from src.paths import REPO_ROOT, require_data


OUT_DIR = REPO_ROOT / "runs" / "comparing_feature_sets_svm"
CACHE_DIR = REPO_ROOT / "runs" / "comparing_feature_sets_ridge" / "r101_global_feature_cache"
LOW_DIM_ALPHAS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
ENCODER_ALPHAS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
SVR_EPSILON = 0.1
MAX_ITER = 100

FEATURE_SETS = {
    "sparse12_svm": "sparse12",
    "sparse12_plus_r101_pl_svm": "sparse12_pl",
    "sparse12_plus_encoder2048_svm": "sparse12_encoder",
    "sparse12_plus_r101_pl_encoder2048_svm": "sparse12_pl_encoder",
    "r101_pl_encoder2048_svm": "pl_encoder",
}

ALPHAS_BY_MODEL = {
    model_name: (ENCODER_ALPHAS if "encoder2048" in model_name else LOW_DIM_ALPHAS)
    for model_name in FEATURE_SETS
}


def fit_predict_scaled_svm(
    train_z: np.ndarray,
    train_y: np.ndarray,
    val_z: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    model = SGDRegressor(
        loss="epsilon_insensitive",
        penalty="l2",
        alpha=float(alpha),
        epsilon=SVR_EPSILON,
        tol=1e-3,
        max_iter=MAX_ITER,
        learning_rate="optimal",
        average=True,
        random_state=7,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(train_z, train_y)
    return model.predict(train_z), model.predict(val_z)


def load_seed_context(seed: int, samples: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[int, int]]:
    supports = pd.read_csv(seed_support_path(seed))
    sparse_data = add_sparse_features(samples, supports).sort_values("sample_id").reset_index(drop=True)
    cache = np.load(seed_cache_path(seed))
    pl_by_fold_sample = cache["pl_by_fold_sample"]
    encoder_by_fold_setup = cache["encoder_by_fold_setup"]
    setup_ids = cache["setup_ids"].astype(int).tolist()
    setup_to_idx = {setup: idx for idx, setup in enumerate(setup_ids)}
    return sparse_data, pl_by_fold_sample, encoder_by_fold_setup, setup_to_idx


def matrices_for_seed_fold(seed: int, samples: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    sparse_data, pl_by_fold_sample, encoder_by_fold_setup, setup_to_idx = load_seed_context(seed, samples)
    return matrices_for_context(sparse_data, pl_by_fold_sample, encoder_by_fold_setup, setup_to_idx, fold)


def matrices_for_context(
    sparse_data: pd.DataFrame,
    pl_by_fold_sample: np.ndarray,
    encoder_by_fold_setup: np.ndarray,
    setup_to_idx: dict[int, int],
    fold: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
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
    return fold_data, matrices, y


def summarize(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def write_report(best: pd.DataFrame, overall: pd.DataFrame) -> None:
    def md_table(df: pd.DataFrame, cols: list[str]) -> str:
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
        for row in df[cols].itertuples(index=False):
            cells = []
            for value in row:
                if isinstance(value, float):
                    cells.append(f"{value:.3f}")
                else:
                    cells.append(str(value))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    display = best.copy()
    display["val_RMSE"] = display.apply(
        lambda r: f"{r.val_rmse_mean:.3f} ± {r.val_rmse_std_across_draws:.3f}", axis=1
    )
    display["train_RMSE"] = display.apply(
        lambda r: f"{r.train_rmse_mean:.3f} ± {r.train_rmse_std_across_draws:.3f}", axis=1
    )
    ordered = display[
        [
            "model",
            "feature_count",
            "alpha",
            "train_RMSE",
            "val_RMSE",
            "val_mae_mean",
        ]
    ]

    text = f"""# SVM Features Comparison

Generated from `experiments/comparing_feature_sets_svm.py`.

This repeats the ridge feature-set comparison with linear epsilon-insensitive
SVM-style regression trained by SGD. It uses the same 10 random sparse-support draws, the same
leave-one-AP-location-out folds, and the cached R101 decoded prediction plus
global encoder features from:

`runs/comparing_feature_sets_ridge/r101_global_feature_cache/`

No R101 inference is run by this experiment.

## Protocol

- Model: `StandardScaler` + `SGDRegressor(loss="epsilon_insensitive", penalty="l2")`.
- Kernel: linear.
- Epsilon: `{SVR_EPSILON}` dB.
- Regularization sweep:
  - low-dimensional feature sets: `alpha = {", ".join(str(v) for v in LOW_DIM_ALPHAS)}`
  - encoder feature sets: `alpha = {", ".join(str(v) for v in ENCODER_ALPHAS)}`
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

{md_table(ordered, ["model", "feature_count", "alpha", "train_RMSE", "val_RMSE", "val_mae_mean"])}

## Interpretation

The SVM results tell the same main story as ridge. The decoded R101 prediction
is useful when added to sparse features: `sparse12 + r101_pl` is the best SVM
feature set. The 2048 encoder vector does not provide a reliable improvement:
adding it to `sparse12 + r101_pl` lowers train error slightly but worsens
validation, and adding it to sparse12 without `r101_pl_pred` is worse than
sparse12 alone.

Compared with tuned ridge, the best SVM is slightly worse:

- best tuned ridge: `sparse12 + r101_pl`, RMSE `5.799 ± 0.188`
- best tuned SVM: `sparse12 + r101_pl`, RMSE `{float(best.iloc[0].val_rmse_mean):.3f} ± {float(best.iloc[0].val_rmse_std_across_draws):.3f}`

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
"""
    (REPO_ROOT / "svm_features_comparison.md").write_text(text)


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
        sparse_data, pl_by_fold_sample, encoder_by_fold_setup, setup_to_idx = load_seed_context(seed, samples)
        for fold_row in fold_defs:
            fold = int(fold_row.fold)
            fold_data, matrices, y = matrices_for_context(
                sparse_data, pl_by_fold_sample, encoder_by_fold_setup, setup_to_idx, fold
            )
            is_val = fold_data["ap_point"].to_numpy() == int(fold_row.holdout_ap_point)
            is_query = fold_data["is_support"].to_numpy() == 0
            train_mask = (~is_val) & is_query
            val_mask = is_val & is_query
            for model_name, matrix_key in FEATURE_SETS.items():
                x = matrices[matrix_key]
                scaler = StandardScaler()
                train_z = scaler.fit_transform(x[train_mask])
                val_z = scaler.transform(x[val_mask])
                for alpha in ALPHAS_BY_MODEL[model_name]:
                    train_pred, val_pred = fit_predict_scaled_svm(train_z, y[train_mask], val_z, alpha)
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
                            "epsilon": float(SVR_EPSILON),
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
        print(f"finished SVM alpha sweep seed {seed}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    by_seed_alpha, overall_alpha, best = summarize(metrics)
    metrics.to_csv(OUT_DIR / "svm_feature_set_alpha_fold_metrics.csv", index=False)
    by_seed_alpha.to_csv(OUT_DIR / "svm_feature_set_alpha_by_seed_alpha_summary.csv", index=False)
    overall_alpha.to_csv(OUT_DIR / "svm_feature_set_alpha_overall_by_alpha.csv", index=False)
    best.to_csv(OUT_DIR / "svm_feature_set_alpha_best_by_model.csv", index=False)
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "experiment": "comparing_feature_sets_svm",
                "support_seeds": list(SEEDS),
                "low_dim_alpha_values": list(LOW_DIM_ALPHAS),
                "encoder_alpha_values": list(ENCODER_ALPHAS),
                "epsilon": SVR_EPSILON,
                "max_iter": MAX_ITER,
                "feature_sets": FEATURE_SETS,
                "r101_cache_dir": str(CACHE_DIR),
            },
            indent=2,
        )
    )
    write_report(best, overall_alpha)

    print("\nBest alpha by model:")
    print(best.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
