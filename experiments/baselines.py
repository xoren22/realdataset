#!/usr/bin/env python
"""Leakage-safe RSSI baselines for Zenodo 15791300.

Run from repo root:
    /home/kpetrosyan/miniconda3/envs/c/bin/python experiments/baselines.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.paths import REPO_ROOT, RSSI_CSV, WIFI_H5, require_data

SEED = 0
CELL_M = 0.5
EXPECTED_GRID_SAMPLES = 20 * 53
EXPECTED_RAW_SAMPLES = 1027
FIRST_POINT_MISSING_SETUPS = {5, 7, 14}

RUNS = REPO_ROOT / "runs"
MODELS = REPO_ROOT / "models"
FIGS = REPO_ROOT / "figs"
REPORT = REPO_ROOT / "report.md"

NUM_COLS = ["ap_x_m", "ap_y_m", "mx_m", "my_m", "log_distance_m", "same_room"]
CAT_COLS = ["room_meas", "room_ap"]
FEATURE_COLS = NUM_COLS + CAT_COLS


def point_to_room(point: int) -> str:
    if point <= 30:
        return "office"
    if point <= 45:
        return "corridor"
    return "elevator"


def parse_rssi(value: str) -> float:
    match = re.search(r"-?\d+", str(value))
    if not match:
        raise ValueError(f"could not parse RSSI value: {value!r}")
    return float(match.group())


def grid_coordinates(indices: np.ndarray, setup_idx: int, point: int) -> tuple[float, float]:
    loc = np.argwhere(indices[setup_idx] == point)
    if loc.shape != (1, 2):
        raise ValueError(f"setup {setup_idx + 1}: point {point} has {len(loc)} grid locations")
    row, col = loc[0]
    return float(col * CELL_M), float(row * CELL_M)


def observed_points_for_setup(setup: int, n_raw: int) -> list[int]:
    first = 2 if setup in FIRST_POINT_MISSING_SETUPS else 1
    points = list(range(first, first + n_raw))
    if points[-1] > 53:
        raise ValueError(f"setup {setup}: {n_raw} raw rows cannot map to points <= 53")
    return points


def load_samples() -> pd.DataFrame:
    raw = pd.read_csv(RSSI_CSV)
    raw["setup"] = raw["Setup"].astype(int)
    raw["rssi"] = raw["Strength"].map(parse_rssi)

    with h5py.File(WIFI_H5, "r") as f:
        grid = f["data"][:]
        indices = f["indices"][:]
        ap_locations = f["ap_locations"][:].astype(int)

    rows = []
    for setup in sorted(raw["setup"].unique()):
        setup_raw = raw.loc[raw["setup"] == setup].reset_index(drop=True)
        ap_point = int(ap_locations[setup - 1])
        ap_x_m, ap_y_m = grid_coordinates(indices, setup - 1, ap_point)
        for source_row, point in enumerate(observed_points_for_setup(setup, len(setup_raw))):
            mx_m, my_m = grid_coordinates(indices, setup - 1, point)
            room_meas = point_to_room(point)
            room_ap = point_to_room(ap_point)
            rows.append(
                {
                    "sample_id": len(rows),
                    "setup": setup,
                    "source_row": int(setup_raw.index[source_row]),
                    "point": point,
                    "ap_point": ap_point,
                    "ap_x_m": ap_x_m,
                    "ap_y_m": ap_y_m,
                    "mx_m": mx_m,
                    "my_m": my_m,
                    "room_meas": room_meas,
                    "room_ap": room_ap,
                    "same_room": int(room_meas == room_ap),
                    "rssi": float(setup_raw.loc[source_row, "rssi"]),
                }
            )

    df = pd.DataFrame(rows)
    df["distance_m"] = np.hypot(df["mx_m"] - df["ap_x_m"], df["my_m"] - df["ap_y_m"])
    df["log_distance_m"] = np.log10(df["distance_m"].clip(lower=CELL_M / 2))
    df["excluded_imputed_h5_targets"] = int(np.count_nonzero(grid) - len(df))
    return df


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ]
    )


def build_models() -> dict[str, object]:
    return {
        "mean": DummyRegressor(strategy="mean"),
        "logdistance": Pipeline(
            [
                ("features", ColumnTransformer([("log_distance", "passthrough", ["log_distance_m"])])),
                ("model", LinearRegression()),
            ]
        ),
        "ridge": Pipeline([("features", preprocessor()), ("model", Ridge(alpha=1.0))]),
        "knn": Pipeline([("features", preprocessor()), ("model", KNeighborsRegressor(n_neighbors=5))]),
        "rf": Pipeline(
            [
                ("features", preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=2,
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hgb": Pipeline(
            [
                ("features", preprocessor()),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=120,
                        learning_rate=0.05,
                        l2_regularization=0.1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def run_cv(models: dict[str, object], df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = df[FEATURE_COLS]
    y = df["rssi"].to_numpy()
    groups = df["ap_point"].to_numpy()
    splitter = LeaveOneGroupOut()

    folds = []
    metrics = []
    predictions = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(x, y, groups)):
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        if train_groups & val_groups:
            raise AssertionError(f"fold {fold}: AP group leaked into train and validation")
        holdout_ap = int(groups[val_idx][0])
        folds.append(
            {
                "fold": fold,
                "holdout_ap_point": holdout_ap,
                "train_n": int(len(train_idx)),
                "val_n": int(len(val_idx)),
                "holdout_setups": ",".join(map(str, sorted(df.iloc[val_idx]["setup"].unique()))),
            }
        )
        for name, template in models.items():
            model = clone(template)
            model.fit(x.iloc[train_idx], y[train_idx])
            for split, idx in (("train", train_idx), ("val", val_idx)):
                y_pred = model.predict(x.iloc[idx])
                metrics.append(
                    {
                        "model": name,
                        "fold": fold,
                        "holdout_ap_point": holdout_ap,
                        "split": split,
                        "n": int(len(idx)),
                        "rmse": rmse(y[idx], y_pred),
                        "mae": float(np.mean(np.abs(y_pred - y[idx]))),
                        "r2": float(r2_score(y[idx], y_pred)) if np.var(y[idx]) > 0 else np.nan,
                    }
                )
            y_val_pred = model.predict(x.iloc[val_idx])
            predictions.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "fold": fold,
                        "holdout_ap_point": holdout_ap,
                        "sample_id": df.iloc[val_idx]["sample_id"].to_numpy(),
                        "setup": df.iloc[val_idx]["setup"].to_numpy(),
                        "point": df.iloc[val_idx]["point"].to_numpy(),
                        "y_true": y[val_idx],
                        "y_pred": y_val_pred,
                        "error": y_val_pred - y[val_idx],
                    }
                )
            )

    return pd.DataFrame(folds), pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


def summarize(metrics: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, preds in predictions.groupby("model"):
        val = metrics[(metrics["model"] == model) & (metrics["split"] == "val")]
        train = metrics[(metrics["model"] == model) & (metrics["split"] == "train")]
        rows.append(
            {
                "model": model,
                "pooled_val_rmse": rmse(preds["y_true"].to_numpy(), preds["y_pred"].to_numpy()),
                "pooled_val_mae": float(np.mean(np.abs(preds["error"]))),
                "mean_fold_val_rmse": float(val["rmse"].mean()),
                "std_fold_val_rmse": float(val["rmse"].std()),
                "mean_fold_val_r2": float(val["r2"].mean()),
                "mean_train_rmse": float(train["rmse"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("pooled_val_rmse").reset_index(drop=True)


def validate_outputs(df: pd.DataFrame, folds: pd.DataFrame, metrics: pd.DataFrame, predictions: pd.DataFrame, summary: pd.DataFrame) -> None:
    if len(df) != EXPECTED_RAW_SAMPLES:
        raise AssertionError(f"expected {EXPECTED_RAW_SAMPLES} observed samples, got {len(df)}")
    if int(df["excluded_imputed_h5_targets"].iloc[0]) != EXPECTED_GRID_SAMPLES - EXPECTED_RAW_SAMPLES:
        raise AssertionError("unexpected imputed-target exclusion count")
    if len(folds) != df["ap_point"].nunique() or len(folds) != 16:
        raise AssertionError(f"expected 16 AP-location folds, got {len(folds)}")

    expected_models = set(summary["model"])
    for model in expected_models:
        pred = predictions[predictions["model"] == model]
        if pred["sample_id"].nunique() != len(df) or len(pred) != len(df):
            raise AssertionError(f"{model}: expected one out-of-fold prediction per sample")
        recomputed = rmse(pred["y_true"].to_numpy(), pred["y_pred"].to_numpy())
        saved = float(summary.loc[summary["model"] == model, "pooled_val_rmse"].iloc[0])
        if abs(recomputed - saved) > 1e-12:
            raise AssertionError(f"{model}: summary RMSE does not match predictions")

    fold_splits = metrics[["model", "fold", "split"]].drop_duplicates()
    if (fold_splits.groupby(["model", "fold"]).size() != 2).any():
        raise AssertionError("each model/fold must have train and val metrics")


ROOM_COLORS = {"office": "C0", "corridor": "C1", "elevator": "C2"}


def plot_target_distribution(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for room, color in ROOM_COLORS.items():
        ax.hist(df.loc[df["room_meas"] == room, "rssi"], bins=28, alpha=0.55, color=color, label=room)
    ax.set_xlabel("RSSI [dBm]")
    ax.set_ylabel("observed samples")
    ax.set_title("Observed RSSI targets by measurement room")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rssi_vs_distance(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for room, color in ROOM_COLORS.items():
        sub = df[df["room_meas"] == room]
        ax.scatter(sub["log_distance_m"], sub["rssi"], s=12, alpha=0.45, color=color, label=room)
    fit = np.polyfit(df["log_distance_m"], df["rssi"], deg=1)
    xs = np.linspace(df["log_distance_m"].min(), df["log_distance_m"].max(), 100)
    ax.plot(xs, fit[0] * xs + fit[1], "k--", lw=1, label=f"fit: {fit[1]:.1f} + {fit[0]:.1f} log10(d)")
    ax.set_xlabel("log10(distance [m])")
    ax.set_ylabel("RSSI [dBm]")
    ax.set_title("Observed RSSI vs AP distance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cv_summary(summary: pd.DataFrame, path: Path) -> None:
    s = summary.sort_values("pooled_val_rmse")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(s["model"], s["pooled_val_rmse"], yerr=s["std_fold_val_rmse"], capsize=4, color="C0", alpha=0.85)
    for i, value in enumerate(s["pooled_val_rmse"]):
        ax.text(i, value + 0.12, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("pooled validation RMSE [dBm]")
    ax.set_title("Leave-one-AP-location-out validation RMSE")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_train_val(summary: pd.DataFrame, path: Path) -> None:
    s = summary.sort_values("pooled_val_rmse")
    x = np.arange(len(s))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, s["mean_train_rmse"], width, label="train", color="C2", alpha=0.82)
    ax.bar(x + width / 2, s["pooled_val_rmse"], width, label="validation", color="C0", alpha=0.82)
    ax.set_xticks(x, s["model"])
    ax.set_ylabel("RMSE [dBm]")
    ax.set_title("Train vs out-of-fold validation error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_predictions(predictions: pd.DataFrame, best_model: str, path: Path) -> None:
    pred = predictions[predictions["model"] == best_model]
    lo = float(min(pred["y_true"].min(), pred["y_pred"].min()) - 2)
    hi = float(max(pred["y_true"].max(), pred["y_pred"].max()) + 2)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.scatter(pred["y_true"], pred["y_pred"], s=13, alpha=0.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("true RSSI [dBm]")
    ax.set_ylabel("predicted RSSI [dBm]")
    ax.set_title(f"{best_model}: pooled out-of-fold predictions")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residual_map(df: pd.DataFrame, predictions: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    best = str(summary.iloc[0]["model"])
    val_metrics = predictions[predictions["model"] == best].copy()
    fold_rmse = pd.Series(
        {
            fold: rmse(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
            for fold, group in val_metrics.groupby("fold")
        }
    )
    fold = int(fold_rmse.sort_values().index[len(fold_rmse) // 2])
    pred = val_metrics[val_metrics["fold"] == fold].sort_values("sample_id")
    sub = df.set_index("sample_id").loc[pred["sample_id"]].copy()
    sub["y_pred"] = pred["y_pred"].to_numpy()
    sub["residual"] = sub["rssi"] - sub["y_pred"]

    true_grid = np.full((28, 30), np.nan)
    pred_grid = np.full((28, 30), np.nan)
    resid_grid = np.full((28, 30), np.nan)
    for row in sub.itertuples(index=False):
        grid_row = int(row.my_m / CELL_M)
        grid_col = int(row.mx_m / CELL_M)
        true_grid[grid_row, grid_col] = row.rssi
        pred_grid[grid_row, grid_col] = row.y_pred
        resid_grid[grid_row, grid_col] = row.residual

    vabs = float(np.nanmax(np.abs(resid_grid)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        (true_grid, "true RSSI", "viridis", {}),
        (pred_grid, "predicted RSSI", "viridis", {}),
        (resid_grid, "residual (true - pred)", "RdBu", {"vmin": -vabs, "vmax": vabs}),
    ]
    for ax, (grid, title, cmap, kwargs) in zip(axes, panels):
        im = ax.imshow(np.flipud(grid), cmap=cmap, **kwargs)
        fig.colorbar(im, ax=ax, shrink=0.72, label="dBm")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    holdout = int(pred["holdout_ap_point"].iloc[0])
    fig.suptitle(f"{best}: held-out AP point {holdout}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def md_table(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    for row in df.itertuples(index=False):
        cells = [f"{value:.3f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(df: pd.DataFrame, folds: pd.DataFrame, summary: pd.DataFrame, figure_paths: dict[str, str]) -> None:
    best = summary.iloc[0]
    table = md_table(summary)
    repeated_aps = (
        df[["ap_point", "setup"]]
        .drop_duplicates()
        .groupby("ap_point")["setup"]
        .apply(lambda s: ",".join(map(str, sorted(s))))
    )
    repeated_ap_text = "; ".join(
        f"AP point {ap} in setups {setups}"
        for ap, setups in repeated_aps[repeated_aps.str.contains(",")].to_dict().items()
    )
    text = f"""# Leakage-safe RSSI baselines

Auto-generated by `experiments/baselines.py`.

- **target**: observed RSSI in dBm from `data/RSSI_raw_data.csv`
- **samples**: {len(df):,} observed measurements
- **excluded targets**: {int(df['excluded_imputed_h5_targets'].iloc[0])} H5-only imputed grid values
- **features**: `{', '.join(FEATURE_COLS)}`
- **CV**: leave-one-AP-location-out, k = {len(folds)}
- **metric**: RMSE in dBm, computed as `sqrt(mean((prediction - target)^2))`
- **seed**: {SEED}

## Dataset and leakage controls

The raw CSV has {len(df):,} observed RSSI readings. The processed H5 grid has
{EXPECTED_GRID_SAMPLES:,} nonzero grid labels because the upstream toolbox fills
missing measurements so every setup has 53 grid points. This benchmark excludes
those {int(df['excluded_imputed_h5_targets'].iloc[0])} filled targets from
training and validation RMSE.

The validation split groups by AP point, not setup. This matters because
{repeated_ap_text} appear in multiple setups; holding out by AP point prevents
the same AP location from appearing in both train and validation.

All scaling and one-hot encoding live inside sklearn pipelines and are fit
inside each fold on training rows only.

![Observed RSSI target distribution]({figure_paths['target']})

![Observed RSSI vs log-distance]({figure_paths['distance']})

## Cross-validation results

The ranking metric is **pooled validation RMSE** across all out-of-fold
predictions. Fold mean/std RMSE show the AP-location-to-AP-location variation.
Mean train RMSE is included to expose overfitting.

{table}

Best model: **{best['model']}** with pooled validation RMSE
**{best['pooled_val_rmse']:.2f} dBm**.

![Validation RMSE by model]({figure_paths['cv']})

![Train vs validation RMSE]({figure_paths['train_val']})

![Pooled out-of-fold predictions for the best model]({figure_paths['pred']})

![Residual map on a representative held-out AP fold]({figure_paths['resid']})

## Artifacts

- `runs/baselines_samples.csv` - observed sample table used by the experiment
- `runs/baselines_folds.csv` - AP-location fold definitions
- `runs/baselines_cv.csv` - train/validation metrics per model and fold
- `runs/baselines_oof_predictions.csv` - out-of-fold predictions
- `runs/baselines_summary.csv` - summary table above
- `runs/baselines_config.json` - reproducibility metadata
- `models/<model>.joblib` - each model refit on all observed samples
- `figs/baseline_*.png` - report figures
"""
    REPORT.write_text(text)


def write_config(df: pd.DataFrame, folds: pd.DataFrame, models: dict[str, object]) -> None:
    config = {
        "seed": SEED,
        "target": "observed raw RSSI dBm",
        "sample_count": int(len(df)),
        "excluded_imputed_h5_targets": int(df["excluded_imputed_h5_targets"].iloc[0]),
        "cv": "LeaveOneGroupOut grouped by AP point",
        "fold_count": int(len(folds)),
        "feature_columns": FEATURE_COLS,
        "models": list(models),
    }
    (RUNS / "baselines_config.json").write_text(json.dumps(config, indent=2) + "\n")


def save_refit_models(models: dict[str, object], df: pd.DataFrame) -> None:
    x = df[FEATURE_COLS]
    y = df["rssi"].to_numpy()
    for name, template in models.items():
        model = clone(template)
        model.fit(x, y)
        joblib.dump(model, MODELS / f"{name}.joblib")


def main() -> int:
    require_data()
    np.random.seed(SEED)
    for path in (RUNS, MODELS, FIGS):
        path.mkdir(parents=True, exist_ok=True)

    df = load_samples()
    models = build_models()
    folds, metrics, predictions = run_cv(models, df)
    summary = summarize(metrics, predictions)
    validate_outputs(df, folds, metrics, predictions, summary)

    df.to_csv(RUNS / "baselines_samples.csv", index=False)
    folds.to_csv(RUNS / "baselines_folds.csv", index=False)
    metrics.to_csv(RUNS / "baselines_cv.csv", index=False)
    predictions.to_csv(RUNS / "baselines_oof_predictions.csv", index=False)
    summary.to_csv(RUNS / "baselines_summary.csv", index=False)
    write_config(df, folds, models)
    save_refit_models(models, df)

    figures = {
        "target": "figs/baseline_target_distribution.png",
        "distance": "figs/baseline_rssi_vs_distance.png",
        "cv": "figs/baseline_cv_rmse.png",
        "train_val": "figs/baseline_train_vs_val_rmse.png",
        "pred": "figs/baseline_pred_vs_true.png",
        "resid": "figs/baseline_residual_map.png",
    }
    plot_target_distribution(df, REPO_ROOT / figures["target"])
    plot_rssi_vs_distance(df, REPO_ROOT / figures["distance"])
    plot_cv_summary(summary, REPO_ROOT / figures["cv"])
    plot_train_val(summary, REPO_ROOT / figures["train_val"])
    plot_predictions(predictions, str(summary.iloc[0]["model"]), REPO_ROOT / figures["pred"])
    plot_residual_map(df, predictions, summary, REPO_ROOT / figures["resid"])
    write_report(df, folds, summary, figures)

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nwrote {REPORT.relative_to(REPO_ROOT)}, runs/, models/, figs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
