#!/usr/bin/env python
"""Compare sparse/proximity/r101_pl feature groups with tuned ridge."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from experiments.baselines import load_samples, rmse
from experiments.comparing_feature_sets_ridge import CACHE_DIR, SEEDS, SPARSE_12_COLS, seed_cache_path, seed_support_path
from experiments.sampling_assisted_r101 import RIDGE_ALPHA_SWEEP, add_sparse_features, folds
from experiments.tune_ridge_feature_set_alpha import ridge_predict, summarize
from src.paths import REPO_ROOT, require_data


OUT_DIR = REPO_ROOT / "runs" / "proximity_feature_groups_ridge"
REPORT_PATH = REPO_ROOT / "proximity_feature_groups_ridge_report.md"
ALPHAS = tuple(RIDGE_ALPHA_SWEEP)
PROXIMITY_COLS = ["ap_rx_same_point", "ap_rx_lt_1m", "ap_rx_lt_2m"]
GROUPS = ("sparse12", "prox3", "r101_pl")


def model_name(groups: tuple[str, ...]) -> str:
    return "_plus_".join(groups) + "_ridge"


FEATURE_GROUPS = {
    model_name(combo): combo
    for r in range(1, len(GROUPS) + 1)
    for combo in itertools.combinations(GROUPS, r)
}


def add_proximity_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    distance = out["distance_m"].to_numpy(dtype=np.float32)
    out["ap_rx_same_point"] = (distance <= 1e-6).astype(np.float32)
    out["ap_rx_lt_1m"] = (distance < 1.0).astype(np.float32)
    out["ap_rx_lt_2m"] = (distance < 2.0).astype(np.float32)
    return out


def load_seed_context(seed: int, samples: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    supports = pd.read_csv(seed_support_path(seed))
    sparse_data = add_sparse_features(samples, supports).sort_values("sample_id").reset_index(drop=True)
    sparse_data = add_proximity_features(sparse_data)
    cache = np.load(seed_cache_path(seed))
    return sparse_data, cache["pl_by_fold_sample"]


def matrices_for_fold(
    sparse_data: pd.DataFrame,
    pl_by_fold_sample: np.ndarray,
    fold: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    fold_data = sparse_data.copy()
    fold_data["r101_pl_pred"] = pl_by_fold_sample[fold, fold_data["sample_id"].to_numpy(dtype=int)]

    sparse12_x = fold_data[SPARSE_12_COLS].to_numpy(dtype=np.float32)
    prox3_x = fold_data[PROXIMITY_COLS].to_numpy(dtype=np.float32)
    pl_x = fold_data[["r101_pl_pred"]].to_numpy(dtype=np.float32)

    group_matrix = {
        "sparse12": sparse12_x,
        "prox3": prox3_x,
        "r101_pl": pl_x,
    }
    matrices = {
        name: np.concatenate([group_matrix[group] for group in groups], axis=1)
        for name, groups in FEATURE_GROUPS.items()
    }
    return fold_data, matrices, fold_data["rssi"].to_numpy(dtype=np.float32)


def table_from_best(best: pd.DataFrame) -> str:
    cols = ["model", "feature_count", "alpha", "train_RMSE", "val_RMSE", "val_mae_mean"]
    display = best.copy()
    display["train_RMSE"] = display.apply(
        lambda r: f"{r.train_rmse_mean:.3f} +/- {r.train_rmse_std_across_draws:.3f}", axis=1
    )
    display["val_RMSE"] = display.apply(
        lambda r: f"{r.val_rmse_mean:.3f} +/- {r.val_rmse_std_across_draws:.3f}", axis=1
    )
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in display[cols].itertuples(index=False):
        cells = []
        for value in row:
            cells.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def per_draw_table(by_seed_alpha: pd.DataFrame, best: pd.DataFrame) -> str:
    selected = by_seed_alpha.merge(best[["model", "alpha"]], on=["model", "alpha"])
    pivot = selected.pivot(index="support_seed", columns="model", values="val_pooled_rmse")
    ordered = list(best["model"])
    lines = [
        "| support seed | " + " | ".join(ordered) + " |",
        "|---:|" + "|".join("---:" for _ in ordered) + "|",
    ]
    for seed, row in pivot[ordered].iterrows():
        lines.append(f"| {int(seed)} | " + " | ".join(f"{row[col]:.3f}" for col in ordered) + " |")
    return "\n".join(lines)


def paired_delta(
    by_seed_alpha: pd.DataFrame,
    best: pd.DataFrame,
    baseline: str,
    candidate: str,
) -> tuple[float, float]:
    selected = by_seed_alpha.merge(best[["model", "alpha"]], on=["model", "alpha"])
    pivot = selected.pivot(index="support_seed", columns="model", values="val_pooled_rmse")
    delta = pivot[baseline] - pivot[candidate]
    return float(delta.mean()), float(delta.std(ddof=1))


def write_report(best: pd.DataFrame, by_seed_alpha: pd.DataFrame, overall_alpha: pd.DataFrame) -> None:
    sparse_to_sparse_prox = paired_delta(
        by_seed_alpha, best, "sparse12_ridge", "sparse12_plus_prox3_ridge"
    )
    sparse_to_sparse_pl = paired_delta(
        by_seed_alpha, best, "sparse12_ridge", "sparse12_plus_r101_pl_ridge"
    )
    sparse_pl_to_sparse_prox_pl = paired_delta(
        by_seed_alpha,
        best,
        "sparse12_plus_r101_pl_ridge",
        "sparse12_plus_prox3_plus_r101_pl_ridge",
    )

    text = f"""# Proximity Feature Groups Ridge Report

Generated from `experiments/proximity_feature_groups_ridge.py`.

This experiment adds three AP-to-receiver proximity indicators to the earlier
random-support ridge comparison and tests every non-empty combination of three
feature groups:

- `sparse12`: the 12 non-constant sparse engineered features.
- `prox3`: `ap_rx_same_point`, `ap_rx_lt_1m`, `ap_rx_lt_2m`.
- `r101_pl`: the cached decoded `r101_pl_pred` scalar.

I interpreted the requested `<0 (same point)` feature as exact same-point
distance, implemented as `distance_m <= 1e-6`. The other two thresholds are
strict `< 1.0 m` and `< 2.0 m`.

The protocol is unchanged from the ridge feature comparison:

- 10 random support draws, seeds `1000..1009`.
- 5 uniformly sampled support points per setup.
- 16 leave-one-AP-location-out folds.
- Query points only are scored.
- `StandardScaler` + ridge regression.
- Alpha sweep: `{", ".join(str(a) for a in ALPHAS)}`.
- Cached R101 outputs from `{CACHE_DIR.relative_to(REPO_ROOT)}`; no R101 inference.

## Best Alpha Results

Mean and standard deviation are across the 10 random support draws.

{table_from_best(best)}

## Per-Draw Validation RMSE

Each column uses that model's selected best alpha above.

{per_draw_table(by_seed_alpha, best)}

## Main Deltas

Positive values mean the added feature group improved validation RMSE.

| comparison | paired RMSE gain mean | paired RMSE gain std |
|---|---:|---:|
| `sparse12` -> `sparse12 + prox3` | {sparse_to_sparse_prox[0]:.3f} | {sparse_to_sparse_prox[1]:.3f} |
| `sparse12` -> `sparse12 + r101_pl` | {sparse_to_sparse_pl[0]:.3f} | {sparse_to_sparse_pl[1]:.3f} |
| `sparse12 + r101_pl` -> `sparse12 + prox3 + r101_pl` | {sparse_pl_to_sparse_prox_pl[0]:.3f} | {sparse_pl_to_sparse_prox_pl[1]:.3f} |

## Interpretation

The proximity indicators help a little when added to sparse engineered
features, but they do not change the main conclusion. `r101_pl_pred` remains
the dominant useful extra scalar beyond the sparse feature block.

The best model is the full `sparse12 + prox3 + r101_pl` group, but its gain over
`sparse12 + r101_pl` is small. The new proximity features mostly encode a local
near-AP exception already partly represented by `distance_m` and
`log_distance_m`, so their contribution is incremental rather than a new source
of signal.

The combinations without `sparse12` are much worse, including `prox3 + r101_pl`.
So the sparse/interpolation feature block remains necessary; neither the new
near-distance flags nor r101's decoded prediction replaces it.

## Artifacts

- `experiments/proximity_feature_groups_ridge.py`
- `runs/proximity_feature_groups_ridge/proximity_feature_groups_ridge_fold_metrics.csv`
- `runs/proximity_feature_groups_ridge/proximity_feature_groups_ridge_by_seed_alpha_summary.csv`
- `runs/proximity_feature_groups_ridge/proximity_feature_groups_ridge_overall_by_alpha.csv`
- `runs/proximity_feature_groups_ridge/proximity_feature_groups_ridge_best_by_model.csv`
- `runs/proximity_feature_groups_ridge/manifest.json`
"""
    REPORT_PATH.write_text(text)


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
        sparse_data, pl_by_fold_sample = load_seed_context(seed, samples)
        for fold_row in fold_defs:
            fold = int(fold_row.fold)
            fold_data, matrices, y = matrices_for_fold(sparse_data, pl_by_fold_sample, fold)
            is_val = fold_data["ap_point"].to_numpy() == int(fold_row.holdout_ap_point)
            is_query = fold_data["is_support"].to_numpy() == 0
            train_mask = (~is_val) & is_query
            val_mask = is_val & is_query
            for name, x in matrices.items():
                for alpha in ALPHAS:
                    train_pred, val_pred = ridge_predict(x[train_mask], y[train_mask], x[val_mask], alpha)
                    train_err = train_pred - y[train_mask]
                    val_err = val_pred - y[val_mask]
                    metric_rows.append(
                        {
                            "support_seed": int(seed),
                            "fold": fold,
                            "holdout_ap_point": int(fold_row.holdout_ap_point),
                            "model": name,
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
        print(f"finished proximity ridge alpha sweep seed {seed}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    by_seed_alpha, overall_alpha = summarize(metrics)
    best = (
        overall_alpha.sort_values(["model", "val_rmse_mean"])
        .groupby("model", as_index=False)
        .first()
        .sort_values("val_rmse_mean")
        .reset_index(drop=True)
    )

    metrics.to_csv(OUT_DIR / "proximity_feature_groups_ridge_fold_metrics.csv", index=False)
    by_seed_alpha.to_csv(OUT_DIR / "proximity_feature_groups_ridge_by_seed_alpha_summary.csv", index=False)
    overall_alpha.to_csv(OUT_DIR / "proximity_feature_groups_ridge_overall_by_alpha.csv", index=False)
    best.to_csv(OUT_DIR / "proximity_feature_groups_ridge_best_by_model.csv", index=False)
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "experiment": "proximity_feature_groups_ridge",
                "support_seeds": list(SEEDS),
                "alphas": list(ALPHAS),
                "feature_groups": FEATURE_GROUPS,
                "proximity_features": {
                    "ap_rx_same_point": "distance_m <= 1e-6",
                    "ap_rx_lt_1m": "distance_m < 1.0",
                    "ap_rx_lt_2m": "distance_m < 2.0",
                },
                "cache_dir": str(CACHE_DIR),
            },
            indent=2,
        )
    )
    write_report(best, by_seed_alpha, overall_alpha)

    print("\nBest alpha by model:")
    print(best.to_string(index=False))
    print(f"\nWrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
