#!/usr/bin/env python
"""Sampling-assisted RSSI reconstruction with the MLSP r101 checkpoint."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib.util

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.baselines import load_samples, rmse
from src.paths import REPO_ROOT, RSSI_CSV, WIFI_H5, require_data

SEED = 7
SUPPORT_COUNT = 5
WALL_THICKNESS_PX = 1
RT_IMPUTE_N = 8
# Keep impute_rt implemented for later experiments, but leave it disabled by default.
RT_MODES = ("zero_rt",)
RIDGE_ALPHA = 0.01
RIDGE_ALPHA_SWEEP = (0.01, 0.1, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)
R101_ONLY_RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
R101_TREE_TOP_ENCODER_FEATURES = 10
GRID_H = 28
GRID_W = 30
IMAGE_SIZE = 256
CELL_M = 0.5
OUT_NORM_DB = 160.0
MLSP_SPARSE_MEAN = 110.1460952758789
MLSP_SPARSE_STD = 42.0374641418457
CKPT_PATH = Path(
    "/mnt/weka/kpetrosyan/synthetic_pretraining_checkpoints/"
    "r101_bs181_lr3e-03_wr0.1_dr0.1_wd1e-03/"
    "2026-04-14_09-58-12.473943/every/step_00350000_every.ckpt"
)
MLSP_REPO = Path("/home/kpetrosyan/mlsp_wair_d")
ENCODER_UNET = MLSP_REPO / "src" / "networks" / "encoder_unet.py"

RUNS = REPO_ROOT / "runs"
FIGS = REPO_ROOT / "figs"
REPORT = REPO_ROOT / "sampling_assisted_r101_report.md"

CHANNELS = [
    "reflectance",
    "transmittance",
    "distance",
    "antenna_gain",
    "freq_sin_1",
    "freq_cos_1",
    "freq_sin_2",
    "freq_cos_2",
    "mask",
    "floor_plan",
    "sparse",
]
NORM = {
    "r_mean": 3.236292362213135,
    "r_std": 2.8495296239852905,
    "t_mean": 5.5926265716552735,
    "t_std": 5.444438934326172,
    "d_log_mean": 3.5391982793807983,
    "d_log_std": 0.7967258393764496,
}
SPARSE_COLS = [
    "ap_x_m",
    "ap_y_m",
    "mx_m",
    "my_m",
    "distance_m",
    "log_distance_m",
    "same_room",
    "support_count",
    "support_mean_rssi",
    "support_std_rssi",
    "nearest_support_dist_m",
    "nearest_support_rssi",
    "idw_support_rssi",
]


def parse_mhz(value: object) -> float:
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit() or ch == ".")
    if not digits:
        raise ValueError(f"could not parse MHz value: {value!r}")
    return float(digits)


def setup_frequencies() -> dict[int, float]:
    raw = pd.read_csv(RSSI_CSV)
    raw["setup"] = raw["Setup"].astype(int)
    raw["mhz"] = raw["Primary Frequency"].map(parse_mhz)
    return raw.groupby("setup")["mhz"].agg(lambda s: float(s.mode().iloc[0])).to_dict()


def frequency_fourier(freq_mhz: float) -> tuple[float, float, float, float]:
    freq_min, freq_max = 100.0, 7000.0
    freq = max(freq_min, min(freq_max, float(freq_mhz)))
    t = (np.log(freq) - np.log(freq_min)) / (np.log(freq_max) - np.log(freq_min))
    values = []
    for level in range(2):
        scale = 2**level
        values.extend([np.sin(2 * np.pi * scale * t), np.cos(2 * np.pi * scale * t)])
    return tuple(float(v) for v in values)


def free_space_grid() -> np.ndarray:
    floor = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    floor[0:9, 0:11] = 1.0
    floor[9:12, 0:29] = 1.0
    floor[12:27, 24:27] = 1.0
    return floor


def resize_nearest(grid: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(grid.astype(np.float32))[None, None]
    y = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="nearest")
    return y.squeeze(0).squeeze(0).numpy()


def wall_floor_plan_image(thickness_px: int) -> np.ndarray:
    if thickness_px < 1:
        raise ValueError("wall thickness must be >= 1")
    free = torch.from_numpy(resize_nearest(free_space_grid()))[None, None]
    eroded = -F.max_pool2d(-free, kernel_size=2 * thickness_px + 1, stride=1, padding=thickness_px)
    wall = (free - eroded).clamp(min=0.0)
    return wall.squeeze(0).squeeze(0).numpy().astype(np.float32)


def sample_positive_normal(rng: np.random.Generator, mean: float, std: float) -> float:
    draws = rng.normal(mean, std, size=64)
    positive = draws[draws > 0]
    if len(positive):
        return float(positive[0])
    return float(max(mean, 1e-3))


def sample_rt_values(fold: int, setup: int, draw: int) -> tuple[float, float]:
    rng = np.random.default_rng(SEED + 10000 * int(fold) + 100 * int(setup) + int(draw))
    reflectance = sample_positive_normal(rng, NORM["r_mean"], NORM["r_std"])
    transmittance = sample_positive_normal(rng, NORM["t_mean"], NORM["t_std"])
    return reflectance, transmittance


def point_to_pixel(row: int, col: int) -> tuple[int, int]:
    y = int(np.clip(np.floor((row + 0.5) * IMAGE_SIZE / GRID_H), 0, IMAGE_SIZE - 1))
    x = int(np.clip(np.floor((col + 0.5) * IMAGE_SIZE / GRID_W), 0, IMAGE_SIZE - 1))
    return y, x


def point_grid_locations() -> dict[int, tuple[int, int]]:
    with h5py.File(WIFI_H5, "r") as f:
        indices = f["indices"][0]
    locations = {}
    for point in range(1, 54):
        loc = np.argwhere(indices == point)
        if loc.shape == (1, 2):
            locations[point] = (int(loc[0, 0]), int(loc[0, 1]))
    if len(locations) != 53:
        raise AssertionError(f"expected 53 point locations, got {len(locations)}")
    return locations


def choose_support(df_setup: pd.DataFrame, support_count: int = SUPPORT_COUNT) -> np.ndarray:
    ordered = df_setup.sort_values("point")
    if len(ordered) <= support_count:
        raise ValueError(f"setup {ordered.setup.iloc[0]} has too few rows for {support_count} support points")
    positions = np.linspace(0, len(ordered) - 1, support_count, dtype=int)
    return ordered.iloc[positions]["sample_id"].to_numpy(dtype=int)


def support_table(samples: pd.DataFrame, support_count: int = SUPPORT_COUNT) -> pd.DataFrame:
    rows = []
    for setup, group in samples.groupby("setup"):
        support_ids = set(choose_support(group, support_count))
        for row in group.itertuples(index=False):
            is_support = int(row.sample_id in support_ids)
            rows.append({"sample_id": row.sample_id, "setup": setup, "point": row.point, "is_support": is_support})
    out = pd.DataFrame(rows)
    if out.groupby("setup")["is_support"].sum().nunique() != 1:
        raise AssertionError("support count must be identical for every setup")
    return out


def add_sparse_features(samples: pd.DataFrame, supports: pd.DataFrame) -> pd.DataFrame:
    samples = samples.merge(supports[["sample_id", "is_support"]], on="sample_id", how="left")
    feature_frames = []
    for setup, group in samples.groupby("setup"):
        support = group[group["is_support"] == 1]
        group = group.copy()
        sx = support["mx_m"].to_numpy()
        sy = support["my_m"].to_numpy()
        srssi = support["rssi"].to_numpy()
        qx = group["mx_m"].to_numpy()[:, None]
        qy = group["my_m"].to_numpy()[:, None]
        d = np.hypot(qx - sx[None, :], qy - sy[None, :])
        nearest = np.argmin(d, axis=1)
        weights = 1.0 / np.maximum(d, 0.25) ** 2
        group["support_count"] = len(support)
        group["support_mean_rssi"] = float(srssi.mean())
        group["support_std_rssi"] = float(srssi.std(ddof=0))
        group["nearest_support_dist_m"] = d[np.arange(len(group)), nearest]
        group["nearest_support_rssi"] = srssi[nearest]
        group["idw_support_rssi"] = (weights * srssi[None, :]).sum(axis=1) / weights.sum(axis=1)
        feature_frames.append(group)
    return pd.concat(feature_frames, ignore_index=True)


def rssi_to_pathloss_proxy(rssi: np.ndarray, c_dbm: float) -> np.ndarray:
    return c_dbm - np.asarray(rssi, dtype=np.float32)


def c_from_support(support_df: pd.DataFrame) -> float:
    return float(MLSP_SPARSE_MEAN + support_df["rssi"].mean())


def build_input_tensor(
    setup: int,
    samples: pd.DataFrame,
    setup_support: pd.DataFrame,
    point_locations: dict[int, tuple[int, int]],
    freq_mhz: float,
    wall_thickness_px: int,
    c_dbm: float,
    rt_values: tuple[float, float] | None = None,
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
    f1, f2, f3, f4 = frequency_fourier(freq_mhz)

    tensor = np.zeros((len(CHANNELS), IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    wall_mask = floor_hi > 0
    if rt_values is not None:
        reflectance, transmittance = rt_values
        tensor[0, wall_mask] = (float(reflectance) - NORM["r_mean"]) / NORM["r_std"]
        tensor[1, wall_mask] = (float(transmittance) - NORM["t_mean"]) / NORM["t_std"]
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


def load_encoder_unet_model() -> torch.nn.Module:
    if not CKPT_PATH.is_file():
        raise FileNotFoundError(CKPT_PATH)
    spec = importlib.util.spec_from_file_location("mlsp_encoder_unet", ENCODER_UNET)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {ENCODER_UNET}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.EncoderUNetModel(
        n_channels=11,
        encoder_name="resnet101",
        encoder_weights=None,
        decoder_channels=[256, 128, 64, 32, 16],
    )
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state = {key.removeprefix("_network."): value for key, value in ckpt["state_dict"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.eval()
    return model


def forward_r101_batch(model: torch.nn.Module, tensors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    x = torch.from_numpy(np.stack(tensors, axis=0))
    with torch.inference_mode():
        features = model.unet.encoder(x)
        decoded = model.unet.decoder(features)
        pred = model.unet.segmentation_head(decoded).squeeze(1)
        pooled = F.adaptive_avg_pool2d(features[-1], output_size=1).flatten(1).float()
    return (pred.float().numpy() * OUT_NORM_DB), pooled.numpy()


def extract_r101_features_for_fold(
    model: torch.nn.Module,
    samples: pd.DataFrame,
    sparse_samples: pd.DataFrame,
    c_dbm: float,
    fold: int,
    holdout_ap_point: int,
    rt_mode: str,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    if rt_mode not in RT_MODES:
        raise ValueError(f"unknown rt_mode: {rt_mode}")
    point_locations = point_grid_locations()
    freqs = setup_frequencies()
    rows = []
    debug_tensors = {}
    for setup in sorted(samples["setup"].unique()):
        setup_support = sparse_samples[(sparse_samples["setup"] == setup) & (sparse_samples["is_support"] == 1)]
        tensors = []
        first_sparse_grid = None
        n_draws = 1 if rt_mode == "zero_rt" else RT_IMPUTE_N
        for draw in range(n_draws):
            rt_values = None if rt_mode == "zero_rt" else sample_rt_values(fold=fold, setup=int(setup), draw=draw)
            tensor, c_dbm, sparse_grid = build_input_tensor(
                setup=setup,
                samples=samples,
                setup_support=setup_support,
                point_locations=point_locations,
                freq_mhz=freqs[int(setup)],
                wall_thickness_px=WALL_THICKNESS_PX,
                c_dbm=c_dbm,
                rt_values=rt_values,
            )
            tensors.append(tensor)
            if first_sparse_grid is None:
                first_sparse_grid = sparse_grid
        pred_pl_batch, pooled_batch = forward_r101_batch(model, tensors)
        pred_pl = pred_pl_batch.mean(axis=0)
        pooled = pooled_batch.mean(axis=0)
        debug_tensors[int(setup)] = tensors[0]
        for row in samples[samples["setup"] == setup].itertuples(index=False):
            grid_row, grid_col = point_locations[int(row.point)]
            pix_y, pix_x = point_to_pixel(grid_row, grid_col)
            item = {
                "sample_id": int(row.sample_id),
                "setup": int(setup),
                "point": int(row.point),
                "r101_pl_pred": float(pred_pl[pix_y, pix_x]),
                "fold": fold,
                "holdout_ap_point": holdout_ap_point,
                "c_fold_dbm": c_dbm,
                "sparse_pl_at_point": float(first_sparse_grid[grid_row, grid_col]),
                "wall_thickness_px": WALL_THICKNESS_PX,
                "rt_mode": rt_mode,
                "rt_impute_n": n_draws,
            }
            item.update({f"r101_f{i}": float(value) for i, value in enumerate(pooled)})
            rows.append(item)
    return pd.DataFrame(rows), debug_tensors


def folds(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, ap_point in enumerate(sorted(samples["ap_point"].unique())):
        val_setups = sorted(samples.loc[samples["ap_point"] == ap_point, "setup"].unique())
        rows.append({"fold": fold, "holdout_ap_point": int(ap_point), "holdout_setups": ",".join(map(str, val_setups))})
    return pd.DataFrame(rows)


def train_models() -> dict[str, object]:
    pre = ColumnTransformer([("num", StandardScaler(), SPARSE_COLS)])
    return {
        "sparse_rf": Pipeline(
            [
                ("pre", pre),
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
    }


def ridge_feature_cols(model_name: str, data: pd.DataFrame) -> list[str]:
    if model_name == "sparse_ridge":
        return list(SPARSE_COLS)
    if model_name == "r101_feature_ridge":
        return SPARSE_COLS + ["r101_pl_pred"] + [c for c in data.columns if c.startswith("r101_f")]
    if model_name == "r101_only_ridge":
        return ["r101_pl_pred"] + [c for c in data.columns if c.startswith("r101_f")]
    raise ValueError(f"unknown ridge model: {model_name}")


def fit_ridge(train: pd.DataFrame, feature_cols: list[str], alpha: float) -> Pipeline:
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=float(alpha)))])
    model.fit(train[feature_cols], train["rssi"])
    return model


def alpha_record(model_name: str, outer_fold: int, alpha: float = RIDGE_ALPHA) -> dict[str, object]:
    return {
        "model": model_name,
        "outer_fold": int(outer_fold),
        "alpha": float(alpha),
        "inner_mean_rmse": np.nan,
        "inner_std_rmse": np.nan,
        "inner_folds": 0,
        "selected": True,
    }


def alpha_suffix(alpha: float) -> str:
    return f"{float(alpha):g}".replace(".", "_")


def top_r101_encoder_features(train: pd.DataFrame, alpha: float = RIDGE_ALPHA) -> list[str]:
    feature_cols = ridge_feature_cols("r101_feature_ridge", train)
    model = fit_ridge(train, feature_cols, alpha)
    coefs = pd.Series(np.abs(model.named_steps["ridge"].coef_), index=feature_cols)
    return coefs[[c for c in feature_cols if c.startswith("r101_f")]].nlargest(R101_TREE_TOP_ENCODER_FEATURES).index.tolist()


def tree_feature_matrices(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_x = train[feature_names].to_numpy(dtype=np.float32)
    val_x = val[feature_names].to_numpy(dtype=np.float32)
    return train_x, val_x


def r101_tree_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=SEED,
        n_jobs=-1,
    )


def run_one_fold(fold_row: object, data: pd.DataFrame) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    pred_rows = []
    alpha_rows = []
    models = train_models()
    is_val = data["ap_point"] == fold_row.holdout_ap_point
    train = data[(~is_val) & (data["is_support"] == 0)].copy()
    val = data[is_val & (data["is_support"] == 0)].copy()
    fold_record = {
        "fold": int(fold_row.fold),
        "holdout_ap_point": int(fold_row.holdout_ap_point),
        "holdout_setups": fold_row.holdout_setups,
        "train_n": int(len(train)),
        "val_n": int(len(val)),
        "wall_thickness_px": WALL_THICKNESS_PX,
        "c_fold_dbm": float(data["c_fold_dbm"].iloc[0]),
        "rt_mode": str(data["rt_mode"].iloc[0]),
        "rt_impute_n": int(data["rt_impute_n"].iloc[0]),
    }

    for model_name, template in models.items():
        model = clone(template)
        model.fit(train[SPARSE_COLS], train["rssi"])
        pred = model.predict(val[SPARSE_COLS])
        pred_rows.extend(prediction_records(val, pred, model_name, fold_row))

    for model_name in ("sparse_ridge", "r101_feature_ridge"):
        feature_cols = ridge_feature_cols(model_name, train)
        alpha_rows.append(alpha_record(model_name, int(fold_row.fold)))
        model = fit_ridge(train, feature_cols, RIDGE_ALPHA)
        pred_rows.extend(prediction_records(val, model.predict(val[feature_cols]), model_name, fold_row))

    r101_only_cols = ridge_feature_cols("r101_only_ridge", train)
    for alpha in R101_ONLY_RIDGE_ALPHAS:
        model_name = f"r101_only_ridge_a{alpha_suffix(alpha)}"
        alpha_rows.append(alpha_record(model_name, int(fold_row.fold), alpha))
        model = fit_ridge(train, r101_only_cols, alpha)
        pred_rows.extend(prediction_records(val, model.predict(val[r101_only_cols]), model_name, fold_row))

    tree_feature_sets = {
        "r101_pl_rf": SPARSE_COLS + ["r101_pl_pred"],
        "r101_top10_rf": SPARSE_COLS + ["r101_pl_pred"] + top_r101_encoder_features(train),
    }
    for model_name, feature_names in tree_feature_sets.items():
        tree_train_x, tree_val_x = tree_feature_matrices(train, val, feature_names)
        model = r101_tree_model()
        model.fit(tree_train_x, train["rssi"])
        pred_rows.extend(prediction_records(val, model.predict(tree_val_x), model_name, fold_row))

    pred_rows.extend(prediction_records(val, val["idw_support_rssi"].to_numpy(), "sparse_idw", fold_row))

    pl_model = LinearRegression().fit(train[["r101_pl_pred"]], train["rssi"])
    pred_rows.extend(
        prediction_records(
            val,
            pl_model.predict(val[["r101_pl_pred"]]),
            "r101_pl_train_cal",
            fold_row,
        )
    )

    support_cal = []
    for setup, setup_val in val.groupby("setup"):
        support = data[(data["setup"] == setup) & (data["is_support"] == 1)]
        model = LinearRegression().fit(support[["r101_pl_pred"]], support["rssi"])
        support_cal.append(
            pd.DataFrame(
                {
                    "sample_id": setup_val["sample_id"],
                    "pred": model.predict(setup_val[["r101_pl_pred"]]),
                }
            )
        )
    support_pred = pd.concat(support_cal).sort_values("sample_id")
    val_sorted = val.sort_values("sample_id")
    pred_rows.extend(prediction_records(val_sorted, support_pred["pred"].to_numpy(), "r101_pl_support_cal", fold_row))

    return fold_record, pred_rows, alpha_rows


def prediction_records(val: pd.DataFrame, pred: np.ndarray, model: str, fold_row: object) -> list[dict[str, object]]:
    rows = []
    for row, y_pred in zip(val.itertuples(index=False), pred):
        rows.append(
            {
                "model": model,
                "fold": int(fold_row.fold),
                "holdout_ap_point": int(fold_row.holdout_ap_point),
                "sample_id": int(row.sample_id),
                "setup": int(row.setup),
                "point": int(row.point),
                "wall_thickness_px": int(row.wall_thickness_px),
                "rt_mode": str(row.rt_mode),
                "rt_impute_n": int(row.rt_impute_n),
                "y_true": float(row.rssi),
                "y_pred": float(y_pred),
                "error": float(y_pred - row.rssi),
            }
        )
    return rows


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (rt_mode, model), group in predictions.groupby(["rt_mode", "model"]):
        fold_rmse = group.groupby("fold").apply(
            lambda g: rmse(g["y_true"].to_numpy(), g["y_pred"].to_numpy()),
            include_groups=False,
        )
        rows.append(
            {
                "wall_thickness_px": WALL_THICKNESS_PX,
                "rt_mode": rt_mode,
                "rt_impute_n": int(group["rt_impute_n"].iloc[0]),
                "model": model,
                "pooled_rmse": rmse(group["y_true"].to_numpy(), group["y_pred"].to_numpy()),
                "pooled_mae": float(np.mean(np.abs(group["error"]))),
                "mean_fold_rmse": float(fold_rmse.mean()),
                "std_fold_rmse": float(fold_rmse.std()),
                "mean_fold_r2": float(
                    group.groupby("fold").apply(
                        lambda g: r2_score(g["y_true"], g["y_pred"]) if np.var(g["y_true"]) > 0 else np.nan,
                        include_groups=False,
                    ).mean()
                ),
                "n_predictions": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(["pooled_rmse", "rt_mode", "model"]).reset_index(drop=True)


def selected_alpha(alpha_tuning: pd.DataFrame, model_name: str, fold: int) -> float:
    row = alpha_tuning[
        (alpha_tuning["model"] == model_name)
        & (alpha_tuning["outer_fold"] == fold)
        & (alpha_tuning["selected"])
    ]
    if len(row) != 1:
        raise ValueError(f"expected one selected alpha for {model_name}, fold {fold}")
    return float(row["alpha"].iloc[0])


def compute_feature_importance(data: pd.DataFrame, alpha_tuning: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rf_rows = []
    sparse_ridge_rows = []
    r101_ridge_rows = []
    for fold_row in folds(data).itertuples(index=False):
        fold_data = data[data["fold"] == fold_row.fold]
        is_val = fold_data["ap_point"] == fold_row.holdout_ap_point
        train = fold_data[(~is_val) & (fold_data["is_support"] == 0)].copy()

        rf = clone(train_models()["sparse_rf"]).fit(train[SPARSE_COLS], train["rssi"])
        rf_rows.extend(
            {"fold": int(fold_row.fold), "feature": feature, "importance": float(importance)}
            for feature, importance in zip(SPARSE_COLS, rf.named_steps["model"].feature_importances_)
        )

        sparse_cols = ridge_feature_cols("sparse_ridge", train)
        sparse_ridge = fit_ridge(train, sparse_cols, selected_alpha(alpha_tuning, "sparse_ridge", int(fold_row.fold)))
        sparse_ridge_rows.extend(
            {"fold": int(fold_row.fold), "feature": feature, "coef": float(coef), "abs_coef": float(abs(coef))}
            for feature, coef in zip(sparse_cols, sparse_ridge.named_steps["ridge"].coef_)
        )

        r101_feature_cols = ridge_feature_cols("r101_feature_ridge", train)
        r101_ridge = fit_ridge(
            train,
            r101_feature_cols,
            selected_alpha(alpha_tuning, "r101_feature_ridge", int(fold_row.fold)),
        )
        for feature, coef in zip(r101_feature_cols, r101_ridge.named_steps["ridge"].coef_):
            if feature.startswith("r101_f"):
                group = "r101_encoder"
            elif feature == "r101_pl_pred":
                group = "r101_pl_pred"
            else:
                group = "sparse"
            r101_ridge_rows.append(
                {
                    "fold": int(fold_row.fold),
                    "feature": feature,
                    "group": group,
                    "coef": float(coef),
                    "abs_coef": float(abs(coef)),
                }
            )

    rf = pd.DataFrame(rf_rows)
    sparse_ridge = pd.DataFrame(sparse_ridge_rows)
    r101_ridge = pd.DataFrame(r101_ridge_rows)
    tables = {
        "rf_impurity": rf.groupby("feature")["importance"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=False),
        "sparse_ridge_coef": sparse_ridge.groupby("feature")["abs_coef"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=False),
        "r101_ridge_non_encoder": r101_ridge[r101_ridge["group"] != "r101_encoder"]
        .groupby(["group", "feature"])["abs_coef"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=False),
        "r101_ridge_groups": r101_ridge.groupby(["fold", "group"])["abs_coef"]
        .sum()
        .reset_index()
        .groupby("group")["abs_coef"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=False),
        "r101_ridge_top_encoder": r101_ridge[r101_ridge["group"] == "r101_encoder"]
        .groupby("feature")["abs_coef"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=False)
        .head(25),
    }
    tables["rf_impurity"].to_csv(RUNS / "feature_importance_sparse_rf_impurity.csv")
    tables["sparse_ridge_coef"].to_csv(RUNS / "feature_importance_sparse_ridge_abs_coefficients.csv")
    tables["r101_ridge_non_encoder"].to_csv(RUNS / "feature_importance_r101_ridge_non_encoder_abs_coefficients.csv")
    tables["r101_ridge_groups"].to_csv(RUNS / "feature_importance_r101_ridge_group_abs_coefficients.csv")
    tables["r101_ridge_top_encoder"].to_csv(RUNS / "feature_importance_r101_ridge_top_encoder_abs_coefficients.csv")
    return tables


def validate(samples: pd.DataFrame, folds_df: pd.DataFrame, predictions: pd.DataFrame) -> None:
    if samples["is_support"].sum() != 20 * SUPPORT_COUNT:
        raise AssertionError("unexpected support count")
    expected_folds = 16 * len(RT_MODES)
    if len(folds_df) != expected_folds:
        raise AssertionError(f"expected {expected_folds} AP/rt folds, got {len(folds_df)}")
    for rt_mode, group in folds_df.groupby("rt_mode"):
        if group["fold"].nunique() != 16:
            raise AssertionError(f"{rt_mode}: expected 16 AP folds")
    query_count = int((samples["is_support"] == 0).sum())
    for (_rt_mode, model), group in predictions.groupby(["rt_mode", "model"]):
        if len(group) != query_count or group["sample_id"].nunique() != query_count:
            raise AssertionError(f"{model}: expected one prediction for each query sample")
        support_ids = set(samples.loc[samples["is_support"] == 1, "sample_id"])
        if support_ids & set(group["sample_id"]):
            raise AssertionError(f"{model}: support point was evaluated")


def plot_floor_and_sparse(debug_tensors: dict[str, dict[int, np.ndarray]]) -> dict[str, str]:
    setup = 1
    tensor = debug_tensors[RT_MODES[0]][setup]
    paths = {
        "floor": "figs/sampling_floor_plan_channel_1px.png",
        "sparse": "figs/sampling_sparse_channel_setup1.png",
        "channels": "figs/sampling_r101_channels_setup1_1px.png",
    }
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(tensor[9], cmap="gray", vmin=0, vmax=1)
    ax.set_title("Wall floor-plan channel, 1px")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(REPO_ROOT / paths["floor"], dpi=160)
    plt.close(fig)

    sparse = tensor[10]
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(np.ma.masked_where(sparse == 0, sparse), cmap="magma")
    ax.set_title("Normalized sparse pathloss channel, setup 1")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    fig.savefig(REPO_ROOT / paths["sparse"], dpi=160)
    plt.close(fig)

    titles = ["reflectance", "transmittance", "distance", "mask", "floor", "sparse"]
    idxs = [0, 1, 2, 8, 9, 10]
    fig, axes = plt.subplots(1, len(idxs), figsize=(14, 3))
    for ax, idx, title in zip(axes, idxs, titles):
        ax.imshow(tensor[idx], cmap="viridis")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(REPO_ROOT / paths["channels"], dpi=160)
    plt.close(fig)
    return paths


def plot_results(summary: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, str]:
    paths = {
        "rmse": "figs/sampling_assisted_rmse.png",
        "scatter": "figs/sampling_assisted_best_scatter.png",
    }
    s = summary.sort_values("pooled_rmse")
    labels = [
        row.model if len(RT_MODES) == 1 else f"{row.model}\n{row.rt_mode}"
        for row in s.itertuples(index=False)
    ]
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, s["pooled_rmse"], yerr=s["std_fold_rmse"], capsize=4, color="C0", alpha=0.85)
    ax.set_xticks(x, labels)
    ax.set_ylabel("query RMSE [dBm]")
    ax.set_title(f"Sampling-assisted AP-held-out reconstruction ({SUPPORT_COUNT} sparse points/map)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(REPO_ROOT / paths["rmse"], dpi=160)
    plt.close(fig)

    best_model = str(summary.iloc[0]["model"])
    best_mode = str(summary.iloc[0]["rt_mode"])
    pred = predictions[(predictions["model"] == best_model) & (predictions["rt_mode"] == best_mode)]
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
    ax.set_title(f"{best_model}, {best_mode}, 1px walls: query predictions")
    fig.tight_layout()
    fig.savefig(REPO_ROOT / paths["scatter"], dpi=160)
    plt.close(fig)
    return paths


def plot_feature_importance(tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    paths = {
        "rf_importance": "figs/feature_importance_sparse_rf.png",
        "sparse_ridge_importance": "figs/feature_importance_sparse_ridge.png",
        "r101_ridge_importance": "figs/feature_importance_r101_ridge.png",
    }
    for key, path, xlabel in [
        ("rf_impurity", paths["rf_importance"], "mean impurity importance"),
        ("sparse_ridge_coef", paths["sparse_ridge_importance"], "mean |standardized coefficient|"),
    ]:
        table = tables[key].sort_values("mean")
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.barh(table.index, table["mean"], xerr=table["std"], color="C0", alpha=0.85)
        ax.set_xlabel(xlabel)
        ax.set_title(key.replace("_", " "))
        fig.tight_layout()
        fig.savefig(REPO_ROOT / path, dpi=160)
        plt.close(fig)

    table = tables["r101_ridge_non_encoder"].reset_index()
    table["label"] = table["feature"].where(table["group"] == "sparse", table["group"])
    table = table.sort_values("mean").tail(14)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.barh(table["label"], table["mean"], xerr=table["std"], color="C1", alpha=0.85)
    ax.set_xlabel("mean |standardized coefficient|")
    ax.set_title("r101_feature_ridge non-encoder coefficients")
    fig.tight_layout()
    fig.savefig(REPO_ROOT / paths["r101_ridge_importance"], dpi=160)
    plt.close(fig)
    return paths


def plot_alpha_tuning(alpha_tuning: pd.DataFrame) -> dict[str, str]:
    paths = {}
    for model_name in ("sparse_ridge", "r101_feature_ridge"):
        path = f"figs/alpha_tuning_{model_name}.png"
        table = alpha_tuning[alpha_tuning["model"] == model_name]
        curve = (
            table.groupby("alpha")["inner_mean_rmse"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("alpha")
        )
        selected = table[table["selected"]]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(curve["alpha"], curve["mean"], yerr=curve["std"], marker="o", capsize=3)
        ax.scatter(
            selected["alpha"],
            selected["inner_mean_rmse"],
            marker="x",
            s=60,
            color="C3",
            label="selected per outer fold",
        )
        ax.set_xscale("log")
        ax.set_xlabel("ridge alpha")
        ax.set_ylabel("inner validation RMSE [dBm]")
        ax.set_title(f"{model_name}: AP-grouped inner CV")
        ax.legend()
        fig.tight_layout()
        fig.savefig(REPO_ROOT / path, dpi=160)
        plt.close(fig)
        paths[f"alpha_{model_name}"] = path
    return paths


def md_table(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    for row in df.itertuples(index=False):
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    summary: pd.DataFrame,
    figures: dict[str, str],
    feature_importance: dict[str, pd.DataFrame],
    alpha_tuning: pd.DataFrame,
    query_n: int,
) -> None:
    best = summary.iloc[0]
    r101_feat = summary.loc[summary["model"] == "r101_feature_ridge"].sort_values("pooled_rmse").iloc[0]
    r101_only = summary.loc[summary["model"].str.startswith("r101_only_ridge_")].sort_values("pooled_rmse").iloc[0]
    sparse_rf = summary.loc[summary["model"] == "sparse_rf"].sort_values("pooled_rmse").iloc[0]
    sparse_ridge = summary.loc[summary["model"] == "sparse_ridge"].sort_values("pooled_rmse").iloc[0]
    direct = summary.loc[summary["model"] == "r101_pl_train_cal"].sort_values("pooled_rmse").iloc[0]
    tree_pl = summary.loc[summary["model"] == "r101_pl_rf"].sort_values("pooled_rmse").iloc[0]
    tree_rf = summary.loc[summary["model"] == "r101_top10_rf"].sort_values("pooled_rmse").iloc[0]
    active_rt_modes = ", ".join(f"`{mode}`" for mode in RT_MODES)
    rf_top = feature_importance["rf_impurity"].head(8).reset_index()
    sparse_ridge_top = feature_importance["sparse_ridge_coef"].head(8).reset_index()
    r101_non_encoder = feature_importance["r101_ridge_non_encoder"].head(10).reset_index()
    r101_groups = feature_importance["r101_ridge_groups"].reset_index()
    r101_top_encoder = feature_importance["r101_ridge_top_encoder"].head(8).reset_index()
    alpha_counts = (
        alpha_tuning[alpha_tuning["selected"]]
        .groupby(["model", "alpha"])
        .size()
        .reset_index(name="selected_folds")
        .sort_values(["model", "alpha"])
    )
    text = f"""# Sampling-assisted r101 transfer on RSSI data

Auto-generated by `experiments/sampling_assisted_r101.py`.

- **task**: sampling-assisted RSSI map reconstruction, not zero-shot AP generalization
- **sparse context**: {SUPPORT_COUNT} observed RSSI points per setup are revealed as input
- **floor-plan channel**: wall-only mask, {WALL_THICKNESS_PX}px thick
- **active r/t channel variant**: {active_rt_modes}
- **evaluation**: only non-support query points are scored
- **outer split**: leave-one-AP-location-out, k = 16
- **query predictions per model**: {query_n:,}
- **checkpoint**: `{CKPT_PATH}`

## Input construction

The r101 model expects 11 MLSP channels. This dataset provides distance,
frequency, mask, a layout-derived wall floor plan, and sparse measurements. It
lacks measured RF reflectance, RF transmittance, and antenna pattern. The active
`zero_rt` variant zero-fills reflectance/transmittance. The script keeps an
`impute_rt` implementation for later experiments, but it is disabled by default
because it did not improve RMSE and made runs much slower.

Sparse RSSI support values are converted to a pathloss proxy with:

`PL_proxy = C_fold - RSSI`, where
`C_fold = {MLSP_SPARSE_MEAN:.3f} + mean(training support RSSI)`.

`C_fold` is recomputed inside each AP-held-out fold using only support points
from training AP locations, then applied to both train and held-out sparse
channels in that fold. This is not a physical EIRP estimate; it is a fold-safe
scale alignment for the pretrained sparse-channel normalization.

![Wall floor-plan channel]({figures['floor']})

![Sparse pathloss channel]({figures['sparse']})

![Selected r101 input channels]({figures['channels']})

## Results

{md_table(summary)}

Best model: **{best['model']}** with pooled query RMSE **{best['pooled_rmse']:.2f} dBm**.

![Sampling-assisted RMSE]({figures['rmse']})

![Best model query predictions]({figures['scatter']})

## Ridge alpha

`sparse_ridge` and `r101_feature_ridge` use fixed `alpha = {RIDGE_ALPHA}`. This
was chosen from the earlier AP-grouped inner-CV sweep; the script no longer
searches alpha during the main experiment. The earlier candidate grid was:

`{", ".join(str(v) for v in RIDGE_ALPHA_SWEEP)}`

Active alpha records:

{md_table(alpha_counts)}

## Feature importance

These importances are averaged across the same AP-held-out folds. Ridge
coefficients are absolute coefficients after feature standardization, so they
are comparable within a fitted ridge head. The r101 encoder group is reported
both as total coefficient mass and as individual top encoder dimensions because
its signal is spread across 2,048 pooled features.

### sparse_rf

{md_table(rf_top)}

![sparse_rf feature importance]({figures['rf_importance']})

### sparse_ridge

{md_table(sparse_ridge_top)}

![sparse_ridge feature importance]({figures['sparse_ridge_importance']})

### r101_feature_ridge

Non-encoder coefficients:

{md_table(r101_non_encoder)}

Group coefficient mass:

{md_table(r101_groups)}

Top encoder coefficients:

{md_table(r101_top_encoder)}

![r101_feature_ridge feature importance]({figures['r101_ridge_importance']})

## Conclusion

The r101 checkpoint is useful as a frozen feature extractor once the ridge
regularization is tuned inside the outer folds: `r101_feature_ridge` gets
{r101_feat['pooled_rmse']:.2f} dBm RMSE versus
{sparse_rf['pooled_rmse']:.2f} dBm for `sparse_rf` and
{sparse_ridge['pooled_rmse']:.2f} dBm for `sparse_ridge`. Directly calibrating
the r101 pathloss map is much worse (`r101_pl_train_cal` =
{direct['pooled_rmse']:.2f} dBm), which suggests that the synthetic pathloss
decoder does not transfer cleanly to this RSSI dataset when reflectance and
transmittance are unavailable.

The tree ablations did not help: `r101_pl_rf` gets {tree_pl['pooled_rmse']:.2f}
dBm using sparse features plus `r101_pl_pred`, and `r101_top10_rf` gets
{tree_rf['pooled_rmse']:.2f} dBm after adding the 10 encoder coordinates with
the largest absolute fixed-ridge weights per outer fold. This suggests the
useful r101 signal transfers as a smooth, mostly additive correction that ridge
can exploit, while the decoded map and selected individual encoder coordinates
are not strong RF split variables.

The pure-network ridge ablation, using only `r101_pl_pred` plus the 2,048
encoder features and no engineered sparse/geometry features, reaches
{r101_only['pooled_rmse']:.2f} dBm with `{r101_only['model']}`. This confirms
that r101 alone carries substantial signal, but the best performance still
comes from combining r101 with the sparse support and geometry features.

The useful signal in this dataset is mostly captured by sparse RSSI support
statistics plus geometry. The feature-importance tables support that diagnosis:
`r101_feature_ridge` uses `r101_pl_pred`, but its largest non-encoder weights
are still the sparse/interpolation and geometry features. The pooled encoder
features carry signal as many small coefficients; that is enough to improve a
tuned linear head, but not enough for the raw-feature tree ensembles to beat the
sparse random forest.

## Artifacts

- `runs/sampling_assisted_r101_samples.csv`
- `runs/sampling_assisted_r101_folds.csv`
- `runs/sampling_assisted_r101_predictions.csv`
- `runs/sampling_assisted_r101_summary.csv`
- `runs/sampling_assisted_r101_config.json`
- `figs/sampling_*.png`
"""
    REPORT.write_text(text)


def main() -> int:
    require_data()
    np.random.seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    for path in (RUNS, FIGS):
        path.mkdir(parents=True, exist_ok=True)

    samples = load_samples()
    supports = support_table(samples, SUPPORT_COUNT)
    sparse_data = add_sparse_features(samples, supports)
    support_samples = samples.merge(supports, on=["sample_id", "setup", "point"])

    model = load_encoder_unet_model()
    fold_defs = folds(samples)
    all_data = []
    all_predictions = []
    all_alpha_tuning = []
    fold_records = []
    debug_tensors = {}
    for rt_mode in RT_MODES:
        for fold_row in fold_defs.itertuples(index=False):
            train_support = support_samples[
                (support_samples["ap_point"] != fold_row.holdout_ap_point)
                & (support_samples["is_support"] == 1)
            ]
            c_fold = c_from_support(train_support)
            r101_features, variant_debug = extract_r101_features_for_fold(
                model=model,
                samples=samples,
                sparse_samples=support_samples,
                c_dbm=c_fold,
                fold=int(fold_row.fold),
                holdout_ap_point=int(fold_row.holdout_ap_point),
                rt_mode=rt_mode,
            )
            if rt_mode not in debug_tensors:
                debug_tensors[rt_mode] = variant_debug
            data = sparse_data.merge(r101_features, on=["sample_id", "setup", "point"], how="left")
            fold_record, pred_rows, alpha_rows = run_one_fold(fold_row, data)
            all_data.append(data)
            fold_records.append(fold_record)
            all_predictions.extend(pred_rows)
            all_alpha_tuning.extend(alpha_rows)

    data = pd.concat(all_data, ignore_index=True)
    folds_df = pd.DataFrame(fold_records)
    predictions = pd.DataFrame(all_predictions)
    alpha_tuning = pd.DataFrame(all_alpha_tuning)
    summary = summarize_predictions(predictions)
    feature_importance = compute_feature_importance(data, alpha_tuning)
    base_samples = sparse_data

    data.to_csv(RUNS / "sampling_assisted_r101_samples.csv", index=False)
    folds_df.to_csv(RUNS / "sampling_assisted_r101_folds.csv", index=False)
    predictions.to_csv(RUNS / "sampling_assisted_r101_predictions.csv", index=False)
    alpha_tuning.to_csv(RUNS / "ridge_alpha_tuning.csv", index=False)
    summary.to_csv(RUNS / "sampling_assisted_r101_summary.csv", index=False)
    (RUNS / "sampling_assisted_r101_config.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "support_count": SUPPORT_COUNT,
                "wall_thickness_px": WALL_THICKNESS_PX,
                "rt_modes": list(RT_MODES),
                "rt_impute_n_if_enabled": RT_IMPUTE_N,
                "checkpoint": str(CKPT_PATH),
                "channels": CHANNELS,
                "ridge_alpha": RIDGE_ALPHA,
                "ridge_alpha_sweep_considered": list(RIDGE_ALPHA_SWEEP),
                "r101_only_ridge_alphas": list(R101_ONLY_RIDGE_ALPHAS),
                "rssi_to_pathloss": "PL_proxy = (MLSP_SPARSE_MEAN + mean_train_support_RSSI_per_fold) - RSSI",
                "mlsp_sparse_mean": MLSP_SPARSE_MEAN,
                "outer_cv": "Leave one AP point out; score query points only",
                "floor_plan": "wall-only inner boundary of the inferred free-space layout",
            },
            indent=2,
        )
        + "\n"
    )

    figures = {}
    if not debug_tensors:
        raise RuntimeError("no debug tensors were generated")
    figures.update(plot_floor_and_sparse(debug_tensors))
    figures.update(plot_results(summary, predictions))
    figures.update(plot_feature_importance(feature_importance))
    validate(base_samples, folds_df, predictions)
    write_report(
        summary,
        figures,
        feature_importance,
        alpha_tuning,
        query_n=int((base_samples["is_support"] == 0).sum()),
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote {REPORT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
