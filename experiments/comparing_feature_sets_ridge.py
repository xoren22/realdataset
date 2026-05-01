#!/usr/bin/env python
"""Fixed-alpha ridge comparison: 12 sparse features vs +1 R101 scalar."""
from __future__ import annotations

import json
import re
import sys
import argparse
import importlib.util
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RSSI_CSV = DATA_DIR / "RSSI_raw_data.csv"
WIFI_H5 = DATA_DIR / "WiFi_RSSI_data.h5"
SEEDS = tuple(range(1000, 1010))
SUPPORT_COUNT = 5
RIDGE_ALPHA = 0.1
CELL_M = 0.5
GRID_H = 28
GRID_W = 30
IMAGE_SIZE = 256
WALL_THICKNESS_PX = 1
OUT_NORM_DB = 160.0
MLSP_SPARSE_MEAN = 110.1460952758789
MLSP_SPARSE_STD = 42.0374641418457
FIRST_POINT_MISSING_SETUPS = {5, 7, 14}
DEFAULT_CKPT_PATH = Path(
    "/mnt/weka/kpetrosyan/synthetic_pretraining_checkpoints/"
    "r101_bs181_lr3e-03_wr0.1_dr0.1_wd1e-03/"
    "2026-04-14_09-58-12.473943/every/step_00350000_every.ckpt"
)
DEFAULT_MLSP_REPO = Path("/home/kpetrosyan/mlsp_wair_d")
NORM = {
    "d_log_mean": 3.5391982793807983,
    "d_log_std": 0.7967258393764496,
}
CHANNELS = 11

OUT_DIR = REPO_ROOT / "runs" / "comparing_feature_sets_ridge"
CACHE_DIR = OUT_DIR / "r101_global_feature_cache"

SPARSE_12_COLS = [
    "ap_x_m",
    "ap_y_m",
    "mx_m",
    "my_m",
    "distance_m",
    "log_distance_m",
    "same_room",
    "support_mean_rssi",
    "support_std_rssi",
    "nearest_support_dist_m",
    "nearest_support_rssi",
    "idw_support_rssi",
]

FEATURE_SETS = {
    "sparse12_ridge": SPARSE_12_COLS,
    "sparse12_plus_r101_pl_ridge": SPARSE_12_COLS + ["r101_pl_pred"],
}


def require_data() -> None:
    missing = [path for path in (RSSI_CSV, WIFI_H5) if not path.exists()]
    if missing:
        joined = ", ".join(str(path.relative_to(REPO_ROOT)) for path in missing)
        raise FileNotFoundError(f"missing required data files: {joined}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT_PATH, help="R101 checkpoint for fresh inference.")
    parser.add_argument("--mlsp-repo", type=Path, default=DEFAULT_MLSP_REPO, help="Repo containing src/networks/encoder_unet.py.")
    parser.add_argument("--refresh-cache", action="store_true", help="Rerun R101 inference even if a cache exists.")
    parser.add_argument("--cache-tag", default=None, help="Optional cache namespace for non-default checkpoints.")
    parser.add_argument("--threads", type=int, default=8, help="Torch CPU thread count for fresh inference.")
    return parser.parse_args()


def point_to_room(point: int) -> str:
    if point <= 30:
        return "office"
    if point <= 45:
        return "corridor"
    return "elevator"


def parse_rssi(value: object) -> float:
    match = re.search(r"-?\d+", str(value))
    if not match:
        raise ValueError(f"could not parse RSSI value: {value!r}")
    return float(match.group())


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
                    "setup": int(setup),
                    "source_row": int(source_row),
                    "point": int(point),
                    "ap_point": int(ap_point),
                    "ap_x_m": ap_x_m,
                    "ap_y_m": ap_y_m,
                    "mx_m": mx_m,
                    "my_m": my_m,
                    "same_room": int(room_meas == room_ap),
                    "rssi": float(setup_raw.loc[source_row, "rssi"]),
                }
            )

    df = pd.DataFrame(rows)
    df["distance_m"] = np.hypot(df["mx_m"] - df["ap_x_m"], df["my_m"] - df["ap_y_m"])
    df["log_distance_m"] = np.log10(df["distance_m"].clip(lower=CELL_M / 2))
    return df


def folds(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, ap_point in enumerate(sorted(samples["ap_point"].unique())):
        setups = sorted(samples.loc[samples["ap_point"] == ap_point, "setup"].unique())
        rows.append({"fold": int(fold), "holdout_ap_point": int(ap_point), "holdout_setups": ",".join(map(str, setups))})
    return pd.DataFrame(rows)


def free_space_grid() -> np.ndarray:
    floor = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    floor[0:9, 0:11] = 1.0
    floor[9:12, 0:29] = 1.0
    floor[12:27, 24:27] = 1.0
    return floor


def resize_nearest(grid: np.ndarray) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(grid.astype(np.float32))[None, None]
    y = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="nearest")
    return y.squeeze(0).squeeze(0).numpy()


def wall_floor_plan_image() -> np.ndarray:
    import torch
    import torch.nn.functional as F

    free = torch.from_numpy(resize_nearest(free_space_grid()))[None, None]
    eroded = -F.max_pool2d(-free, kernel_size=2 * WALL_THICKNESS_PX + 1, stride=1, padding=WALL_THICKNESS_PX)
    wall = (free - eroded).clamp(min=0.0)
    return wall.squeeze(0).squeeze(0).numpy().astype(np.float32)


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


def checkpoint_tag(checkpoint: Path, cache_tag: str | None) -> str:
    if cache_tag:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", cache_tag)
    if checkpoint == DEFAULT_CKPT_PATH:
        return "default"
    digest = hashlib.sha1(str(checkpoint).encode("utf-8")).hexdigest()[:8]
    return f"{checkpoint.stem}_{digest}"


def seed_cache_path(seed: int, tag: str = "default") -> Path:
    if tag == "default":
        return CACHE_DIR / f"seed_{seed}_r101_pl_and_encoder.npz"
    return CACHE_DIR / f"seed_{seed}_{tag}_r101_pl.npz"


def seed_support_path(seed: int) -> Path:
    return CACHE_DIR / f"seed_{seed}_support_table.csv"


def load_supports(samples: pd.DataFrame, seed: int) -> pd.DataFrame:
    path = seed_support_path(seed)
    if path.exists():
        return pd.read_csv(path)
    return random_support_table(samples, seed)


def add_sparse_features(samples: pd.DataFrame, supports: pd.DataFrame) -> pd.DataFrame:
    samples = samples.merge(supports[["sample_id", "is_support"]], on="sample_id", how="left")
    feature_frames = []
    for _setup, group in samples.groupby("setup"):
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
    c_dbm: float,
) -> np.ndarray:
    setup_all = samples[samples["setup"] == setup]
    floor_hi = wall_floor_plan_image()
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

    tensor = np.zeros((CHANNELS, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
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
    return tensor


def load_r101_model(checkpoint: Path, mlsp_repo: Path):
    import torch

    encoder_unet = mlsp_repo / "src" / "networks" / "encoder_unet.py"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not encoder_unet.is_file():
        raise FileNotFoundError(encoder_unet)
    spec = importlib.util.spec_from_file_location("mlsp_encoder_unet", encoder_unet)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {encoder_unet}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.EncoderUNetModel(
        n_channels=11,
        encoder_name="resnet101",
        encoder_weights=None,
        decoder_channels=[256, 128, 64, 32, 16],
    )
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {key.removeprefix("_network."): value for key, value in ckpt["state_dict"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.eval()
    return model


def forward_r101_batch(model, tensors: list[np.ndarray]) -> np.ndarray:
    import torch

    x = torch.from_numpy(np.stack(tensors, axis=0))
    with torch.inference_mode():
        features = model.unet.encoder(x)
        decoded = model.unet.decoder(features)
        pred = model.unet.segmentation_head(decoded).squeeze(1)
    return pred.float().numpy() * OUT_NORM_DB


def build_seed_cache(seed: int, samples: pd.DataFrame, model, tag: str, checkpoint: Path) -> None:
    supports = load_supports(samples, seed)
    support_path = seed_support_path(seed)
    if not support_path.exists():
        supports.to_csv(support_path, index=False)

    support_samples = samples.merge(supports, on=["sample_id", "setup", "point"])
    fold_defs = folds(samples)
    point_locations = point_grid_locations()
    freqs = setup_frequencies()
    setup_ids = np.array(sorted(samples["setup"].unique()), dtype=np.int16)
    sample_ids = samples.sort_values("sample_id")["sample_id"].to_numpy(dtype=np.int32)
    pl_by_fold_sample = np.zeros((len(fold_defs), len(sample_ids)), dtype=np.float32)

    for fold_row in fold_defs.itertuples(index=False):
        train_support = support_samples[
            (support_samples["ap_point"] != fold_row.holdout_ap_point) & (support_samples["is_support"] == 1)
        ]
        c_fold = c_from_support(train_support)
        tensors = []
        for setup in setup_ids:
            setup_support = support_samples[(support_samples["setup"] == int(setup)) & (support_samples["is_support"] == 1)]
            tensors.append(
                build_input_tensor(
                    setup=int(setup),
                    samples=samples,
                    setup_support=setup_support,
                    point_locations=point_locations,
                    freq_mhz=freqs[int(setup)],
                    c_dbm=c_fold,
                )
            )

        pred_pl_batch = forward_r101_batch(model, tensors)
        for setup_idx, setup in enumerate(setup_ids):
            setup_rows = samples[samples["setup"] == int(setup)]
            for row in setup_rows.itertuples(index=False):
                grid_row, grid_col = point_locations[int(row.point)]
                pix_y, pix_x = point_to_pixel(grid_row, grid_col)
                pl_by_fold_sample[int(fold_row.fold), int(row.sample_id)] = pred_pl_batch[setup_idx, pix_y, pix_x]
        print(f"cached R101 inference seed {seed}, fold {int(fold_row.fold)}", flush=True)

    np.savez_compressed(
        seed_cache_path(seed, tag),
        support_seed=np.array([seed], dtype=np.int32),
        sample_ids=sample_ids,
        setup_ids=setup_ids,
        pl_by_fold_sample=pl_by_fold_sample,
        checkpoint=str(checkpoint),
    )


def ensure_r101_caches(samples: pd.DataFrame, args: argparse.Namespace, tag: str) -> None:
    missing = [seed for seed in SEEDS if not seed_cache_path(seed, tag).exists()]
    if not args.refresh_cache and not missing:
        return
    import torch

    torch.set_num_threads(max(1, int(args.threads)))
    model = load_r101_model(args.checkpoint, args.mlsp_repo)
    seeds = SEEDS if args.refresh_cache else tuple(missing)
    for seed in seeds:
        build_seed_cache(seed, samples, model, tag, args.checkpoint)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def fit_predict_ridge(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=RIDGE_ALPHA))])
    model.fit(train_x, train_y)
    return model.predict(train_x), model.predict(val_x)


def evaluate_seed(seed: int, samples: pd.DataFrame, tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_path = seed_cache_path(seed, tag)
    if not cache_path.exists():
        raise FileNotFoundError(f"missing cached R101 feature file: {cache_path}")

    sparse_data = add_sparse_features(samples, load_supports(samples, seed)).sort_values("sample_id").reset_index(drop=True)
    cache = np.load(cache_path)
    pl_by_fold_sample = cache["pl_by_fold_sample"]

    metric_rows = []
    pred_rows = []
    for fold_row in folds(samples).itertuples(index=False):
        fold = int(fold_row.fold)
        fold_data = sparse_data.copy()
        fold_data["r101_pl_pred"] = pl_by_fold_sample[fold, fold_data["sample_id"].to_numpy(dtype=int)]
        y = fold_data["rssi"].to_numpy(dtype=np.float32)
        is_val = fold_data["ap_point"].to_numpy() == int(fold_row.holdout_ap_point)
        is_query = fold_data["is_support"].to_numpy() == 0
        train_mask = (~is_val) & is_query
        val_mask = is_val & is_query

        for model_name, feature_cols in FEATURE_SETS.items():
            x = fold_data[feature_cols].to_numpy(dtype=np.float32)
            train_pred, val_pred = fit_predict_ridge(x[train_mask], y[train_mask], x[val_mask])
            train_err = train_pred - y[train_mask]
            val_err = val_pred - y[val_mask]
            metric_rows.append(
                {
                    "support_seed": int(seed),
                    "fold": fold,
                    "holdout_ap_point": int(fold_row.holdout_ap_point),
                    "model": model_name,
                    "feature_count": len(feature_cols),
                    "alpha": RIDGE_ALPHA,
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

            val_rows = fold_data.loc[val_mask, ["sample_id", "setup", "point", "rssi"]]
            for row, y_pred, err in zip(val_rows.itertuples(index=False), val_pred, val_err):
                pred_rows.append(
                    {
                        "support_seed": int(seed),
                        "fold": fold,
                        "holdout_ap_point": int(fold_row.holdout_ap_point),
                        "model": model_name,
                        "feature_count": len(feature_cols),
                        "alpha": RIDGE_ALPHA,
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
                "model": str(model),
                "feature_count": int(metric_group["feature_count"].iloc[0]),
                "alpha": RIDGE_ALPHA,
                "train_rmse_mean_over_16_folds": float(metric_group["train_rmse"].mean()),
                "val_mean_fold_rmse": float(metric_group["val_rmse"].mean()),
                "val_pooled_rmse": rmse(pred_group["y_true"].to_numpy(), pred_group["y_pred"].to_numpy()),
                "val_mae": float(np.mean(np.abs(pred_group["error"]))),
            }
        )
    by_seed = pd.DataFrame(seed_rows)

    overall_rows = []
    for model, group in by_seed.groupby("model"):
        overall_rows.append(
            {
                "model": str(model),
                "feature_count": int(group["feature_count"].iloc[0]),
                "alpha": RIDGE_ALPHA,
                "support_draws": int(len(group)),
                "train_rmse_mean": float(group["train_rmse_mean_over_16_folds"].mean()),
                "train_rmse_std_across_draws": float(group["train_rmse_mean_over_16_folds"].std()),
                "val_rmse_mean": float(group["val_pooled_rmse"].mean()),
                "val_rmse_std_across_draws": float(group["val_pooled_rmse"].std()),
                "val_mae_mean": float(group["val_mae"].mean()),
            }
        )
    overall = pd.DataFrame(overall_rows).sort_values("val_rmse_mean").reset_index(drop=True)
    return by_seed, overall


def main() -> int:
    args = parse_args()
    require_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    samples = load_samples()
    tag = checkpoint_tag(args.checkpoint, args.cache_tag)
    ensure_r101_caches(samples, args, tag)

    metric_frames = []
    prediction_frames = []
    for seed in SEEDS:
        metrics, predictions = evaluate_seed(seed, samples, tag)
        metric_frames.append(metrics)
        prediction_frames.append(predictions)
        print(f"evaluated fixed-alpha ridge feature sets for seed {seed}", flush=True)

    metrics = pd.concat(metric_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    by_seed, overall = summarize(metrics, predictions)

    metrics.to_csv(OUT_DIR / "ridge_fixed_alpha_fold_metrics.csv", index=False)
    predictions.to_csv(OUT_DIR / "ridge_fixed_alpha_predictions.csv", index=False)
    by_seed.to_csv(OUT_DIR / "ridge_fixed_alpha_by_seed_summary.csv", index=False)
    overall.to_csv(OUT_DIR / "ridge_fixed_alpha_overall_summary.csv", index=False)
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "experiment": "comparing_feature_sets_ridge",
                "support_seeds": list(SEEDS),
                "support_count": SUPPORT_COUNT,
                "ridge_alpha": RIDGE_ALPHA,
                "feature_sets": FEATURE_SETS,
                "r101_cache_dir": str(CACHE_DIR),
                "r101_cache_tag": tag,
                "checkpoint": str(args.checkpoint),
                "note": "Fixed alpha; compares only 12 sparse-feature ridge against 13-feature sparse+r101_pl ridge.",
            },
            indent=2,
        )
    )

    print("\nOverall fixed-alpha ridge comparison:")
    print(overall.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
