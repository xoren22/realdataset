#!/usr/bin/env python
"""LOSO cross-validated RSSI baselines on Zenodo 15791300.

Run from repo root:
    /home/kpetrosyan/miniconda3/envs/c/bin/python experiments/baselines.py
"""
from __future__ import annotations

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

from src.paths import REPO_ROOT, WIFI_H5, require_data

SEED = 0
CELL_M = 0.5
RUNS = REPO_ROOT / "runs"
MODELS_DIR = REPO_ROOT / "models"
FIGS = REPO_ROOT / "figs"
REPORT = REPO_ROOT / "report.md"

NUM_COLS = ["ap_x_m", "ap_y_m", "mx_m", "my_m", "dist_m", "log_d", "same_room"]
CAT_COLS = ["room_meas", "room_ap"]


def point_to_room(p: int) -> str:
    return "office" if p <= 30 else "corridor" if p <= 45 else "elevator"


def load_samples(h5_path: Path) -> pd.DataFrame:
    with h5py.File(h5_path, "r") as f:
        data, idx_all, ap_loc = f["data"][:], f["indices"][:], f["ap_locations"][:]
    rows = []
    for s in range(20):
        rssi, idx = data[s], idx_all[s]
        ap_pt = int(ap_loc[s])
        ap_y, ap_x = np.where(idx == ap_pt)
        assert ap_y.size == 1, f"AP position ambiguous for setup {s + 1}"
        apy, apx = int(ap_y[0]), int(ap_x[0])
        ys, xs = np.nonzero(rssi)
        for y, x in zip(ys, xs):
            p = int(idx[y, x])
            rows.append(dict(
                setup=s + 1, point=p,
                ap_x_m=apx * CELL_M, ap_y_m=apy * CELL_M,
                mx_m=int(x) * CELL_M, my_m=int(y) * CELL_M,
                room_meas=point_to_room(p), room_ap=point_to_room(ap_pt),
                rssi=float(rssi[y, x]),
            ))
    df = pd.DataFrame(rows)
    df["dist_m"] = np.hypot(df.mx_m - df.ap_x_m, df.my_m - df.ap_y_m)
    # Half-cell floor so log10(0) can't bite when measurement point == AP.
    df["log_d"] = np.log10(df.dist_m.clip(lower=CELL_M / 2))
    df["same_room"] = (df.room_meas == df.room_ap).astype(int)
    return df


def _pre() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), CAT_COLS),
    ])


def build_models() -> dict[str, object]:
    return {
        "mean": DummyRegressor(strategy="mean"),
        "logdistance": Pipeline([
            ("pre", ColumnTransformer([("keep", "passthrough", ["log_d"])])),
            ("lr", LinearRegression()),
        ]),
        "linear": Pipeline([("pre", _pre()), ("lr", LinearRegression())]),
        "ridge": Pipeline([("pre", _pre()), ("ridge", Ridge(alpha=1.0))]),
        "knn": Pipeline([("pre", _pre()), ("knn", KNeighborsRegressor(n_neighbors=5))]),
        "rf": Pipeline([("pre", _pre()), ("rf", RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=SEED, n_jobs=-1))]),
        "gbr": Pipeline([("pre", _pre()), ("gbr", HistGradientBoostingRegressor(
            random_state=SEED))]),
    }


def run_cv(models, X, y, groups) -> tuple[pd.DataFrame, pd.DataFrame]:
    logo = LeaveOneGroupOut()
    metric_rows, pred_frames = [], []
    for name, template in models.items():
        for fold, (tr, va) in enumerate(logo.split(X, y, groups)):
            m = clone(template)
            m.fit(X.iloc[tr], y[tr])
            holdout = int(groups[va][0])
            for split, idx in [("train", tr), ("val", va)]:
                yt, yh = y[idx], m.predict(X.iloc[idx])
                err = yh - yt
                metric_rows.append(dict(
                    model=name, fold=fold, holdout_setup=holdout, split=split,
                    n=len(idx),
                    rmse=float(np.sqrt((err ** 2).mean())),
                    mae=float(np.abs(err).mean()),
                    r2=float(r2_score(yt, yh)) if np.var(yt) > 0 else np.nan,
                ))
            pred_frames.append(pd.DataFrame(dict(
                model=name, fold=fold, holdout_setup=holdout,
                sample_idx=va, y_true=y[va], y_hat=m.predict(X.iloc[va]),
            )))
        pooled = pd.concat(
            [pf for pf in pred_frames if pf["model"].iloc[0] == name],
            ignore_index=True,
        )
        rmse = float(np.sqrt(((pooled.y_hat - pooled.y_true) ** 2).mean()))
        print(f"  {name:<12s}  pooled val RMSE = {rmse:6.3f} dBm", flush=True)
    return pd.DataFrame(metric_rows), pd.concat(pred_frames, ignore_index=True)


def summarise(metrics: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g_val in metrics[metrics.split == "val"].groupby("model"):
        g_tr = metrics[(metrics.model == name) & (metrics.split == "train")]
        p = preds[preds.model == name]
        err = (p.y_hat - p.y_true).to_numpy()
        rows.append(dict(
            model=name,
            pooled_rmse=float(np.sqrt((err ** 2).mean())),
            pooled_mae=float(np.abs(err).mean()),
            mean_fold_rmse=float(g_val.rmse.mean()),
            std_fold_rmse=float(g_val.rmse.std()),
            mean_fold_r2=float(g_val.r2.mean()),
            train_rmse=float(g_tr.rmse.mean()),
        ))
    return (pd.DataFrame(rows)
            .sort_values("pooled_rmse").reset_index(drop=True))


def sanity(df, metrics, preds, summary):
    assert len(df) == 1060, f"expected 1060 samples, got {len(df)}"
    assert (df.groupby("setup").size() == 53).all()
    folds = metrics[["model", "fold", "holdout_setup"]].drop_duplicates()
    per_model_folds = folds.groupby("model").fold.nunique()
    assert (per_model_folds == 20).all()
    assert folds.holdout_setup.nunique() == 20
    for _, r in summary.iterrows():
        p = preds[preds.model == r.model]
        rederived = float(np.sqrt(((p.y_hat - p.y_true) ** 2).mean()))
        assert abs(rederived - r.pooled_rmse) < 1e-9, r.model
    mean_rmse = float(summary.set_index("model").loc["mean", "pooled_rmse"])
    assert abs(mean_rmse - float(np.std(df.rssi))) < 2.0


# ---------- plots ----------
ROOM_COLORS = {"office": "C0", "corridor": "C1", "elevator": "C2"}


def plot_target(df, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for r, c in ROOM_COLORS.items():
        ax.hist(df.loc[df.room_meas == r, "rssi"], bins=30, alpha=0.55,
                color=c, label=r)
    ax.set_xlabel("RSSI [dBm]"); ax.set_ylabel("count"); ax.legend()
    ax.set_title("RSSI distribution by measurement room")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_rssi_vs_logd(df, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for r, c in ROOM_COLORS.items():
        sub = df[df.room_meas == r]
        ax.scatter(sub.log_d, sub.rssi, s=10, alpha=0.4, color=c, label=r)
    A = np.column_stack([np.ones(len(df)), df.log_d])
    (a, b), *_ = np.linalg.lstsq(A, df.rssi, rcond=None)
    xs = np.linspace(df.log_d.min(), df.log_d.max(), 100)
    ax.plot(xs, a + b * xs, "k--",
            label=f"fit: RSSI = {a:.1f} + {b:.1f}·log10(d)")
    ax.set_xlabel("log10(distance [m])"); ax.set_ylabel("RSSI [dBm]")
    ax.legend(); ax.set_title("RSSI vs log-distance, all samples")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_cv_bars(summary, path):
    s = summary.sort_values("pooled_rmse")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(s.model, s.pooled_rmse, yerr=s.std_fold_rmse, capsize=4,
           color="C0", alpha=0.85)
    for i, (v, sd) in enumerate(zip(s.pooled_rmse, s.std_fold_rmse)):
        ax.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("pooled val RMSE [dBm]")
    ax.set_title("LOSO-CV val RMSE  (error bars = std across 20 folds)")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_pred_vs_true(preds, best, path):
    p = preds[preds.model == best]
    lo, hi = float(p.y_true.min()) - 2, float(p.y_true.max()) + 2
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.scatter(p.y_true, p.y_hat, s=10, alpha=0.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("true RSSI [dBm]"); ax.set_ylabel("predicted RSSI [dBm]")
    ax.set_title(f"{best}: pooled val predictions (20 LOSO folds)")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def pick_median_fold(metrics, best):
    val = metrics[(metrics.model == best) & (metrics.split == "val")].sort_values("rmse")
    return int(val.iloc[len(val) // 2].holdout_setup)


def plot_residual_map(df, preds, best, holdout_setup, path):
    p = (preds[(preds.model == best) & (preds.holdout_setup == holdout_setup)]
         .sort_values("sample_idx"))
    sub = df.iloc[p.sample_idx.to_numpy()].copy()
    sub["y_hat"] = p.y_hat.to_numpy()
    sub["resid"] = sub.rssi - sub.y_hat
    gt = np.full((28, 30), np.nan)
    gp = np.full_like(gt, np.nan)
    gr = np.full_like(gt, np.nan)
    for _, row in sub.iterrows():
        r, c = int(row.my_m / CELL_M), int(row.mx_m / CELL_M)
        gt[r, c], gp[r, c], gr[r, c] = row.rssi, row.y_hat, row.resid
    vabs = float(np.nanmax(np.abs(gr)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, g, title, cmap, kw in [
        (axes[0], gt, "true RSSI", "viridis", {}),
        (axes[1], gp, "predicted RSSI", "viridis", {}),
        (axes[2], gr, "residual (true − pred)", "RdBu", {"vmin": -vabs, "vmax": vabs}),
    ]:
        im = ax.imshow(np.flipud(g), cmap=cmap, **kw)
        fig.colorbar(im, ax=ax, shrink=0.7, label="dBm")
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{best} — held-out setup {holdout_setup}")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ---------- report ----------
def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(df, summary, figs_used):
    best = summary.iloc[0].model
    best_rmse = summary.iloc[0].pooled_rmse
    table = md_table(summary)
    txt = f"""# RSSI baselines on Zenodo 15791300

Auto-generated by `experiments/baselines.py`.

- **samples**: {len(df):,} (20 setups × 53 points)
- **features**: `{', '.join(NUM_COLS + CAT_COLS)}`
- **target**: RSSI in dBm, observed range {df.rssi.min():.0f} to {df.rssi.max():.0f}
- **CV**: Leave-One-Setup-Out, k = 20; `StandardScaler` fit per fold
- **seed**: {SEED}

## Dataset

![RSSI distribution by measurement room]({figs_used['target']})

![RSSI vs log10(distance) with overall path-loss fit]({figs_used['logd']})

## Cross-validated results

Ranking metric is **pooled val RMSE** — sqrt(mean of squared errors over all
1 060 val predictions concatenated across the 20 folds. `mean_fold_rmse ±
std_fold_rmse` report the per-fold distribution (equal-weight average of the
20 per-fold RMSEs). `train_rmse` is the average in-fold training RMSE and
exposes under/overfitting.

{table}

![Pooled val RMSE per model]({figs_used['cv_bars']})

Best model: **{best}** at pooled val RMSE **{best_rmse:.2f} dBm** over
{len(df)} held-out predictions.

![Predicted vs true, best model pooled across folds]({figs_used['pred']})

![True / predicted / residual maps on one held-out setup]({figs_used['resid']})

## Caveats

- The 20 AP positions include 4 that appear in two setups each (the
  'with-people' / 'no-people' pair design: setups 1↔13, 4↔14, 5↔16, 6↔15).
  Those four held-out folds still have their AP anchor *location* present
  in training (different occupancy). This is a property of the dataset,
  not a bug in the split.
- Coordinates are grid-based (0.5 m cells). The LiDAR point clouds are not
  used here — an obvious next step is to add geometric features derived from
  `registration_example.ply` (wall intersections, through-wall path count,
  room volumes) to the same pipeline.
- No hyperparameter search was done. The point of this run is a fair,
  reproducible floor; tuning comes later.

## Artifacts

- `runs/baselines_cv.csv`   — long-format per-fold metrics
- `runs/baselines_summary.csv` — this table
- `models/<name>.joblib`    — each baseline refit on all 1 060 samples
- `figs/fig_*.png`          — the figures above
"""
    REPORT.write_text(txt)


def main():
    require_data()
    np.random.seed(SEED)
    for d in (RUNS, MODELS_DIR, FIGS):
        d.mkdir(parents=True, exist_ok=True)

    df = load_samples(WIFI_H5)
    X = df[NUM_COLS + CAT_COLS]
    y = df["rssi"].to_numpy()
    groups = df["setup"].to_numpy()

    models = build_models()
    print(f"running LOSO-CV (k=20) over {len(models)} baselines on {len(df)} samples:")
    metrics, preds = run_cv(models, X, y, groups)
    summary = summarise(metrics, preds)
    sanity(df, metrics, preds, summary)

    metrics.to_csv(RUNS / "baselines_cv.csv", index=False)
    summary.to_csv(RUNS / "baselines_summary.csv", index=False)

    best = summary.iloc[0].model
    holdout = pick_median_fold(metrics, best)
    figs_used = {
        "target": "figs/fig_target_hist.png",
        "logd": "figs/fig_rssi_vs_logd.png",
        "cv_bars": "figs/fig_cv_rmse_bars.png",
        "pred": "figs/fig_pred_vs_true.png",
        "resid": "figs/fig_residual_map.png",
    }
    plot_target(df, FIGS / "fig_target_hist.png")
    plot_rssi_vs_logd(df, FIGS / "fig_rssi_vs_logd.png")
    plot_cv_bars(summary, FIGS / "fig_cv_rmse_bars.png")
    plot_pred_vs_true(preds, best, FIGS / "fig_pred_vs_true.png")
    plot_residual_map(df, preds, best, holdout, FIGS / "fig_residual_map.png")

    for name, template in models.items():
        m = clone(template); m.fit(X, y)
        joblib.dump(m, MODELS_DIR / f"{name}.joblib")

    write_report(df, summary, figs_used)
    print(f"\nbest: {best}  (pooled val RMSE {summary.iloc[0].pooled_rmse:.3f} dBm)")
    print(f"wrote {REPORT.relative_to(REPO_ROOT)}, runs/, models/, figs/")


if __name__ == "__main__":
    main()
