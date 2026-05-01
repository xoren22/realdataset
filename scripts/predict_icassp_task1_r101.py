"""Run the r101 MLSP checkpoint on ICASSP 2025 Task 1 validation and plot
ground-truth vs predicted pathloss with pooled R^2 / RMSE plus a per-sample
RMSE histogram.

Per-sample RMSE := RMSE computed inside the masked pixels of one map (one
value per validation sample). Pooled metrics are computed over the union of
masked pixels of all validation samples.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MLSP_REPO = Path("/home/kpetrosyan/mlsp_wair_d")
CKPT_PATH = Path(
    "/mnt/weka/kpetrosyan/synthetic_pretraining_checkpoints/"
    "r101_bs181_lr3e-03_wr0.1_dr0.1_wd1e-03/2026-04-14_09-58-12.473943/every/"
    "step_00350000_every.ckpt"
)
VAL_MANIFEST = Path(
    "/data/indoor/icassp2025/manifests/icassp_val_Task_1_ICASSP_21_22_23_24_25.csv"
)
OUT_PNG = Path("/home/kpetrosyan/icassp_task1_r101_pathloss_scatter.png")

OUT_NORM = 160.0
SCATTER_N = 200_000
SEED = 0


def main() -> None:
    sys.path.insert(0, str(MLSP_REPO))
    from src.datamodules.indoor import IndoorDatamodule
    from src.datamodules.datasets import PathlossDataset
    from src.networks.encoder_unet import EncoderUNetModel

    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a GPU; run on gpu01..gpu08.")
    device = torch.device("cuda")

    inputs_list = IndoorDatamodule.get_inputs_list(str(VAL_MANIFEST))
    print(f"[data] {len(inputs_list)} val samples from {VAL_MANIFEST.name}")

    dataset = PathlossDataset(
        inputs_list,
        training=False,
        inference=True,
        augmentations=None,
        sparse_range=[0.0, 0.0],
        modality_dropout_prob=0.0,
        sparse_dropout_given_dropout=1.0,
        force_drop_sparse=True,
        force_drop_trans_ref=False,
    )

    model = (
        EncoderUNetModel(n_channels=11, encoder_name="resnet101", encoder_weights=None)
        .to(device)
        .eval()
    )
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state = {k.removeprefix("_network."): v for k, v in ckpt["state_dict"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")

    rng = np.random.default_rng(SEED)
    keep_per_sample = max(1, SCATTER_N // max(1, len(dataset)))

    per_sample_rmse: list[float] = []
    scatter_gt: list[np.ndarray] = []
    scatter_pred: list[np.ndarray] = []
    sum_se = 0.0
    sum_n = 0
    sum_y = 0.0
    sum_y2 = 0.0

    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(len(dataset)):
            input_tensor, target_tensor, mask, meta = dataset[i]
            x = input_tensor.unsqueeze(0).to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(x)
            orig_h = int(meta["orig_h"])
            orig_w = int(meta["orig_w"])
            pred_full = (
                F.interpolate(
                    pred.float(),
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
                .cpu()
                .numpy()
            )
            target_np = target_tensor.cpu().numpy()
            mask_np = mask.cpu().numpy() > 0

            pred_db = pred_full * OUT_NORM
            target_db = target_np * OUT_NORM

            valid = mask_np
            n_valid = int(valid.sum())
            if n_valid == 0:
                continue
            err = pred_db[valid] - target_db[valid]
            mse_i = float((err ** 2).mean())
            per_sample_rmse.append(float(np.sqrt(mse_i)))

            sum_se += float((err ** 2).sum())
            sum_n += n_valid
            sum_y += float(target_db[valid].sum())
            sum_y2 += float((target_db[valid] ** 2).sum())

            gt = target_db[valid]
            pr = pred_db[valid]
            if n_valid > keep_per_sample:
                idx = rng.choice(n_valid, size=keep_per_sample, replace=False)
                gt = gt[idx]
                pr = pr[idx]
            scatter_gt.append(gt)
            scatter_pred.append(pr)

            if (i + 1) % 25 == 0 or i == len(dataset) - 1:
                dt = time.perf_counter() - t0
                print(f"[forward] {i + 1}/{len(dataset)} ({dt:.1f}s)")

    rmse = float(np.sqrt(sum_se / sum_n))
    mean_y = sum_y / sum_n
    ss_tot = sum_y2 - sum_n * mean_y ** 2
    r2 = float(1.0 - sum_se / ss_tot)
    psr = np.asarray(per_sample_rmse, dtype=np.float64)

    print(f"[metrics] pooled RMSE={rmse:.3f} dB  R^2={r2:.4f}  N_pixels={sum_n}")
    print(
        f"[per-sample RMSE] mean={psr.mean():.3f}  median={np.median(psr):.3f}  "
        f"std={psr.std(ddof=0):.3f}  min={psr.min():.3f}  max={psr.max():.3f}"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_all = np.concatenate(scatter_gt)
    pr_all = np.concatenate(scatter_pred)
    if gt_all.size > SCATTER_N:
        idx = rng.choice(gt_all.size, size=SCATTER_N, replace=False)
        gt_all = gt_all[idx]
        pr_all = pr_all[idx]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    lo = float(min(gt_all.min(), pr_all.min()))
    hi = float(max(gt_all.max(), pr_all.max()))
    hb = ax.hexbin(gt_all, pr_all, gridsize=80, cmap="viridis", bins="log", mincnt=1)
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.0, label="y = x")
    ax.set_xlabel("Ground truth pathloss [dB]")
    ax.set_ylabel("Predicted pathloss [dB]")
    ax.set_title(
        f"r101 MLSP -> ICASSP Task 1 val\n"
        f"pooled RMSE = {rmse:.2f} dB   R^2 = {r2:.4f}   "
        f"N_samples = {len(psr)}   N_pixels = {sum_n:,}"
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left")
    fig.colorbar(hb, ax=ax, label="log10(count)")

    ax = axes[1]
    ax.hist(psr, bins=30, color="#3a7ca5", edgecolor="black")
    ax.axvline(psr.mean(), color="red", linestyle="--", linewidth=1, label=f"mean = {psr.mean():.2f} dB")
    ax.axvline(np.median(psr), color="orange", linestyle="--", linewidth=1, label=f"median = {np.median(psr):.2f} dB")
    ax.set_xlabel("Per-sample RMSE [dB]")
    ax.set_ylabel("# samples")
    ax.set_title("Per-sample RMSE distribution")
    ax.legend()

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    print(f"[plot] saved {OUT_PNG}")


if __name__ == "__main__":
    main()
