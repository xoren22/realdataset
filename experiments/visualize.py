"""Render point clouds and RSSI heatmaps from the initialised dataset.

Writes PNGs to ``figs/``. Requires ``scripts/init_data.py`` to have been run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from src.paths import (
    CORRIDOR_PCD,
    FIGS_DIR,
    OFFICE_PCD,
    REGISTERED_PCD,
    RSSI_CSV,
    WIFI_H5,
    require_data,
)


def read_ply(path):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors) if pcd.has_colors() else None
    return xyz, rgb


def render_pcd(path, title, out_png):
    xyz, rgb = read_ply(path)
    print(f"{path.name}: {xyz.shape[0]} pts   bbox {xyz.min(0)} -> {xyz.max(0)}")

    lo = np.percentile(xyz, 0.5, axis=0)
    hi = np.percentile(xyz, 99.5, axis=0)
    mask = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    xyz_c = xyz[mask]
    rgb_c = rgb[mask] if rgb is not None and rgb.size else None
    print(f"  kept {mask.sum()}/{xyz.shape[0]} points in 0.5-99.5 pctile box")

    if rgb_c is None or rgb_c.size == 0:
        c, cmap = xyz_c[:, 2], "viridis"
    else:
        c, cmap = rgb_c, None

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (a, b), name in zip(
        axes, [(0, 1), (0, 2), (1, 2)], ["Top-down (X-Y)", "Side (X-Z)", "Side (Y-Z)"]
    ):
        ax.scatter(xyz_c[:, a], xyz_c[:, b], s=0.3, c=c, cmap=cmap)
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xlabel("XYZ"[a])
        ax.set_ylabel("XYZ"[b])
    fig.suptitle(f"{title}  ({xyz.shape[0]:,} points, showing core 99%)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", out_png)


def rssi_grid_figure(h5_path, out_png):
    with h5py.File(h5_path, "r") as f:
        data = f["data"][:]
        indices = f["indices"][:]
        ap_locations = f["ap_locations"][:]
        setups = f["setup"][:]

    vmin = data[data != 0].min()
    vmax = data[data != 0].max()
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(4, 5, figsize=(22, 16))
    for i, ax in enumerate(axes.flat):
        mat = np.flipud(data[i])
        mat_ind = np.flipud(indices[i])
        ys, xs = np.nonzero(mat)
        zs = mat[ys, xs]
        ax.tricontourf(xs, ys, zs, levels=24, cmap="viridis", norm=norm)
        ax.scatter(xs, ys, s=12, c="red", edgecolors="white", linewidths=0.3)
        ap_idx = ap_locations[i]
        ap_y, ap_x = np.where(mat_ind == ap_idx)
        if ap_y.size:
            ax.scatter(
                ap_x, ap_y, s=120, marker="*", c="yellow",
                edgecolors="black", linewidths=0.7, label=f"AP @ pt {ap_idx}",
            )
            ax.legend(loc="lower right", fontsize=7)
        ax.set_title(f"setup {setups[i]}  (AP pt {ap_idx})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Wi-Fi RSSI across 20 setups  (dBm, viridis)  range [{vmin:.0f}, {vmax:.0f}]",
        fontsize=14,
    )
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
        ax=axes.ravel().tolist(), shrink=0.6, label="RSSI [dBm]",
    )
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_png)


def rssi_layout_figure(h5_path, out_png):
    with h5py.File(h5_path, "r") as f:
        data = f["data"][:]
        indices = f["indices"][:]
        ap_locations = f["ap_locations"][:]

    i = 0
    mat = np.flipud(data[i])
    mat_ind = np.flipud(indices[i])
    fig, ax = plt.subplots(figsize=(12, 10))
    ys, xs = np.nonzero(mat)
    zs = mat[ys, xs]
    cf = ax.tricontourf(xs, ys, zs, levels=32, cmap="viridis")
    ax.scatter(xs, ys, s=40, c="red", edgecolors="white", linewidths=0.5, label="measurement points")
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            if mat[r, c] != 0:
                ax.text(c + 0.25, r + 0.25, int(mat_ind[r, c]), fontsize=8, color="white")
    ap_idx = ap_locations[i]
    ap_y, ap_x = np.where(mat_ind == ap_idx)
    ax.scatter(
        ap_x, ap_y, s=350, marker="*", c="yellow",
        edgecolors="black", linewidths=1.2, label=f"AP @ point {ap_idx}",
    )
    ax.text(5, 24, "OFFICE\n(rows 0-8, cols 0-10)", color="white", ha="center", fontsize=11, fontweight="bold")
    ax.text(13, 17, "CORRIDOR", color="white", ha="center", fontsize=11, fontweight="bold")
    ax.text(25, 8, "ELEVATOR\nHALL", color="white", ha="center", fontsize=11, fontweight="bold")
    plt.colorbar(cf, ax=ax, label="RSSI [dBm]")
    ax.set_title(
        f"Setup 1 — RSSI grid with point enumeration (1..53)\n"
        f"1 m spacing, 3 rooms, AP at position {ap_idx}"
    )
    ax.set_xlabel("grid X (0.5 m per cell)")
    ax.set_ylabel("grid Y (0.5 m per cell)")
    ax.legend()
    ax.set_aspect("equal")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_png)


def main():
    require_data()
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    render_pcd(OFFICE_PCD, "Office LiDAR point cloud", FIGS_DIR / "pcd_office.png")
    render_pcd(CORRIDOR_PCD, "Corridor LiDAR point cloud", FIGS_DIR / "pcd_corridor.png")
    render_pcd(REGISTERED_PCD, "Registered combined point cloud", FIGS_DIR / "pcd_registered.png")

    rssi_grid_figure(WIFI_H5, FIGS_DIR / "rssi_all_setups.png")
    rssi_layout_figure(WIFI_H5, FIGS_DIR / "rssi_layout_setup1.png")
    print("done.")


if __name__ == "__main__":
    main()
