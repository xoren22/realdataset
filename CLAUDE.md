# realdataset — indoor RF-signal dataset investigation

## What this repo is
A working area for investigating **Zenodo record 15791300** — the
*"Multimodal Dataset with 3D Point Cloud and RSSI Measurement"* — and
building models on it to see how well the Wi-Fi RSSI field can be
predicted from the indoor geometry (and vice versa). Expect exploratory
analysis, baseline models, and benchmarking of learning approaches.

## The dataset in one screen
- One indoor venue composed of three connected rooms: **office +
  corridor + elevator hall** (roughly an L shape, ~14 × 15 m).
- **LiDAR point clouds** — three `.ply` files:
  - `office_pcd.ply` (~149 k pts, coords in **meters**).
  - `corridor_pcd.ply` (~164 k pts, coords in **millimetres** — note the
    unit mismatch vs the office scan).
  - `registration_example.ply` (~313 k pts) — the two above after manual
    ICP registration into a single frame; office points painted orange,
    corridor points blue.
- **Wi-Fi RSSI** from one TP-Link router (BSSID `50:d4:f7:be:44:62`, 2.4 GHz,
  802.11n, 40 MHz):
  - 53 measurement points per setup at 1 m spacing, snake-walked through
    the office (points 1–30), corridor (31–45), elevator hall (46–53).
  - **20 setups** × 53 points ≈ 1 027 raw scan rows. Each setup places
    the router at a different one of the 53 anchor points. Setups 13–16
    revisit AP positions 1 / 18 / 26 / 30 from setups 1–6 to produce the
    "people present" counterpart of the "empty room" baselines.
  - Stored two ways: `RSSI_raw_data.csv` (full 802.11 metadata per scan)
    and `WiFi_RSSI_data.h5` (dense `(20, 28, 30)` dBm tensor on a
    0.5 m grid, with companion `indices`, `setup`, `ap_locations`).
  - Observed RSSI range: **−82 dBm (shadowed) to −19 dBm (on AP)**.

## Repo layout
```
src/             importable helpers — all paths come from src/paths.py
  paths.py       single source of truth for file locations (DATA_DIR, filenames, Zenodo URL)
  rssi_utils.py  parse CSV, build the 28x30 RSSI grid (from upstream)
  pcd_utils.py   Open3D helpers: manual ICP registration (from upstream)
scripts/
  init_data.py   download the Zenodo zip, verify md5, place files in ./data/
experiments/
  visualize.py   regenerate the figures under figs/
```

## Data location contract
**All code reads raw files from `./data/` via `src.paths`.** Nothing
else writes there; nothing reads raw data from anywhere else. The
`data/`, `figs/`, and any `*.zip` are gitignored.

Bootstrap (idempotent — safe to re-run):

```
/home/kpetrosyan/miniconda3/envs/c/bin/python scripts/init_data.py
```

After running you get exactly these five files under `data/`:

```
office_pcd.ply
corridor_pcd.ply
registration_example.ply
RSSI_raw_data.csv
WiFi_RSSI_data.h5
```

Any script that needs the dataset should call `src.paths.require_data()`
at the top — it raises a clear error if the data has not been
initialised.

## Conventions for future work
- Always use `/home/kpetrosyan/miniconda3/envs/c/bin/python`.
- This machine is the **root SLURM node with no GPUs**. Run training /
  GPU profiling on `gpu01`..`gpu08` via ssh or SLURM, never here.
- New artifacts (figures, model checkpoints, cached tensors) go in
  gitignored directories (`figs/`, `models/`, `runs/`, ...). Only source
  code is committed.
- When adding new scripts, reuse `src.paths` rather than hard-coding
  paths, so there remains exactly one place to change a filename.
