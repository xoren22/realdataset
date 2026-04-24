# realdataset

Investigating Zenodo record [15791300](https://zenodo.org/records/15791300)
— *A Multimodal Dataset with 3D Point Cloud and RSSI Measurement* — and
building models on it to see how learnable the Wi-Fi RSSI field is given
the indoor geometry.

See [`CLAUDE.md`](CLAUDE.md) for a full description of the dataset and
repo conventions.

## Setup

```bash
# 1. install deps (numpy, pandas, h5py, matplotlib, open3d)
pip install -r requirements.txt

# 2. fetch the dataset (~10 MB zip from Zenodo, md5-checked, unpacked into ./data/)
python scripts/init_data.py

# 3. sanity-check: regenerate the overview figures into ./figs/
python experiments/visualize.py
```

`scripts/init_data.py` is idempotent — re-running it is a no-op once
`./data/` contains all five expected files.

## Layout

```
src/           importable library (paths, RSSI grid parsing, point-cloud helpers)
scripts/       one-shot commands (data bootstrap, ...)
experiments/   analysis + modelling scripts
data/          raw dataset (gitignored, populated by init_data.py)
figs/          generated figures (gitignored)
```
