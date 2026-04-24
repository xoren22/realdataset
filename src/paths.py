"""Single source of truth for dataset paths.

All code in this repo reads/writes dataset artifacts through the constants in
this module.  The raw files live under ``DATA_DIR`` and are produced by
``scripts/init_data.py``.  ``DATA_DIR`` is gitignored — nothing outside the
init script should write there, and nothing should read data from anywhere
else.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
FIGS_DIR = REPO_ROOT / "figs"

OFFICE_PCD = DATA_DIR / "office_pcd.ply"
CORRIDOR_PCD = DATA_DIR / "corridor_pcd.ply"
REGISTERED_PCD = DATA_DIR / "registration_example.ply"
RSSI_CSV = DATA_DIR / "RSSI_raw_data.csv"
WIFI_H5 = DATA_DIR / "WiFi_RSSI_data.h5"

ALL_FILES = [OFFICE_PCD, CORRIDOR_PCD, REGISTERED_PCD, RSSI_CSV, WIFI_H5]

ZENODO_RECORD = "15791300"
ZENODO_URL = (
    "https://zenodo.org/records/15791300/files/"
    "A%20Multimodal%20Dataset%20with%203D%20Point%20Cloud%20and%20RSSI%20Measurement.zip"
    "?download=1"
)
ZIP_MD5 = "33ceb074667ef704cae6e51996c83416"

# Mapping from file path inside the Zenodo zip -> canonical name in DATA_DIR.
ZIP_TO_LOCAL = {
    "toolbox/data/office_pcd_20250619_134824.ply": OFFICE_PCD.name,
    "toolbox/data/corridor_pcd_20250619_134851.ply": CORRIDOR_PCD.name,
    "toolbox/data/registration_example.ply": REGISTERED_PCD.name,
    "toolbox/data/RSSI_raw_data.csv": RSSI_CSV.name,
    "toolbox/data/WiFi_RSSI_data2025-06-25_11-32-25.h5": WIFI_H5.name,
}


def require_data():
    """Raise if the dataset is not initialised. Call at the top of any script."""
    missing = [p for p in ALL_FILES if not p.exists()]
    if missing:
        rel = "\n  ".join(str(p.relative_to(REPO_ROOT)) for p in missing)
        raise FileNotFoundError(
            f"Dataset not initialised. Missing:\n  {rel}\n"
            f"Run: python scripts/init_data.py"
        )
