#!/usr/bin/env python
"""Download and unzip the Zenodo 15791300 RF signal dataset into ./data/.

Idempotent — does nothing if every expected file is already present with a
non-zero size. Run from the repo root:

    python scripts/init_data.py

After a successful run, ``data/`` contains:
  * the 5 canonical-name files that ``src.paths`` points at, and
  * the full unzipped Zenodo package under ``data/toolbox/`` (including the
    upstream example notebook), plus the verified zip at ``data/dataset.zip``.
All of ``data/`` is gitignored.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import (
    ALL_FILES,
    DATA_DIR,
    REPO_ROOT,
    ZENODO_URL,
    ZIP_MD5,
    ZIP_TO_LOCAL,
)

ZIP_PATH = DATA_DIR / "dataset.zip"
UNZIP_DIR = DATA_DIR  # zip contains a top-level ``toolbox/`` folder.


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg: str) -> None:
    print(f"[init_data] {msg}", flush=True)


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_canonical_present = all(p.exists() and p.stat().st_size > 0 for p in ALL_FILES)
    unzip_present = (DATA_DIR / "toolbox" / "data").is_dir()
    if all_canonical_present and unzip_present:
        log(f"already initialised — {len(ALL_FILES)} files + unzipped toolbox/ in data/")
        return 0

    if not ZIP_PATH.exists() or md5sum(ZIP_PATH) != ZIP_MD5:
        log("downloading zip from Zenodo record 15791300 ...")
        urllib.request.urlretrieve(ZENODO_URL, ZIP_PATH)
        got = md5sum(ZIP_PATH)
        if got != ZIP_MD5:
            print(f"[init_data] md5 mismatch: expected {ZIP_MD5}, got {got}", file=sys.stderr)
            return 2
        log(f"md5 ok ({got})")
    else:
        log(f"zip already cached at data/{ZIP_PATH.name} (md5 ok)")

    log(f"unzipping into data/ ...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(UNZIP_DIR)
    # Extracted files are read-only by default; make them writable so the dir
    # can be cleaned up with plain ``rm -rf`` later.
    for p in (UNZIP_DIR / "toolbox").rglob("*"):
        p.chmod(p.stat().st_mode | 0o200)

    for zip_rel, local_name in ZIP_TO_LOCAL.items():
        src = UNZIP_DIR / zip_rel
        dst = DATA_DIR / local_name
        if not src.exists():
            print(f"[init_data] missing inside zip: {zip_rel}", file=sys.stderr)
            return 3
        shutil.copy2(src, dst)
        log(f"  -> data/{dst.name}  ({dst.stat().st_size:,} B)")

    log(f"done. canonical files + data/toolbox/ in {DATA_DIR.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
