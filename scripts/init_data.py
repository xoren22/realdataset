#!/usr/bin/env python
"""Download the Zenodo 15791300 RF signal dataset into ./data/.

Idempotent — does nothing if every expected file is already present with a
non-zero size. Run from the repo root:

    python scripts/init_data.py
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
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

    present = [p for p in ALL_FILES if p.exists() and p.stat().st_size > 0]
    if len(present) == len(ALL_FILES):
        log(f"all {len(ALL_FILES)} files already present in {DATA_DIR.relative_to(REPO_ROOT)}/")
        return 0
    missing = [p.name for p in ALL_FILES if p not in present]
    log(f"missing {len(missing)} file(s): {missing}")

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        zip_path = tmp / "dataset.zip"

        log(f"downloading from Zenodo record 15791300 ...")
        urllib.request.urlretrieve(ZENODO_URL, zip_path)
        got = md5sum(zip_path)
        if got != ZIP_MD5:
            print(
                f"[init_data] md5 mismatch: expected {ZIP_MD5}, got {got}",
                file=sys.stderr,
            )
            return 2
        log(f"md5 ok ({got})")

        extract_dir = tmp / "unzip"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

        for zip_rel, local_name in ZIP_TO_LOCAL.items():
            src = extract_dir / zip_rel
            dst = DATA_DIR / local_name
            if not src.exists():
                print(f"[init_data] missing inside zip: {zip_rel}", file=sys.stderr)
                return 3
            shutil.copy2(src, dst)
            log(f"  -> data/{dst.name}  ({dst.stat().st_size:,} B)")

    log(f"done. {len(ALL_FILES)} files in {DATA_DIR.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
