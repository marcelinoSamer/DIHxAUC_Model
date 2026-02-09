"""
File: data_downloader.py
Description: Downloads dataset from the public GitHub release if not present locally.
Author: FlavorFlow Team

The Hackathon dataset (~670 MB uncompressed) lives on GitHub Releases
so it does not bloat the repository.  This module downloads and unzips
it on first run (or when the data/ directory is missing).

Source:
    https://github.com/ynakhla/DIH-X-AUC-Hackathon/releases/tag/v1.0-data

The "Inventory.Management.zip" asset contains every CSV the pipeline needs.
"""

from __future__ import annotations

import io
import os
import shutil
import zipfile
import logging
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("flavorflow.data_downloader")

# ── Constants ────────────────────────────────────────────────────────────────

RELEASE_ZIP_URL = (
    "https://github.com/ynakhla/DIH-X-AUC-Hackathon/releases/download/"
    "v1.0-data/Inventory.Management.zip"
)

# The zip contains files under "Inventory Management/" sub-folder
_ZIP_INNER_DIR = "Inventory Management"

# Minimum expected CSV files — if fewer exist, re-download
_MIN_CSV_COUNT = 15


def _data_dir(project_root: Optional[Path] = None) -> Path:
    """Resolve the canonical data directory."""
    # data_downloader.py lives at src/utils/ → go up 3 levels to project root
    root = project_root or Path(__file__).resolve().parent.parent.parent
    return root / "data"


def data_is_present(project_root: Optional[Path] = None) -> bool:
    """Return True if the data/ directory looks complete."""
    d = _data_dir(project_root)
    if not d.exists():
        return False
    csv_count = len(list(d.glob("*.csv")))
    return csv_count >= _MIN_CSV_COUNT


def download_data(
    project_root: Optional[Path] = None,
    *,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Download and extract the dataset from GitHub Releases.

    Args:
        project_root: Project root directory (auto-detected if omitted).
        force:        Re-download even if data already present.
        verbose:      Print progress messages.

    Returns:
        Path to the data/ directory.
    """
    dest = _data_dir(project_root)

    if not force and data_is_present(project_root):
        if verbose:
            print(f"✅ Data already present at {dest}  ({len(list(dest.glob('*.csv')))} CSVs)")
        return dest

    dest.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"⬇️  Downloading dataset from GitHub Releases …")
        print(f"   URL: {RELEASE_ZIP_URL}")

    try:
        resp = requests.get(RELEASE_ZIP_URL, stream=True, timeout=300)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Download failed: %s", exc)
        raise RuntimeError(f"Failed to download dataset: {exc}") from exc

    total = int(resp.headers.get("content-length", 0))
    if verbose and total:
        print(f"   Size: {total / 1024 / 1024:.1f} MB")

    # Read into memory then extract
    buf = io.BytesIO()
    downloaded = 0
    for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):  # 8 MB chunks
        buf.write(chunk)
        downloaded += len(chunk)
        if verbose and total:
            pct = downloaded / total * 100
            print(f"\r   Progress: {pct:.0f}%  ({downloaded / 1024 / 1024:.0f} MB)", end="", flush=True)
    if verbose:
        print()  # newline after progress

    if verbose:
        print("📦 Extracting CSV files …")

    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        for member in zf.infolist():
            # Skip directories
            if member.is_dir():
                continue
            # Only extract .csv files
            if not member.filename.lower().endswith(".csv"):
                continue
            # Strip the inner directory prefix → flat into data/
            filename = Path(member.filename).name
            target = dest / filename
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if verbose:
                size_mb = member.file_size / 1024 / 1024
                print(f"   ✅ {filename}  ({size_mb:.1f} MB)")

    csv_count = len(list(dest.glob("*.csv")))
    if verbose:
        print(f"🎉 Done — {csv_count} CSV files in {dest}")

    return dest


# ── CLI usage ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download FlavorFlow dataset")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args()

    download_data(force=args.force)
