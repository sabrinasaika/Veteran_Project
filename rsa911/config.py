from __future__ import annotations

import os
import sqlite3
from pathlib import Path


DEFAULT_RAW_DIR = Path("~/secure_data/rsa911_raw")
DEFAULT_WORK_DIR = Path("data/rsa911_private")

EXPECTED_FILES = [
    "PY21exiters.csv",
    "PY22exiters.csv",
    "PY23exiters.csv",
    "PY24exiters.csv",
]


def raw_dir() -> Path:
    return Path(os.getenv("RSA911_RAW_DIR", str(DEFAULT_RAW_DIR))).expanduser()


def work_dir() -> Path:
    return Path(os.getenv("RSA911_WORK_DIR", str(DEFAULT_WORK_DIR))).expanduser()


def sqlite_path() -> Path:
    return work_dir() / "rsa911.sqlite3"


def connect_existing(path: Path) -> sqlite3.Connection:
    """Open the built RSA-911 database read-only, or fail clearly if it has not been built.

    A plain sqlite3.connect would silently create an empty database instead.
    """
    path = Path(path).expanduser()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"The RSA-911 database has not been built yet ({path} is missing or empty). "
            "Run: python -m rsa911.build_sqlite --replace"
        )
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def discover_csv_files(src: Path, requested_files: list[str] | None = None) -> list[Path]:
    """Return private CSV files to process without opening their contents."""
    if requested_files:
        return [src / name for name in requested_files]
    return sorted(p for p in src.glob("*.csv") if p.is_file())
