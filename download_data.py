"""Download the TyDi QA dev file and verify it (docs/decisions/0001).

    uv run python download_data.py

Writes to a .part file first and only moves it into place after the SHA-256 matches, so an
interrupted or corrupted download can never be mistaken for the real file. The samples in
samples/samples.json were drawn from exactly this file (same checksum).
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path
from typing import Callable

URL = ("https://huggingface.co/datasets/google-research-datasets/tydiqa/resolve/main/"
       "primary_task/validation-00000-of-00001.parquet")
SHA256 = "2e9557a6ea4bfd65f381c64b91122c90554e4b00f86e15a8b4f1353be1c81510"
SIZE = 232_035_124
DEST = Path("data/raw/tydiqa-primary-validation.parquet")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def urllib_fetch(url: str, dest: Path) -> None:
    urllib.request.urlretrieve(url, dest)


def ensure_dataset(dest: Path = DEST, url: str = URL, expected: str = SHA256,
                   fetch: Callable[[str, Path], None] = urllib_fetch) -> str:
    """Return 'present' if a verified file already exists, else download and verify it."""
    if dest.exists():
        if sha256(dest) == expected:
            return "present"
        raise SystemExit(f"{dest} exists but its checksum doesn't match; delete it and rerun")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    fetch(url, part)
    got = sha256(part)
    if got != expected:
        part.unlink()
        raise SystemExit(f"checksum mismatch: expected {expected}, got {got}; download discarded")
    part.replace(dest)
    return "downloaded"


if __name__ == "__main__":
    print(f"Downloading ~{SIZE / 1e6:.0f} MB to {DEST} if needed...", file=sys.stderr)
    print(ensure_dataset())
