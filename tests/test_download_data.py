"""No network: a fake fetch writes known bytes."""
import hashlib

import pytest

from download_data import ensure_dataset

GOOD = b"tydi"
GOOD_SHA = hashlib.sha256(GOOD).hexdigest()


def fake_fetch(content):
    def fetch(url, dest):
        dest.write_bytes(content)
    return fetch


def test_downloads_and_verifies(tmp_path):
    dest = tmp_path / "raw" / "data.parquet"
    assert ensure_dataset(dest, "u", GOOD_SHA, fake_fetch(GOOD)) == "downloaded"
    assert dest.read_bytes() == GOOD and not dest.with_suffix(".parquet.part").exists()


def test_existing_verified_file_is_not_downloaded_again(tmp_path):
    dest = tmp_path / "data.parquet"
    dest.write_bytes(GOOD)

    def must_not_fetch(url, dest):
        raise AssertionError("should not download")
    assert ensure_dataset(dest, "u", GOOD_SHA, must_not_fetch) == "present"


def test_corrupted_download_is_discarded_and_never_promoted(tmp_path):
    dest = tmp_path / "data.parquet"
    with pytest.raises(SystemExit, match="checksum mismatch"):
        ensure_dataset(dest, "u", GOOD_SHA, fake_fetch(b"truncated"))
    assert not dest.exists() and not dest.with_suffix(".parquet.part").exists()


def test_existing_wrong_file_fails_loudly(tmp_path):
    dest = tmp_path / "data.parquet"
    dest.write_bytes(b"other")
    with pytest.raises(SystemExit, match="doesn't match"):
        ensure_dataset(dest, "u", GOOD_SHA, fake_fetch(GOOD))
