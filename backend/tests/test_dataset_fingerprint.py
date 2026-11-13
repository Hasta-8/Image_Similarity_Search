"""Tests for deterministic dataset discovery and fingerprinting."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import dataset_fingerprint
from dataset_fingerprint import (
    DatasetFingerprintError,
    compute_dataset_fingerprint,
    discover_dataset_images,
    sha256_file,
)


def _write(root: Path, relative_path: str, payload: bytes) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_discover_returns_sorted_supported_images(tmp_path):
    _write(tmp_path, "z.PNG", b"z")
    _write(tmp_path, "nested/b.jpeg", b"b")
    _write(tmp_path, "a.jpg", b"a")
    _write(tmp_path, "ignored.gif", b"g")
    _write(tmp_path, "ignored.txt", b"t")

    discovered = discover_dataset_images(tmp_path)

    assert [relative for relative, _ in discovered] == [
        "a.jpg",
        "nested/b.jpeg",
        "z.PNG",
    ]
    assert all(Path(absolute).is_file() for _, absolute in discovered)


def test_discover_rejects_missing_or_non_directory(tmp_path):
    with pytest.raises(DatasetFingerprintError, match="not found"):
        discover_dataset_images(tmp_path / "missing")

    file_path = tmp_path / "file"
    file_path.write_text("not a directory", encoding="utf-8")
    with pytest.raises(DatasetFingerprintError, match="not a directory"):
        discover_dataset_images(file_path)


def test_sha256_file_hashes_raw_bytes(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == hashlib.sha256(b"abc").hexdigest()


def test_fingerprint_is_order_independent_and_deterministic(tmp_path):
    _write(tmp_path, "a.jpg", b"alpha")
    _write(tmp_path, "nested/b.png", b"beta")

    first = compute_dataset_fingerprint(tmp_path, ["nested/b.png", "a.jpg"])
    second = compute_dataset_fingerprint(tmp_path, ["a.jpg", "nested/b.png"])

    assert first.algorithm == "sha256-path-and-content-v1"
    assert first.fingerprint == second.fingerprint
    assert first.file_count == 2
    assert first.total_bytes == len(b"alpha") + len(b"beta")
    assert set(first.content_hashes) == {"a.jpg", "nested/b.png"}
    assert first.unreadable_files == {}


def test_fingerprint_changes_with_content_or_path(tmp_path):
    _write(tmp_path, "a.jpg", b"alpha")
    original = compute_dataset_fingerprint(tmp_path, ["a.jpg"]).fingerprint

    _write(tmp_path, "a.jpg", b"alpha changed")
    changed = compute_dataset_fingerprint(tmp_path, ["a.jpg"]).fingerprint
    assert changed != original

    _write(tmp_path, "renamed.jpg", b"alpha changed")
    renamed = compute_dataset_fingerprint(tmp_path, ["renamed.jpg"]).fingerprint
    assert renamed != changed


def test_fingerprint_records_unreadable_files_and_excludes_their_bytes(tmp_path, monkeypatch):
    readable = _write(tmp_path, "readable.jpg", b"readable")
    unreadable = _write(tmp_path, "unreadable.jpg", b"unreadable")
    real_sha256_file = dataset_fingerprint.sha256_file

    def failing_sha256(path, chunk_size=dataset_fingerprint.HASH_CHUNK_SIZE):
        if Path(path).name == "unreadable.jpg":
            raise OSError("permission denied")
        return real_sha256_file(path, chunk_size=chunk_size)

    monkeypatch.setattr(dataset_fingerprint, "sha256_file", failing_sha256)

    fingerprint = compute_dataset_fingerprint(
        tmp_path, ["readable.jpg", "unreadable.jpg"]
    )

    assert fingerprint.file_count == 2
    assert set(fingerprint.content_hashes) == {"readable.jpg"}
    assert fingerprint.unreadable_files == {"unreadable.jpg": "OSError"}
    assert fingerprint.total_bytes == readable.stat().st_size
    assert unreadable.stat().st_size > 0

    changed_marker = compute_dataset_fingerprint(
        tmp_path, ["readable.jpg", "unreadable.jpg", "new.jpg"]
    ).fingerprint
    assert changed_marker != fingerprint.fingerprint


def test_fingerprint_rejects_duplicate_paths(tmp_path):
    with pytest.raises(DatasetFingerprintError, match="duplicate"):
        compute_dataset_fingerprint(tmp_path, ["a.jpg", "./a.jpg"])


def test_fingerprint_rejects_invalid_relative_paths(tmp_path):
    with pytest.raises(ValueError):
        compute_dataset_fingerprint(tmp_path, ["../outside.jpg"])
