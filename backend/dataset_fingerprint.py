"""Dataset discovery and deterministic content fingerprinting."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from stable_ids import (
    StableIDError,
    dataset_relative_path,
    normalize_dataset_relative_path,
)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
FINGERPRINT_ALGORITHM = "sha256-path-and-content-v1"
FINGERPRINT_DOMAIN = b"imgsimisearch:dataset-fingerprint:v1\x00"
HASH_CHUNK_SIZE = 1024 * 1024


class DatasetFingerprintError(ValueError):
    """Raised when dataset discovery or fingerprinting cannot proceed."""


@dataclass
class DatasetFingerprint:
    algorithm: str
    fingerprint: str
    file_count: int
    total_bytes: int
    content_hashes: Dict[str, str]
    unreadable_files: Dict[str, str]


def discover_dataset_images(dataset_dir: str | os.PathLike) -> List[Tuple[str, str]]:
    """Return sorted ``(relative_path, absolute_path)`` supported image pairs."""
    root = Path(dataset_dir).expanduser().resolve()
    if not root.exists():
        raise DatasetFingerprintError(f"Dataset directory not found: {root}")
    if not root.is_dir():
        raise DatasetFingerprintError(f"Dataset path is not a directory: {root}")

    discovered: List[Tuple[str, str]] = []
    for current_dir, dir_names, file_names in os.walk(root, followlinks=False):
        dir_names.sort()
        for file_name in sorted(file_names):
            if Path(file_name).suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            absolute_path = Path(current_dir) / file_name
            try:
                relative_path = dataset_relative_path(root, absolute_path)
            except StableIDError as exc:
                raise DatasetFingerprintError(str(exc)) from exc
            discovered.append((relative_path, str(absolute_path)))

    discovered.sort(key=lambda item: item[0])
    return discovered


def sha256_file(path: str | os.PathLike, chunk_size: int = HASH_CHUNK_SIZE) -> str:
    """Return the SHA-256 hex digest of a file's raw bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def compute_dataset_fingerprint(
    dataset_dir: str | os.PathLike,
    relative_paths: Iterable[str],
) -> DatasetFingerprint:
    """Fingerprint sorted relative paths plus their raw file content hashes.

    Unreadable files are represented by a marker in the fingerprint stream so
    additions/removals still change the fingerprint. Content changes in a file
    that remains unreadable cannot be observed by this method.
    """
    normalized_paths = [normalize_dataset_relative_path(path) for path in relative_paths]
    if len(set(normalized_paths)) != len(normalized_paths):
        raise DatasetFingerprintError("Dataset contains duplicate relative paths")

    root = Path(dataset_dir).expanduser().resolve()
    content_hashes: Dict[str, str] = {}
    unreadable_files: Dict[str, str] = {}
    total_bytes = 0

    for relative_path in sorted(normalized_paths):
        absolute_path = root / relative_path
        try:
            file_size = os.path.getsize(absolute_path)
            content_hash = sha256_file(absolute_path)
        except OSError as exc:
            content_hashes.pop(relative_path, None)
            unreadable_files[relative_path] = type(exc).__name__
        else:
            content_hashes[relative_path] = content_hash
            total_bytes += file_size

    digest = hashlib.sha256()
    digest.update(FINGERPRINT_DOMAIN)
    digest.update(len(normalized_paths).to_bytes(8, "big"))

    for relative_path in sorted(normalized_paths):
        encoded_path = relative_path.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        if relative_path in content_hashes:
            digest.update(b"\x01")
            digest.update(bytes.fromhex(content_hashes[relative_path]))
        else:
            digest.update(b"\x00")

    return DatasetFingerprint(
        algorithm=FINGERPRINT_ALGORITHM,
        fingerprint=digest.hexdigest(),
        file_count=len(normalized_paths),
        total_bytes=total_bytes,
        content_hashes=content_hashes,
        unreadable_files=unreadable_files,
    )
