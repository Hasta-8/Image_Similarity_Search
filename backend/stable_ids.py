"""Deterministic image IDs for candidate index rebuilds.

Image identity is the normalized dataset-relative path. IDs are derived from a
SHA-256 digest of that path, encoded as a positive 63-bit integer so they are
stable across processes and directly usable by FAISS ``IndexIDMap``.
"""

from __future__ import annotations

import hashlib
import os
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Union

ID_STRATEGY_NAME = "sha256-dataset-relative-path"
ID_STRATEGY_VERSION = 1
ID_NAMESPACE = b"imgsimisearch:image-id:v1\x00"
ID_BITS = 63
ID_MASK = (1 << ID_BITS) - 1

PathLike = Union[str, os.PathLike]


class StableIDError(ValueError):
    """Raised when a stable ID cannot be generated safely."""


class IDCollisionError(StableIDError):
    """Raised when two different image identities map to the same ID."""


def normalize_dataset_relative_path(relative_path: PathLike) -> str:
    """Return a canonical, forward-slash dataset-relative path."""
    raw = os.fspath(relative_path).replace("\\", "/")
    path = PurePosixPath(raw)

    if path.is_absolute():
        raise StableIDError(f"Image identity must be relative to the dataset: {raw!r}")

    parts = path.parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise StableIDError(f"Invalid dataset-relative path: {raw!r}")

    normalized_parts = tuple(unicodedata.normalize("NFC", part) for part in parts)
    normalized = "/".join(normalized_parts)
    if not normalized or normalized.startswith("/"):
        raise StableIDError(f"Invalid dataset-relative path: {raw!r}")
    return normalized


def dataset_relative_path(dataset_dir: PathLike, image_path: PathLike) -> str:
    """Build a normalized dataset-relative path without resolving symlinks."""
    root = Path(os.path.abspath(os.fspath(dataset_dir)))
    image = Path(os.path.abspath(os.fspath(image_path)))

    try:
        relative = image.relative_to(root)
    except ValueError as exc:
        raise StableIDError(
            f"Image path is outside dataset root: image={image}, root={root}"
        ) from exc

    return normalize_dataset_relative_path(relative.as_posix())


def stable_image_id(normalized_relative_path: PathLike) -> int:
    """Return the stable positive 63-bit ID for a relative image path."""
    normalized = normalize_dataset_relative_path(normalized_relative_path)
    encoded = normalized.encode("utf-8")
    digest = hashlib.sha256(
        ID_NAMESPACE + len(encoded).to_bytes(8, "big") + encoded
    ).digest()
    return int.from_bytes(digest[:8], "big") & ID_MASK


def build_stable_ids(relative_paths: Iterable[PathLike]) -> Dict[str, int]:
    """Build a deterministic path-to-ID map and reject duplicates/collisions."""
    normalized_paths = [normalize_dataset_relative_path(path) for path in relative_paths]

    seen_paths = set()
    for path in normalized_paths:
        if path in seen_paths:
            raise StableIDError(f"Duplicate image identity: {path!r}")
        seen_paths.add(path)

    ids: Dict[str, int] = {}
    id_owners: Dict[int, str] = {}
    for path in sorted(normalized_paths):
        image_id = stable_image_id(path)
        if image_id in id_owners:
            raise IDCollisionError(
                f"Stable ID collision for {image_id}: "
                f"{id_owners[image_id]!r} and {path!r}"
            )
        ids[path] = image_id
        id_owners[image_id] = path

    return ids


def id_strategy_metadata() -> dict:
    """Return the manifest description of the active ID strategy."""
    return {
        "name": ID_STRATEGY_NAME,
        "version": ID_STRATEGY_VERSION,
        "identity": "normalized dataset-relative path",
        "id_bits": ID_BITS,
        "encoding": "positive int64-compatible integer",
        "hash": "sha256",
    }
