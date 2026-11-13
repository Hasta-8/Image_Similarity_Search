"""Safe resolution of dataset-relative image paths.

Artifact metadata stores paths relative to a dataset root. Runtime code must not
trust those strings: they can contain traversal, absolute paths, NUL bytes, or
symlinks that escape the dataset root.
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path, PurePosixPath
from urllib.parse import quote

_DRIVE_RE = re.compile(r"^[A-Za-z]:")


class DatasetPathError(ValueError):
    """Raised when a dataset-relative path is unsafe or cannot be resolved."""


def normalize_relative_path(value: str) -> str:
    """Validate and normalize a dataset-relative POSIX path.

    The returned path is NFC-normalized and never starts with ``/``. Empty,
    dot, parent, backslash, drive-letter, UNC, NUL, and colon-containing paths
    are rejected so the same metadata is safe on POSIX and Windows.
    """
    if not isinstance(value, str):
        raise DatasetPathError("Image path must be a string")
    if not value:
        raise DatasetPathError("Image path must not be empty")
    if "\x00" in value or "\\" in value or ":" in value:
        raise DatasetPathError(f"Image path contains a forbidden character: {value!r}")
    if value.startswith("/"):
        raise DatasetPathError(f"Image path must be dataset-relative: {value!r}")
    if _DRIVE_RE.match(value):
        raise DatasetPathError(f"Image path must not contain a drive letter: {value!r}")

    normalized = unicodedata.normalize("NFC", value)
    parts = normalized.split("/")
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise DatasetPathError(f"Image path is not normalized: {value!r}")
    return PurePosixPath(normalized).as_posix()


def _encode_relative(value: str) -> str:
    relative = normalize_relative_path(value)
    return "/".join(quote(part, safe="") for part in PurePosixPath(relative).parts)


class DatasetPathResolver:
    """Resolve metadata paths under exactly one dataset root."""

    def __init__(self, dataset_root: str | os.PathLike):
        root = Path(dataset_root).expanduser()
        if not root.is_dir():
            raise DatasetPathError(f"Dataset root is not a directory: {root}")
        self.root = root.resolve(strict=True)

    def relative_path(self, value: str) -> str:
        return normalize_relative_path(value)

    def resolve(self, value: str, *, must_exist: bool = False) -> Path:
        """Return a resolved path that remains inside the dataset root."""
        relative = normalize_relative_path(value)
        candidate = self.root.joinpath(*PurePosixPath(relative).parts)

        try:
            resolved = candidate.resolve(strict=must_exist)
        except FileNotFoundError as exc:
            raise DatasetPathError(f"Image does not exist: {relative}") from exc
        except OSError as exc:
            raise DatasetPathError(f"Could not resolve image path {relative!r}: {exc}") from exc

        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise DatasetPathError(
                f"Image path escapes the dataset root: {relative!r}"
            ) from exc

        if must_exist and not resolved.is_file():
            raise DatasetPathError(f"Image is not a regular file: {relative}")
        return resolved

    def image_url(self, value: str) -> str:
        """Build an encoded ``/images/...`` URL for a relative image path."""
        return f"/images/{_encode_relative(value)}"

    def thumbnail_url(self, value: str) -> str:
        """Build an encoded ``/thumbnails/...`` URL for a relative image path."""
        return f"/thumbnails/{_encode_relative(value)}"

    def metadata_paths(self, metadata: dict[int, str], *, must_exist: bool = False) -> dict[int, str]:
        """Validate every metadata path while preserving the mapping."""
        result = {}
        for image_id, path in metadata.items():
            relative = self.relative_path(path)
            if must_exist:
                self.resolve(relative, must_exist=True)
            result[image_id] = relative
        return result
