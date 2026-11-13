"""Candidate index manifest creation and loading."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from dataset_fingerprint import sha256_file

MANIFEST_SCHEMA_VERSION = "1.0"
INDEX_FILENAME = "faiss_index.bin"
METADATA_FILENAME = "metadata.json"
MANIFEST_FILENAME = "manifest.json"

SEMANTIC_MANIFEST_DOMAIN = b"imgsimisearch:manifest-semantic:v1\0"
SEMANTIC_MANIFEST_KEYS = (
    "dataset",
    "embedding",
    "id_strategy",
    "indexing",
    "index",
    "failures",
)


class ManifestError(ValueError):
    """Raised when a manifest cannot be built, parsed, or written."""


def utc_now_iso() -> str:
    """Return a UTC timestamp in ISO-8601 ``Z`` form."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"Duplicate JSON key in manifest: {key!r}")
        result[key] = value
    return result


def load_manifest(manifest_path: str | os.PathLike) -> dict:
    """Load a manifest, rejecting duplicate keys and non-object roots."""
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    except ManifestError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Could not read manifest {manifest_path}: {exc}") from exc

    if not isinstance(manifest, dict):
        raise ManifestError("Manifest root must be a JSON object")
    return manifest


def semantic_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash retrieval semantics, excluding timestamps, roles, and release state.

    This fingerprint covers the dataset identity, embedding configuration,
    stable-ID strategy, indexing behavior, index metadata, and failures. It does
    not cover ``generated_at_utc``, ``artifact_role``, ``promotion_status``, or
    the ``release`` block, so re-promoting semantically identical artifacts
    remains idempotent while semantic changes cannot reuse the same identity.
    """
    if not isinstance(manifest, Mapping):
        raise ManifestError("Manifest root must be a JSON object")
    missing = [key for key in SEMANTIC_MANIFEST_KEYS if key not in manifest]
    if missing:
        raise ManifestError(
            "Manifest is missing semantic field(s): " + ", ".join(missing)
        )

    payload = {"schema_version": manifest.get("schema_version")}
    for key in SEMANTIC_MANIFEST_KEYS:
        payload[key] = manifest[key]

    try:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"Manifest semantic fields are not canonical JSON: {exc}") from exc
    return hashlib.sha256(SEMANTIC_MANIFEST_DOMAIN + canonical).hexdigest()


def write_manifest(manifest: Mapping[str, Any], manifest_path: str | os.PathLike) -> None:
    """Write a manifest deterministically as UTF-8 JSON."""
    try:
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
    except (OSError, TypeError) as exc:
        raise ManifestError(f"Could not write manifest {manifest_path}: {exc}") from exc


def build_manifest(
    artifact_dir: str | os.PathLike,
    embedding: Mapping[str, Any],
    dataset: Mapping[str, Any],
    id_strategy: Mapping[str, Any],
    indexing: Mapping[str, Any],
    indexed_images: int,
    failed_images: int,
    metadata_records: int,
    failures: Sequence[Mapping[str, Any]],
    faiss_type: str,
    index_filename: str = INDEX_FILENAME,
    metadata_filename: str = METADATA_FILENAME,
    generated_at_utc: str | None = None,
) -> dict:
    """Build a manifest after the candidate index and metadata are written."""
    artifact_dir = Path(artifact_dir)
    index_path = artifact_dir / index_filename
    metadata_path = artifact_dir / metadata_filename

    if not index_path.is_file():
        raise ManifestError(f"Candidate index file is missing: {index_path}")
    if not metadata_path.is_file():
        raise ManifestError(f"Candidate metadata file is missing: {metadata_path}")

    embedding_dim = int(embedding["embedding_dim"])
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_role": "candidate",
        "promotion_status": "not_promoted",
        "generated_at_utc": generated_at_utc or utc_now_iso(),
        "dataset": dict(dataset),
        "embedding": dict(embedding),
        "id_strategy": dict(id_strategy),
        "indexing": dict(indexing),
        "index": {
            "index_filename": index_filename,
            "index_sha256": sha256_file(index_path),
            "index_size_bytes": index_path.stat().st_size,
            "metadata_filename": metadata_filename,
            "metadata_sha256": sha256_file(metadata_path),
            "metadata_size_bytes": metadata_path.stat().st_size,
            "faiss_type": faiss_type,
            "dimension": embedding_dim,
            "distance_metric": "inner_product",
            "similarity_strategy": "cosine_on_l2_normalized_vectors",
            "indexed_vectors": indexed_images,
            "indexed_images": indexed_images,
            "failed_images": failed_images,
            "metadata_records": metadata_records,
        },
        "failures": [dict(failure) for failure in failures],
    }
    return manifest
