"""Reusable integrity validation for candidate index artifact sets."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import faiss

from artifact_manifest import (
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    METADATA_FILENAME,
    ManifestError,
    load_manifest,
    semantic_manifest_sha256,
)
from dataset_fingerprint import sha256_file
from faiss_index import load_faiss_index

_ID_KEY_RE = re.compile(r"^(0|[1-9][0-9]*)$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_INT64_ID = (1 << 63) - 1


class ArtifactValidationError(ValueError):
    """Raised when callers require a valid artifact set and it is not valid."""


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def raise_for_errors(self) -> None:
        if not self.valid:
            joined = "\n".join(f"- {error}" for error in self.errors)
            raise ArtifactValidationError(f"Artifact validation failed:\n{joined}")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json_strict(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=_reject_duplicate_keys)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_mapping(
    value: Any, field_name: str, errors: List[str]
) -> Optional[Mapping[str, Any]]:
    if not isinstance(value, dict):
        errors.append(f"{field_name} must be a JSON object")
        return None
    return value


def _require_str(
    mapping: Mapping[str, Any],
    key: str,
    field_name: str,
    errors: List[str],
    *,
    allow_empty: bool = False,
) -> Optional[str]:
    value = mapping.get(key)
    if not isinstance(value, str):
        errors.append(f"{field_name} must be a string")
        return None
    if not allow_empty and not value:
        errors.append(f"{field_name} must not be empty")
        return None
    return value


def _require_int(
    mapping: Mapping[str, Any],
    key: str,
    field_name: str,
    errors: List[str],
    *,
    minimum: int = 0,
) -> Optional[int]:
    value = mapping.get(key)
    if not _is_int(value):
        errors.append(f"{field_name} must be an integer")
        return None
    if value < minimum:
        errors.append(f"{field_name} must be >= {minimum}, got {value}")
        return None
    return value


def _require_sha256(
    mapping: Mapping[str, Any], key: str, field_name: str, errors: List[str]
) -> Optional[str]:
    value = _require_str(mapping, key, field_name, errors)
    if value is not None and not _SHA256_RE.match(value):
        errors.append(f"{field_name} must be a 64-character lowercase SHA-256 hex digest")
        return None
    return value


def _is_normalized_relative_path(value: str) -> bool:
    if not value or value.startswith("/") or "\\" in value:
        return False
    parts = value.split("/")
    return all(part not in ("", ".", "..") for part in parts)


def _extract_index_ids(index: faiss.Index) -> List[int]:
    if not hasattr(index, "id_map"):
        raise ValueError("Index is not an IndexIDMap and has no id_map")
    ids = faiss.vector_to_array(index.id_map)
    return [int(value) for value in ids]


def validate_candidate_artifacts(
    artifact_dir: str | os.PathLike,
    expected_embedding_dim: Optional[int] = None,
    expected_count: Optional[int] = None,
    *,
    expected_role: str = "candidate",
    expected_promotion_status: str = "not_promoted",
) -> ValidationResult:
    """Validate an index, metadata file, and manifest as one artifact set.

    Candidate validation remains the default. Release validation supplies the
    production role and promoted status explicitly.
    """
    result = ValidationResult(valid=False)
    errors = result.errors
    warnings = result.warnings
    directory = Path(artifact_dir)

    if not directory.is_dir():
        errors.append(f"Artifact directory does not exist or is not a directory: {directory}")
        return result

    manifest_path = directory / MANIFEST_FILENAME
    index_path = directory / INDEX_FILENAME
    metadata_path = directory / METADATA_FILENAME

    for path, label in (
        (manifest_path, "manifest"),
        (index_path, "FAISS index"),
        (metadata_path, "metadata"),
    ):
        if not path.is_file():
            errors.append(f"Missing {label} file: {path}")

    if errors:
        return result

    try:
        manifest = load_manifest(manifest_path)
    except ManifestError as exc:
        errors.append(str(exc))
        return result

    schema_version = manifest.get("schema_version")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        errors.append(
            f"Unsupported manifest schema_version {schema_version!r}; "
            f"expected {MANIFEST_SCHEMA_VERSION!r}"
        )
    if manifest.get("artifact_role") != expected_role:
        errors.append(
            f"manifest.artifact_role must be {expected_role!r}, "
            f"got {manifest.get('artifact_role')!r}"
        )
    if manifest.get("promotion_status") != expected_promotion_status:
        errors.append(
            "manifest.promotion_status must be "
            f"{expected_promotion_status!r}, got {manifest.get('promotion_status')!r}"
        )
    _require_str(manifest, "generated_at_utc", "manifest.generated_at_utc", errors)

    semantic_sha256 = None
    try:
        semantic_sha256 = semantic_manifest_sha256(manifest)
    except ManifestError as exc:
        errors.append(str(exc))

    if expected_role == "production":
        release = _require_mapping(manifest.get("release"), "manifest.release", errors)
        if release is not None:
            _require_str(release, "release_id", "manifest.release.release_id", errors)
            _require_str(
                release,
                "promoted_at_utc",
                "manifest.release.promoted_at_utc",
                errors,
            )
            _require_sha256(
                release,
                "source_manifest_sha256",
                "manifest.release.source_manifest_sha256",
                errors,
            )
            recorded_semantic_sha = _require_sha256(
                release,
                "semantic_manifest_sha256",
                "manifest.release.semantic_manifest_sha256",
                errors,
            )
            if (
                recorded_semantic_sha is not None
                and semantic_sha256 is not None
                and recorded_semantic_sha != semantic_sha256
            ):
                errors.append(
                    "manifest.release.semantic_manifest_sha256 does not match "
                    "the release manifest's semantic content"
                )
            previous = release.get("previous_release_id")
            if previous is not None and (
                not isinstance(previous, str) or not previous
            ):
                errors.append(
                    "manifest.release.previous_release_id must be a non-empty "
                    "string or null"
                )

    dataset = _require_mapping(manifest.get("dataset"), "manifest.dataset", errors)
    embedding = _require_mapping(manifest.get("embedding"), "manifest.embedding", errors)
    index_manifest = _require_mapping(manifest.get("index"), "manifest.index", errors)
    id_strategy = _require_mapping(manifest.get("id_strategy"), "manifest.id_strategy", errors)
    indexing = _require_mapping(manifest.get("indexing"), "manifest.indexing", errors)

    if dataset is not None:
        _require_str(dataset, "identifier", "manifest.dataset.identifier", errors)
        _require_str(
            dataset,
            "fingerprint_algorithm",
            "manifest.dataset.fingerprint_algorithm",
            errors,
        )
        _require_sha256(
            dataset, "fingerprint", "manifest.dataset.fingerprint", errors
        )
        dataset_file_count = _require_int(
            dataset, "file_count", "manifest.dataset.file_count", errors
        )
        _require_int(dataset, "total_bytes", "manifest.dataset.total_bytes", errors)
    else:
        dataset_file_count = None

    embedding_dim = None
    if embedding is not None:
        _require_str(embedding, "model_name", "manifest.embedding.model_name", errors)
        model_revision = embedding.get("model_revision")
        if model_revision is not None and not isinstance(model_revision, str):
            errors.append("manifest.embedding.model_revision must be a string or null")
        elif not model_revision:
            warnings.append("manifest.embedding.model_revision is absent")
        embedding_dim = _require_int(
            embedding,
            "embedding_dim",
            "manifest.embedding.embedding_dim",
            errors,
            minimum=1,
        )
        _require_str(
            embedding, "normalization", "manifest.embedding.normalization", errors
        )
        preprocessing = _require_mapping(
            embedding.get("preprocessing"), "manifest.embedding.preprocessing", errors
        )
        if preprocessing is not None and not preprocessing:
            errors.append("manifest.embedding.preprocessing must not be empty")

    manifest_counts: Dict[str, Optional[int]] = {}
    if index_manifest is not None:
        index_filename = _require_str(
            index_manifest, "index_filename", "manifest.index.index_filename", errors
        )
        if index_filename is not None and index_filename != INDEX_FILENAME:
            errors.append(
                f"manifest.index.index_filename must be {INDEX_FILENAME!r}, "
                f"got {index_filename!r}"
            )
        metadata_filename = _require_str(
            index_manifest,
            "metadata_filename",
            "manifest.index.metadata_filename",
            errors,
        )
        if metadata_filename is not None and metadata_filename != METADATA_FILENAME:
            errors.append(
                f"manifest.index.metadata_filename must be {METADATA_FILENAME!r}, "
                f"got {metadata_filename!r}"
            )

        _require_sha256(
            index_manifest, "index_sha256", "manifest.index.index_sha256", errors
        )
        _require_sha256(
            index_manifest,
            "metadata_sha256",
            "manifest.index.metadata_sha256",
            errors,
        )
        _require_int(
            index_manifest, "index_size_bytes", "manifest.index.index_size_bytes", errors
        )
        _require_int(
            index_manifest,
            "metadata_size_bytes",
            "manifest.index.metadata_size_bytes",
            errors,
        )
        _require_str(index_manifest, "faiss_type", "manifest.index.faiss_type", errors)
        _require_str(
            index_manifest,
            "distance_metric",
            "manifest.index.distance_metric",
            errors,
        )
        _require_str(
            index_manifest,
            "similarity_strategy",
            "manifest.index.similarity_strategy",
            errors,
        )
        _require_int(
            index_manifest, "dimension", "manifest.index.dimension", errors, minimum=1
        )
        for count_key in (
            "indexed_vectors",
            "indexed_images",
            "failed_images",
            "metadata_records",
        ):
            manifest_counts[count_key] = _require_int(
                index_manifest,
                count_key,
                f"manifest.index.{count_key}",
                errors,
            )

    if id_strategy is not None:
        _require_str(id_strategy, "name", "manifest.id_strategy.name", errors)
        _require_int(
            id_strategy, "version", "manifest.id_strategy.version", errors, minimum=1
        )

    if indexing is not None:
        _require_int(
            indexing, "batch_size", "manifest.indexing.batch_size", errors, minimum=1
        )

    failures = manifest.get("failures")
    if not isinstance(failures, list):
        errors.append("manifest.failures must be a list")
        failures = []
    else:
        failure_paths = []
        for item_index, failure in enumerate(failures):
            if not isinstance(failure, dict):
                errors.append(f"manifest.failures[{item_index}] must be an object")
                continue
            path_value = failure.get("path")
            if not isinstance(path_value, str) or not _is_normalized_relative_path(path_value):
                errors.append(
                    f"manifest.failures[{item_index}].path must be a normalized "
                    "dataset-relative path"
                )
            else:
                failure_paths.append(path_value)
            _require_str(
                failure, "stage", f"manifest.failures[{item_index}].stage", errors
            )
            _require_str(
                failure,
                "error_type",
                f"manifest.failures[{item_index}].error_type",
                errors,
            )
        if len(failure_paths) != len(set(failure_paths)):
            errors.append("manifest.failures contains duplicate image paths")

    if (
        dataset_file_count is not None
        and manifest_counts.get("indexed_images") is not None
        and manifest_counts.get("failed_images") is not None
    ):
        expected_files = manifest_counts["indexed_images"] + manifest_counts["failed_images"]
        if dataset_file_count != expected_files:
            errors.append(
                "manifest.dataset.file_count must equal indexed_images + failed_images "
                f"({expected_files}), got {dataset_file_count}"
            )

    if (
        manifest_counts.get("metadata_records") is not None
        and manifest_counts.get("indexed_images") is not None
        and manifest_counts["metadata_records"] != manifest_counts["indexed_images"]
    ):
        errors.append(
            "manifest.index.metadata_records must equal indexed_images "
            f"({manifest_counts['indexed_images']}), "
            f"got {manifest_counts['metadata_records']}"
        )

    if len(failures) != manifest_counts.get("failed_images"):
        errors.append(
            "manifest.failures length must equal failed_images "
            f"({manifest_counts.get('failed_images')}), got {len(failures)}"
        )

    index = None
    try:
        index = load_faiss_index(str(index_path))
    except Exception as exc:
        errors.append(f"Could not load FAISS index {index_path}: {exc}")

    index_ids: List[int] = []
    if index is not None:
        if expected_embedding_dim is not None and index.d != expected_embedding_dim:
            errors.append(
                f"FAISS index dimension is {index.d}; expected {expected_embedding_dim}"
            )
        if embedding_dim is not None and index.d != embedding_dim:
            errors.append(
                f"FAISS index dimension {index.d} does not match "
                f"manifest embedding dimension {embedding_dim}"
            )
        if index_manifest is not None and _is_int(index_manifest.get("dimension")):
            if index.d != index_manifest["dimension"]:
                errors.append(
                    f"FAISS index dimension {index.d} does not match "
                    f"manifest.index.dimension {index_manifest['dimension']}"
                )
        if expected_count is not None and index.ntotal != expected_count:
            errors.append(
                f"FAISS index contains {index.ntotal} vectors; expected {expected_count}"
            )
        if manifest_counts.get("indexed_vectors") is not None and (
            index.ntotal != manifest_counts["indexed_vectors"]
        ):
            errors.append(
                f"FAISS index contains {index.ntotal} vectors; manifest records "
                f"{manifest_counts['indexed_vectors']}"
            )
        if (
            index_manifest is not None
            and isinstance(index_manifest.get("faiss_type"), str)
            and index_manifest["faiss_type"] != type(index).__name__
        ):
            errors.append(
                f"Manifest FAISS type {index_manifest['faiss_type']!r} does not match "
                f"loaded type {type(index).__name__!r}"
            )

        try:
            index_ids = _extract_index_ids(index)
        except Exception as exc:
            errors.append(f"Could not extract FAISS IDs: {exc}")
        else:
            if len(index_ids) != index.ntotal:
                errors.append(
                    f"FAISS ID map contains {len(index_ids)} IDs but index.ntotal is "
                    f"{index.ntotal}"
                )
            if len(index_ids) != len(set(index_ids)):
                duplicate_count = len(index_ids) - len(set(index_ids))
                errors.append(
                    f"FAISS index contains {duplicate_count} duplicate ID(s)"
                )
            invalid_ids = [value for value in index_ids if value < 0 or value > _MAX_INT64_ID]
            if invalid_ids:
                errors.append(
                    f"FAISS index contains {len(invalid_ids)} invalid ID(s); "
                    f"first invalid IDs: {invalid_ids[:5]}"
                )

    metadata: Optional[Dict[int, str]] = None
    try:
        raw_metadata = _load_json_strict(metadata_path)
    except Exception as exc:
        errors.append(f"Could not parse metadata {metadata_path}: {exc}")
    else:
        if not isinstance(raw_metadata, dict):
            errors.append("Metadata root must be a JSON object")
        else:
            metadata = {}
            for raw_id, path_value in raw_metadata.items():
                if not isinstance(raw_id, str) or not _ID_KEY_RE.match(raw_id):
                    errors.append(
                        f"Metadata ID {raw_id!r} is not a canonical non-negative integer string"
                    )
                    continue
                image_id = int(raw_id)
                if not isinstance(path_value, str) or not path_value:
                    errors.append(
                        f"Metadata record {raw_id!r} must have a non-empty string path"
                    )
                    continue
                if not _is_normalized_relative_path(path_value):
                    errors.append(
                        f"Metadata record {raw_id!r} path must be a normalized "
                        f"dataset-relative path, got {path_value!r}"
                    )
                    continue
                metadata[image_id] = path_value

            if metadata is not None:
                if len(raw_metadata) != len(metadata):
                    errors.append("Metadata contains invalid records")
                if manifest_counts.get("metadata_records") is not None and (
                    len(metadata) != manifest_counts["metadata_records"]
                ):
                    errors.append(
                        f"Metadata contains {len(metadata)} records; manifest records "
                        f"{manifest_counts['metadata_records']}"
                    )
                if manifest_counts.get("indexed_images") is not None and (
                    len(metadata) != manifest_counts["indexed_images"]
                ):
                    errors.append(
                        f"Metadata contains {len(metadata)} records; manifest records "
                        f"{manifest_counts['indexed_images']} indexed images"
                    )
                if expected_count is not None and len(metadata) != expected_count:
                    errors.append(
                        f"Metadata contains {len(metadata)} records; expected {expected_count}"
                    )

                index_id_set = set(index_ids)
                metadata_id_set = set(metadata)
                missing_ids = sorted(index_id_set - metadata_id_set)
                unexpected_ids = sorted(metadata_id_set - index_id_set)
                if missing_ids:
                    errors.append(
                        f"Metadata is missing {len(missing_ids)} indexed ID(s); "
                        f"first missing IDs: {missing_ids[:5]}"
                    )
                if unexpected_ids:
                    errors.append(
                        f"Metadata contains {len(unexpected_ids)} unexpected ID(s); "
                        f"first unexpected IDs: {unexpected_ids[:5]}"
                    )

    checksum_errors_before = len(errors)
    actual_index_sha = sha256_file(index_path)
    actual_metadata_sha = sha256_file(metadata_path)
    if index_manifest is not None:
        expected_index_sha = index_manifest.get("index_sha256")
        if isinstance(expected_index_sha, str) and expected_index_sha != actual_index_sha:
            errors.append(
                "Index checksum mismatch: manifest records "
                f"{expected_index_sha}, actual {actual_index_sha}"
            )
        expected_metadata_sha = index_manifest.get("metadata_sha256")
        if isinstance(expected_metadata_sha, str) and expected_metadata_sha != actual_metadata_sha:
            errors.append(
                "Metadata checksum mismatch: manifest records "
                f"{expected_metadata_sha}, actual {actual_metadata_sha}"
            )
        if _is_int(index_manifest.get("index_size_bytes")) and (
            index_manifest["index_size_bytes"] != index_path.stat().st_size
        ):
            errors.append(
                f"Index size mismatch: manifest records "
                f"{index_manifest['index_size_bytes']}, actual {index_path.stat().st_size}"
            )
        if _is_int(index_manifest.get("metadata_size_bytes")) and (
            index_manifest["metadata_size_bytes"] != metadata_path.stat().st_size
        ):
            errors.append(
                f"Metadata size mismatch: manifest records "
                f"{index_manifest['metadata_size_bytes']}, "
                f"actual {metadata_path.stat().st_size}"
            )

    result.details = {
        "artifact_dir": str(directory),
        "schema_version": schema_version,
        "artifact_role": manifest.get("artifact_role"),
        "promotion_status": manifest.get("promotion_status"),
        "release_id": (
            manifest.get("release", {}).get("release_id")
            if isinstance(manifest.get("release"), dict)
            else None
        ),
        "index_dimension": index.d if index is not None else None,
        "index_ntotal": index.ntotal if index is not None else None,
        "index_id_count": len(index_ids),
        "metadata_count": len(metadata) if metadata is not None else None,
        "dataset_fingerprint": dataset.get("fingerprint") if dataset else None,
        "semantic_manifest_sha256": semantic_sha256,
        "index_sha256": actual_index_sha,
        "metadata_sha256": actual_metadata_sha,
        "checksums_verified": len(errors) == checksum_errors_before,
    }
    result.valid = not errors
    return result


def validate_release_artifacts(
    artifact_dir: str | os.PathLike,
    expected_embedding_dim: Optional[int] = None,
    expected_count: Optional[int] = None,
) -> ValidationResult:
    """Validate an immutable production release artifact set."""
    return validate_candidate_artifacts(
        artifact_dir,
        expected_embedding_dim=expected_embedding_dim,
        expected_count=expected_count,
        expected_role="production",
        expected_promotion_status="promoted",
    )
