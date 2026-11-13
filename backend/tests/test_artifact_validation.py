"""Tests for candidate artifact manifest and integrity validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from artifact_manifest import (
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    METADATA_FILENAME,
    build_manifest,
    load_manifest,
    write_manifest,
)
from artifact_validation import (
    ArtifactValidationError,
    validate_candidate_artifacts,
)
from faiss_index import (
    add_embeddings,
    create_faiss_index,
    save_faiss_index,
    save_metadata,
)


def _embedding_metadata(dim: int = 8) -> dict:
    return {
        "model_name": "test-model",
        "model_revision": "test-revision",
        "embedding_dim": dim,
        "normalization": "l2",
        "device": "cpu",
        "preprocessing": {"input_mode": "RGB", "image_size": 224},
    }


def _dataset_metadata(file_count: int = 2) -> dict:
    return {
        "identifier": "test-dataset",
        "root": "test-dataset",
        "fingerprint": "a" * 64,
        "fingerprint_algorithm": "sha256-path-and-content-v1",
        "file_count": file_count,
        "total_bytes": 1234,
    }


def _build_candidate(
    directory: Path,
    *,
    dim: int = 8,
    ids: tuple[int, ...] = (11, 22),
    paths: tuple[str, ...] = ("nested/a.jpg", "b.png"),
    failed_images: int = 0,
    failures: tuple[dict, ...] = (),
    dataset_file_count: int | None = None,
    apply_generated_at: bool = True,
) -> tuple[Path, dict]:
    directory.mkdir()
    index = create_faiss_index(dim)
    vectors = np.eye(dim, dtype=np.float32)[: len(ids)]
    add_embeddings(index, vectors, list(ids))
    save_faiss_index(index, str(directory / INDEX_FILENAME))
    save_metadata(dict(zip(ids, paths)), str(directory / METADATA_FILENAME))

    manifest = build_manifest(
        artifact_dir=directory,
        embedding=_embedding_metadata(dim),
        dataset=_dataset_metadata(
            dataset_file_count
            if dataset_file_count is not None
            else len(ids) + failed_images
        ),
        id_strategy={
            "name": "sha256-dataset-relative-path",
            "version": 1,
            "identity": "normalized dataset-relative path",
            "id_bits": 63,
        },
        indexing={"mode": "full_candidate_rebuild", "batch_size": 2, "device": "cpu"},
        indexed_images=len(ids),
        failed_images=failed_images,
        metadata_records=len(ids),
        failures=list(failures),
        faiss_type=type(index).__name__,
        generated_at_utc="2026-01-01T00:00:00Z" if apply_generated_at else None,
    )
    write_manifest(manifest, directory / MANIFEST_FILENAME)
    return directory, manifest


def _rewrite_manifest(directory: Path, manifest: dict) -> None:
    write_manifest(manifest, directory / MANIFEST_FILENAME)


def test_valid_candidate_passes_all_checks(tmp_path):
    directory, _ = _build_candidate(tmp_path / "candidate")

    result = validate_candidate_artifacts(
        directory, expected_embedding_dim=8, expected_count=2
    )

    assert result.valid, result.errors
    assert result.errors == []
    assert result.details["index_dimension"] == 8
    assert result.details["index_ntotal"] == 2
    assert result.details["index_id_count"] == 2
    assert result.details["metadata_count"] == 2
    assert result.details["checksums_verified"] is True
    result.raise_for_errors()


def test_valid_candidate_with_documented_failures(tmp_path):
    directory, _ = _build_candidate(
        tmp_path / "candidate",
        failed_images=1,
        failures=(
            {
                "path": "corrupt.jpg",
                "stage": "embedding",
                "error_type": "EmbeddingError",
                "message": "Could not load image",
            },
        ),
    )

    result = validate_candidate_artifacts(directory, expected_count=2)

    assert result.valid, result.errors


def test_missing_artifact_files_are_rejected(tmp_path):
    directory, _ = _build_candidate(tmp_path / "candidate")
    (directory / METADATA_FILENAME).unlink()

    result = validate_candidate_artifacts(directory)

    assert not result.valid
    assert any("Missing metadata file" in error for error in result.errors)


def test_checksum_and_size_mismatch_are_rejected(tmp_path):
    directory, _ = _build_candidate(tmp_path / "candidate")
    metadata_path = directory / METADATA_FILENAME
    metadata_path.write_text(
        metadata_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    result = validate_candidate_artifacts(directory)

    assert not result.valid
    assert any("Metadata checksum mismatch" in error for error in result.errors)
    assert any("Metadata size mismatch" in error for error in result.errors)
    assert result.details["checksums_verified"] is False


def test_metadata_must_cover_indexed_ids(tmp_path):
    directory, _ = _build_candidate(tmp_path / "candidate")
    save_metadata({11: "nested/a.jpg"}, str(directory / METADATA_FILENAME))

    result = validate_candidate_artifacts(directory)

    assert not result.valid
    assert any("Metadata is missing 1 indexed ID" in error for error in result.errors)
    assert any("Metadata contains 1 records" in error for error in result.errors)


def test_metadata_rejects_absolute_and_noncanonical_paths(tmp_path):
    directory, _ = _build_candidate(tmp_path / "candidate")
    save_metadata(
        {11: "/absolute/a.jpg", 22: "nested/../b.png"},
        str(directory / METADATA_FILENAME),
    )

    result = validate_candidate_artifacts(directory)

    assert not result.valid
    assert sum("normalized dataset-relative path" in error for error in result.errors) >= 2


def test_metadata_rejects_noncanonical_id_keys(tmp_path):
    directory, _ = _build_candidate(tmp_path / "candidate")
    save_metadata({1: "a.jpg", 2: "b.jpg"}, str(directory / METADATA_FILENAME))
    metadata_path = directory / METADATA_FILENAME
    metadata_path.write_text(
        '{"01": "a.jpg", "2": "b.jpg"}', encoding="utf-8"
    )

    result = validate_candidate_artifacts(directory)

    assert not result.valid
    assert any("canonical non-negative integer string" in error for error in result.errors)


def test_duplicate_metadata_json_keys_are_rejected(tmp_path):
    directory, _ = _build_candidate(tmp_path / "candidate")
    (directory / METADATA_FILENAME).write_text(
        '{"11": "a.jpg", "11": "b.jpg", "22": "c.jpg"}', encoding="utf-8"
    )

    result = validate_candidate_artifacts(directory)

    assert not result.valid
    assert any("Duplicate JSON key" in error for error in result.errors)


def test_expected_dimension_and_count_are_checked(tmp_path):
    directory, _ = _build_candidate(tmp_path / "candidate")

    result = validate_candidate_artifacts(
        directory, expected_embedding_dim=16, expected_count=3
    )

    assert not result.valid
    assert any("expected 16" in error for error in result.errors)
    assert any("expected 3" in error for error in result.errors)


def test_manifest_schema_role_and_promotion_status_are_checked(tmp_path):
    directory, manifest = _build_candidate(tmp_path / "candidate")
    manifest["schema_version"] = "999"
    manifest["artifact_role"] = "production"
    manifest["promotion_status"] = "promoted"
    _rewrite_manifest(directory, manifest)

    result = validate_candidate_artifacts(directory)

    assert not result.valid
    assert any("Unsupported manifest schema_version" in error for error in result.errors)
    assert any("artifact_role" in error for error in result.errors)
    assert any("promotion_status" in error for error in result.errors)


def test_manifest_counts_must_be_consistent(tmp_path):
    directory, manifest = _build_candidate(tmp_path / "candidate")
    manifest["index"]["indexed_images"] = 3
    manifest["index"]["metadata_records"] = 1
    manifest["index"]["failed_images"] = 1
    _rewrite_manifest(directory, manifest)

    result = validate_candidate_artifacts(directory)

    assert not result.valid
    assert any(
        "file_count must equal indexed_images + failed_images" in error
        for error in result.errors
    )
    assert any("metadata_records must equal indexed_images" in error for error in result.errors)
    assert any("failures length must equal failed_images" in error for error in result.errors)


def test_failure_records_reject_nonrelative_or_duplicate_paths(tmp_path):
    directory, _ = _build_candidate(
        tmp_path / "candidate",
        failed_images=2,
        failures=(
            {"path": "/absolute.jpg", "stage": "embedding", "error_type": "OSError"},
            {"path": "same.jpg", "stage": "embedding", "error_type": "OSError"},
            {"path": "same.jpg", "stage": "embedding", "error_type": "OSError"},
        ),
        dataset_file_count=4,
    )

    result = validate_candidate_artifacts(directory)

    assert not result.valid
    assert any("normalized dataset-relative path" in error for error in result.errors)
    assert any("duplicate image paths" in error for error in result.errors)


def test_malformed_manifest_is_rejected(tmp_path):
    directory, _ = _build_candidate(tmp_path / "candidate")
    (directory / MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")

    result = validate_candidate_artifacts(directory)

    assert not result.valid
    assert any("Could not read manifest" in error for error in result.errors)


def test_raise_for_errors_raises_for_invalid_result(tmp_path):
    directory, _ = _build_candidate(tmp_path / "candidate")
    manifest = load_manifest(directory / MANIFEST_FILENAME)
    manifest["artifact_role"] = "production"
    _rewrite_manifest(directory, manifest)

    result = validate_candidate_artifacts(directory)

    with pytest.raises(ArtifactValidationError):
        result.raise_for_errors()
