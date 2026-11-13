"""Tests for immutable release promotion, activation, rollback, and loading."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import release_store
from artifact_manifest import (
    MANIFEST_FILENAME,
    load_manifest,
    semantic_manifest_sha256,
    write_manifest,
)
from artifact_validation import validate_release_artifacts
from dataset_fingerprint import sha256_file
from release_store import (
    ACTIVE_POINTER_FILENAME,
    ReleaseStore,
    ReleaseStoreError,
    release_id_for,
)
from tests.artifact_fixtures import build_candidate, make_dataset, unit_vector


def _promote(store: ReleaseStore, candidate: Path, dataset: Path, *, confirm: bool = True):
    return store.promote(candidate, confirm=confirm, dataset_root=dataset)


def _pointer(store: ReleaseStore) -> dict:
    return json.loads(store.pointer_path.read_text(encoding="utf-8"))


def test_dry_run_reports_plan_without_creating_store(tmp_path):
    dataset = make_dataset(tmp_path / "dataset")
    candidate, _ = build_candidate(tmp_path / "candidate")
    store = ReleaseStore(tmp_path / "store")

    plan = _promote(store, candidate, dataset, confirm=False)

    assert plan.dry_run is True
    assert plan.activated is False
    assert plan.release_id.startswith("rel-")
    assert plan.previous_release_id is None
    assert plan.indexed_images == 2
    assert not plan.release_dir.exists()
    assert not store.root.exists()
    assert not store.pointer_path.exists()


def test_confirmed_promotion_creates_immutable_release_and_pointer(tmp_path):
    dataset = make_dataset(tmp_path / "dataset")
    candidate, candidate_manifest = build_candidate(tmp_path / "candidate")
    candidate_index_hash = sha256_file(candidate / "faiss_index.bin")
    candidate_metadata_hash = sha256_file(candidate / "metadata.json")
    candidate_manifest_hash = sha256_file(candidate / MANIFEST_FILENAME)
    candidate_semantic_hash = semantic_manifest_sha256(candidate_manifest)
    store = ReleaseStore(tmp_path / "store")

    plan = _promote(store, candidate, dataset)

    assert plan.activated is True
    expected_id = release_id_for(
        candidate_index_hash, candidate_metadata_hash, candidate_semantic_hash
    )
    assert plan.release_id == expected_id
    assert plan.release_dir.is_dir()
    assert store.active_release_id() == expected_id

    manifest = json.loads(plan.release_dir.joinpath(MANIFEST_FILENAME).read_text())
    assert manifest["artifact_role"] == "production"
    assert manifest["promotion_status"] == "promoted"
    assert manifest["release"]["release_id"] == expected_id
    assert manifest["release"]["previous_release_id"] is None
    assert manifest["release"]["source_manifest_sha256"] == candidate_manifest_hash
    assert (
        manifest["release"]["semantic_manifest_sha256"] == candidate_semantic_hash
    )
    assert manifest["index"]["index_sha256"] == candidate_index_hash
    assert manifest["dataset"] == candidate_manifest["dataset"]

    validation = validate_release_artifacts(plan.release_dir)
    assert validation.valid, validation.errors
    assert validation.details["release_id"] == expected_id

    assert sha256_file(candidate / "faiss_index.bin") == candidate_index_hash
    assert sha256_file(candidate / "metadata.json") == candidate_metadata_hash
    assert sha256_file(candidate / MANIFEST_FILENAME) == candidate_manifest_hash

    pointer = _pointer(store)
    assert pointer["release_id"] == expected_id
    assert pointer["index_sha256"] == candidate_index_hash
    assert pointer["metadata_sha256"] == candidate_metadata_hash
    assert pointer["semantic_manifest_sha256"] == candidate_semantic_hash
    assert pointer["source_manifest_sha256"] == candidate_manifest_hash
    assert pointer["operation"] == "promote"


def test_repeated_promotion_is_idempotent(tmp_path):
    dataset = make_dataset(tmp_path / "dataset")
    candidate, _ = build_candidate(tmp_path / "candidate")
    store = ReleaseStore(tmp_path / "store")

    first = _promote(store, candidate, dataset)
    second = _promote(store, candidate, dataset)
    third = _promote(store, candidate, dataset, confirm=False)

    assert first.release_id == second.release_id == third.release_id
    assert second.already_active is True
    assert third.release_exists is True
    assert second.activated is False
    assert [entry.name for entry in store.releases_dir.iterdir() if entry.is_dir()] == [
        first.release_id
    ]


def test_repromoting_same_artifacts_reuses_release_manifest(tmp_path):
    dataset = make_dataset(tmp_path / "dataset")
    candidate_a1, _ = build_candidate(
        tmp_path / "candidate-a1", generated_at_utc="2026-01-01T00:00:00Z"
    )
    candidate_a2, _ = build_candidate(
        tmp_path / "candidate-a2", generated_at_utc="2026-01-02T00:00:00Z"
    )
    candidate_b, _ = build_candidate(
        tmp_path / "candidate-b",
        ids=(303, 404),
        vectors=[unit_vector(2, 8), unit_vector(3, 8)],
    )
    store = ReleaseStore(tmp_path / "store")

    first = _promote(store, candidate_a1, dataset)
    _promote(store, candidate_b, dataset)
    repromoted = _promote(store, candidate_a2, dataset)

    assert repromoted.release_id == first.release_id
    assert repromoted.activated is True
    assert _pointer(store)["source_manifest_sha256"] == sha256_file(
        candidate_a1 / MANIFEST_FILENAME
    )
    assert store.load_active_release(dataset).release_id == first.release_id


def test_invalid_candidate_is_never_promoted(tmp_path):
    dataset = make_dataset(tmp_path / "dataset")
    candidate, _ = build_candidate(tmp_path / "candidate")
    metadata_path = candidate / "metadata.json"
    metadata_path.write_text(metadata_path.read_text() + "\n", encoding="utf-8")
    store = ReleaseStore(tmp_path / "store")

    with pytest.raises(ReleaseStoreError, match="Candidate artifact validation failed"):
        _promote(store, candidate, dataset, confirm=False)
    with pytest.raises(ReleaseStoreError, match="Candidate artifact validation failed"):
        _promote(store, candidate, dataset, confirm=True)

    assert not store.releases_dir.exists()
    assert not store.pointer_path.exists()


def test_candidate_dataset_paths_are_checked_before_promotion(tmp_path):
    dataset = make_dataset(tmp_path / "dataset")
    candidate, _ = build_candidate(
        tmp_path / "candidate", paths=("nested/cat.jpg", "missing.png")
    )
    store = ReleaseStore(tmp_path / "store")

    with pytest.raises(ReleaseStoreError, match="metadata path validation failed"):
        _promote(store, candidate, dataset, confirm=False)

    assert not store.pointer_path.exists()


def test_staging_failure_leaves_no_release_or_pointer(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path / "dataset")
    candidate, _ = build_candidate(tmp_path / "candidate")
    store = ReleaseStore(tmp_path / "store")
    calls = {"count": 0}
    original = release_store._copy_file

    def fail_second_copy(source, destination):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("simulated copy failure")
        return original(source, destination)

    monkeypatch.setattr(release_store, "_copy_file", fail_second_copy)

    with pytest.raises(OSError, match="simulated copy failure"):
        _promote(store, candidate, dataset)

    assert not store.pointer_path.exists()
    assert not any(
        entry.is_dir() for entry in store.releases_dir.iterdir()
    )


def test_promotion_refuses_corrupt_existing_release(tmp_path):
    dataset = make_dataset(tmp_path / "dataset")
    candidate, _ = build_candidate(tmp_path / "candidate")
    store = ReleaseStore(tmp_path / "store")
    plan = _promote(store, candidate, dataset)

    metadata_path = plan.release_dir / "metadata.json"
    metadata_path.write_text(metadata_path.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ReleaseStoreError, match="Release .* is invalid"):
        _promote(store, candidate, dataset, confirm=False)


def test_load_active_release_returns_verified_runtime_artifacts(tmp_path):
    dataset = make_dataset(tmp_path / "dataset")
    candidate, _ = build_candidate(tmp_path / "candidate")
    store = ReleaseStore(tmp_path / "store")
    plan = _promote(store, candidate, dataset)

    runtime = store.load_active_release(dataset, expected_embedding_dim=8)

    assert runtime.release_id == plan.release_id
    assert runtime.index.ntotal == 2
    assert runtime.metadata == {101: "nested/cat.jpg", 202: "dog.png"}
    assert runtime.path_resolver.image_url("nested/cat.jpg") == "/images/nested/cat.jpg"
    assert runtime.manifest["release"]["release_id"] == plan.release_id


def test_pointer_or_release_corruption_fails_closed(tmp_path):
    dataset = make_dataset(tmp_path / "dataset")
    candidate, _ = build_candidate(tmp_path / "candidate")
    store = ReleaseStore(tmp_path / "store")
    plan = _promote(store, candidate, dataset)

    pointer = _pointer(store)
    pointer["index_sha256"] = "0" * 64
    store.pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(ReleaseStoreError, match="checksum does not match pointer"):
        store.load_active_release(dataset)

    pointer = _pointer(store)
    pointer["index_sha256"] = plan.index_sha256
    store.pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    metadata_path = plan.release_dir / "metadata.json"
    metadata_path.write_text(metadata_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ReleaseStoreError, match="Release .* is invalid"):
        store.load_active_release(dataset)


def test_runtime_rejects_dimension_mismatch_and_missing_dataset_file(tmp_path):
    dataset = make_dataset(tmp_path / "dataset")
    candidate, _ = build_candidate(tmp_path / "candidate")
    store = ReleaseStore(tmp_path / "store")
    _promote(store, candidate, dataset)

    with pytest.raises(ReleaseStoreError, match="expected 16"):
        store.load_active_release(dataset, expected_embedding_dim=16)

    (dataset / "nested" / "cat.jpg").unlink()
    with pytest.raises(ReleaseStoreError, match="path validation failed"):
        store.load_active_release(dataset)


def test_rollback_restores_previous_release_without_deleting_anything(tmp_path):
    dataset = make_dataset(tmp_path / "dataset")
    candidate_a, _ = build_candidate(tmp_path / "candidate-a")
    candidate_b, _ = build_candidate(
        tmp_path / "candidate-b",
        ids=(303, 404),
        vectors=[unit_vector(2, 8), unit_vector(3, 8)],
    )
    store = ReleaseStore(tmp_path / "store")

    first = _promote(store, candidate_a, dataset)
    second = _promote(store, candidate_b, dataset)
    assert second.previous_release_id == first.release_id
    assert store.active_release_id() == second.release_id

    dry_run = store.rollback(first.release_id, confirm=False)
    assert dry_run.dry_run is True
    assert store.active_release_id() == second.release_id

    rolled_back = store.rollback(first.release_id, confirm=True)
    assert rolled_back.activated is True
    assert store.active_release_id() == first.release_id
    assert _pointer(store)["previous_release_id"] == second.release_id
    assert _pointer(store)["operation"] == "rollback"

    assert (first.release_dir / "faiss_index.bin").is_file()
    assert (second.release_dir / "faiss_index.bin").is_file()
    assert store.load_active_release(dataset).release_id == first.release_id


def test_rollback_rejects_invalid_or_unknown_targets(tmp_path):
    dataset = make_dataset(tmp_path / "dataset")
    candidate, _ = build_candidate(tmp_path / "candidate")
    store = ReleaseStore(tmp_path / "store")
    _promote(store, candidate, dataset)

    with pytest.raises(ReleaseStoreError, match="Invalid release ID"):
        store.rollback("../escape", confirm=False)
    with pytest.raises(ReleaseStoreError, match="does not exist"):
        store.rollback("rel-" + "0" * 24, confirm=False)


def test_list_releases_marks_invalid_entries(tmp_path):
    dataset = make_dataset(tmp_path / "dataset")
    candidate, _ = build_candidate(tmp_path / "candidate")
    store = ReleaseStore(tmp_path / "store")
    plan = _promote(store, candidate, dataset)

    listings = store.list_releases()
    assert len(listings) == 1
    assert listings[0].release_id == plan.release_id
    assert listings[0].valid is True
    assert listings[0].info.indexed_images == 2

    (plan.release_dir / "faiss_index.bin").write_bytes(b"corrupt")
    listings = store.list_releases()
    assert listings[0].valid is False
    assert listings[0].errors


def _copy_candidate_with_semantic_change(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    manifest = load_manifest(destination / MANIFEST_FILENAME)
    manifest["embedding"]["model_revision"] = "different-model-revision"
    manifest["embedding"]["normalization"] = "none"
    manifest["embedding"]["preprocessing"]["image_size"] = 512
    write_manifest(manifest, destination / MANIFEST_FILENAME)
    return destination


def test_semantically_different_manifest_does_not_reuse_release(tmp_path):
    dataset = make_dataset(tmp_path / "dataset")
    candidate_a, _ = build_candidate(tmp_path / "candidate-a")
    candidate_b = _copy_candidate_with_semantic_change(
        candidate_a, tmp_path / "candidate-b"
    )
    store = ReleaseStore(tmp_path / "store")

    assert sha256_file(candidate_a / "faiss_index.bin") == sha256_file(
        candidate_b / "faiss_index.bin"
    )
    assert sha256_file(candidate_a / "metadata.json") == sha256_file(
        candidate_b / "metadata.json"
    )

    first = _promote(store, candidate_a, dataset)
    second = _promote(store, candidate_b, dataset)

    assert first.release_id != second.release_id
    assert first.release_dir.is_dir()
    assert second.release_dir.is_dir()
    assert store.active_release_id() == second.release_id
    pointer = _pointer(store)
    assert pointer["semantic_manifest_sha256"] != first.semantic_manifest_sha256
    assert pointer["source_manifest_sha256"] == sha256_file(
        candidate_b / MANIFEST_FILENAME
    )


def test_existing_release_semantic_conflict_is_rejected(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path / "dataset")
    candidate_a, _ = build_candidate(tmp_path / "candidate-a")
    candidate_b = _copy_candidate_with_semantic_change(
        candidate_a, tmp_path / "candidate-b"
    )
    store = ReleaseStore(tmp_path / "store")
    first = _promote(store, candidate_a, dataset)

    monkeypatch.setattr(
        release_store, "release_id_for", lambda *args, **kwargs: first.release_id
    )

    with pytest.raises(ReleaseStoreError, match="semantic manifest conflicts"):
        _promote(store, candidate_b, dataset, confirm=False)

    assert store.active_release_id() == first.release_id
