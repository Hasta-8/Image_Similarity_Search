"""Phase 3.1 tests using copies of the real candidate artifacts.

These tests never touch the production release store, production artifacts, or
the real candidate directory. All promotion and rollback operations happen in
``tmp_path``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import faiss
import numpy as np
import pytest
from fastapi.testclient import TestClient

import main
import rollback_release
from artifact_manifest import (
    MANIFEST_FILENAME,
    load_manifest,
    semantic_manifest_sha256,
    write_manifest,
)
from artifact_validation import validate_candidate_artifacts, validate_release_artifacts
from dataset_fingerprint import discover_dataset_images, sha256_file
from embeddings import generate_embeddings_batch
from faiss_index import load_metadata, save_metadata
from release_store import ReleaseStore
from stable_ids import build_stable_ids

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
CANDIDATE_DIR = BACKEND_DIR / "candidate_index"
DATASET_DIR = REPO_ROOT / "dataset"
PRODUCTION_ARTIFACTS = (
    BACKEND_DIR / "faiss_index.bin",
    BACKEND_DIR / "metadata.json",
)
EXPECTED_IMAGE_COUNT = 70
EXPECTED_DIMENSION = 512

pytestmark = pytest.mark.skipif(
    not CANDIDATE_DIR.is_dir() or not DATASET_DIR.is_dir(),
    reason="Real candidate artifacts or dataset are not available",
)


def _hashes(paths) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths}


def _reset_runtime(monkeypatch, store: ReleaseStore, dataset: Path) -> None:
    monkeypatch.setattr(main, "release_store", store)
    monkeypatch.setattr(main, "DATASET_DIR", dataset)
    monkeypatch.setattr(main, "index", None)
    monkeypatch.setattr(main, "metadata", {})
    monkeypatch.setattr(main, "active_release_id", None)
    monkeypatch.setattr(main, "runtime_mode", "unavailable")
    monkeypatch.setattr(main, "runtime_error", None)
    monkeypatch.setattr(main, "path_resolver", None)
    monkeypatch.setattr(main, "embedding_dim", EXPECTED_DIMENSION)


def _copy_candidate(tmp_path: Path, name: str = "candidate") -> Path:
    destination = tmp_path / name
    shutil.copytree(CANDIDATE_DIR, destination)
    return destination


def _candidate_files(candidate: Path):
    return (
        candidate / "faiss_index.bin",
        candidate / "metadata.json",
        candidate / MANIFEST_FILENAME,
    )


def test_real_candidate_confirmed_promotion_retrieval_and_readiness(
    tmp_path, monkeypatch
):
    before = _hashes(_candidate_files(CANDIDATE_DIR) + PRODUCTION_ARTIFACTS)
    candidate = _copy_candidate(tmp_path)
    copied_hashes = _hashes(_candidate_files(candidate))

    validation = validate_candidate_artifacts(
        candidate,
        expected_embedding_dim=EXPECTED_DIMENSION,
        expected_count=EXPECTED_IMAGE_COUNT,
    )
    assert validation.valid, validation.errors

    store = ReleaseStore(tmp_path / "release-store")
    plan = store.promote(candidate, confirm=True, dataset_root=DATASET_DIR)

    assert plan.activated is True
    assert plan.release_id.startswith("rel-")
    assert plan.release_dir.is_dir()
    assert sorted(path.name for path in plan.release_dir.iterdir()) == [
        "faiss_index.bin",
        "manifest.json",
        "metadata.json",
    ]

    assert sha256_file(plan.release_dir / "faiss_index.bin") == copied_hashes[
        str(candidate / "faiss_index.bin")
    ]
    assert sha256_file(plan.release_dir / "metadata.json") == copied_hashes[
        str(candidate / "metadata.json")
    ]
    release_manifest = load_manifest(plan.release_dir / MANIFEST_FILENAME)
    assert release_manifest["release"]["release_id"] == plan.release_id
    assert release_manifest["release"]["source_manifest_sha256"] == copied_hashes[
        str(candidate / MANIFEST_FILENAME)
    ]
    assert release_manifest["release"]["semantic_manifest_sha256"] == (
        semantic_manifest_sha256(release_manifest)
    )

    release_validation = validate_release_artifacts(
        plan.release_dir,
        expected_embedding_dim=EXPECTED_DIMENSION,
        expected_count=EXPECTED_IMAGE_COUNT,
    )
    assert release_validation.valid, release_validation.errors
    assert release_validation.details["index_dimension"] == EXPECTED_DIMENSION
    assert release_validation.details["index_ntotal"] == EXPECTED_IMAGE_COUNT
    assert release_validation.details["index_id_count"] == EXPECTED_IMAGE_COUNT
    assert release_validation.details["metadata_count"] == EXPECTED_IMAGE_COUNT

    runtime = store.load_active_release(
        DATASET_DIR, expected_embedding_dim=EXPECTED_DIMENSION
    )
    assert runtime.release_id == plan.release_id
    assert runtime.index.d == EXPECTED_DIMENSION
    assert runtime.index.ntotal == EXPECTED_IMAGE_COUNT
    assert len(runtime.metadata) == EXPECTED_IMAGE_COUNT

    index_ids = {int(value) for value in faiss.vector_to_array(runtime.index.id_map)}
    assert index_ids == set(runtime.metadata)
    assert len(index_ids) == EXPECTED_IMAGE_COUNT

    discovered = discover_dataset_images(DATASET_DIR)
    relative_paths = [relative for relative, _ in discovered]
    assert len(relative_paths) == EXPECTED_IMAGE_COUNT
    stable_ids = build_stable_ids(relative_paths)
    assert runtime.metadata == {image_id: path for path, image_id in stable_ids.items()}

    for relative_path in relative_paths:
        resolved = runtime.path_resolver.resolve(relative_path, must_exist=True)
        assert resolved.is_file()
        image_url = runtime.path_resolver.image_url(relative_path)
        assert image_url.startswith("/images/")
        assert str(DATASET_DIR) not in image_url

    vectors = generate_embeddings_batch(
        [absolute for _, absolute in discovered], batch_size=16
    )
    top1_ids = []
    top1_scores = []
    for vector, relative_path in zip(vectors, relative_paths):
        results = main.search_index(runtime.index, vector, top_k=1)
        assert len(results) == 1
        image_id, score = results[0]
        top1_ids.append(image_id)
        top1_scores.append(float(score))
        assert runtime.metadata[image_id] == relative_path

    assert top1_ids == [stable_ids[path] for path in relative_paths]
    assert len(top1_ids) == EXPECTED_IMAGE_COUNT
    assert all(score >= 0.999999 for score in top1_scores)
    assert np.isclose(min(top1_scores), 1.0, atol=1e-6)
    assert np.isclose(max(top1_scores), 1.0, atol=1e-6)

    _reset_runtime(monkeypatch, store, DATASET_DIR)
    main.load_index_and_metadata()
    client = TestClient(main.app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["runtime_mode"] == "release"
    assert health.json()["ready"] is True
    assert health.json()["active_release_id"] == plan.release_id

    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json() == {
        "status": "ready",
        "runtime_mode": "release",
        "active_release_id": plan.release_id,
        "indexed_images": EXPECTED_IMAGE_COUNT,
        "metadata_count": EXPECTED_IMAGE_COUNT,
    }

    after = _hashes(_candidate_files(CANDIDATE_DIR) + PRODUCTION_ARTIFACTS)
    assert after == before
    assert _hashes(_candidate_files(candidate)) == copied_hashes


def test_real_candidate_rollback_using_cli_preserves_immutable_releases(
    tmp_path, monkeypatch
):
    before = _hashes(_candidate_files(CANDIDATE_DIR) + PRODUCTION_ARTIFACTS)
    first_candidate = _copy_candidate(tmp_path, "candidate-first")
    second_candidate = _copy_candidate(tmp_path, "candidate-second")

    metadata = load_metadata(str(second_candidate / "metadata.json"))
    reordered = dict(reversed(list(metadata.items())))
    save_metadata(reordered, str(second_candidate / "metadata.json"))
    manifest = load_manifest(second_candidate / MANIFEST_FILENAME)
    manifest["index"]["metadata_sha256"] = sha256_file(
        second_candidate / "metadata.json"
    )
    manifest["index"]["metadata_size_bytes"] = (
        second_candidate / "metadata.json"
    ).stat().st_size
    write_manifest(manifest, second_candidate / MANIFEST_FILENAME)

    second_validation = validate_candidate_artifacts(
        second_candidate,
        expected_embedding_dim=EXPECTED_DIMENSION,
        expected_count=EXPECTED_IMAGE_COUNT,
    )
    assert second_validation.valid, second_validation.errors
    assert sha256_file(first_candidate / "faiss_index.bin") == sha256_file(
        second_candidate / "faiss_index.bin"
    )
    assert sha256_file(first_candidate / "metadata.json") != sha256_file(
        second_candidate / "metadata.json"
    )

    store = ReleaseStore(tmp_path / "release-store")
    first = store.promote(first_candidate, confirm=True, dataset_root=DATASET_DIR)
    second = store.promote(second_candidate, confirm=True, dataset_root=DATASET_DIR)

    assert first.release_id != second.release_id
    assert store.active_release_id() == second.release_id
    assert first.release_dir.is_dir()
    assert second.release_dir.is_dir()

    first_hashes = _hashes(_candidate_files(first.release_dir))
    second_hashes = _hashes(_candidate_files(second.release_dir))

    exit_code = rollback_release.main(
        [
            first.release_id,
            "--release-store",
            str(store.root),
            "--confirm",
        ]
    )
    assert exit_code == 0
    assert store.active_release_id() == first.release_id

    pointer = json.loads(store.pointer_path.read_text(encoding="utf-8"))
    assert pointer["operation"] == "rollback"
    assert pointer["previous_release_id"] == second.release_id
    assert pointer["semantic_manifest_sha256"] == first.semantic_manifest_sha256

    assert first.release_dir.is_dir()
    assert second.release_dir.is_dir()
    assert _hashes(_candidate_files(first.release_dir)) == first_hashes
    assert _hashes(_candidate_files(second.release_dir)) == second_hashes
    assert validate_release_artifacts(first.release_dir).valid
    assert validate_release_artifacts(second.release_dir).valid

    runtime = store.load_active_release(
        DATASET_DIR, expected_embedding_dim=EXPECTED_DIMENSION
    )
    assert runtime.release_id == first.release_id
    assert runtime.metadata == load_metadata(str(first_candidate / "metadata.json"))

    after = _hashes(_candidate_files(CANDIDATE_DIR) + PRODUCTION_ARTIFACTS)
    assert after == before
