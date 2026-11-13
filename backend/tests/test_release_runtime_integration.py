"""Promotion -> runtime load -> search -> rollback integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main
from faiss_index import add_embeddings, create_faiss_index, save_faiss_index, save_metadata
from release_store import ReleaseStore
from tests.artifact_fixtures import build_candidate, make_dataset, unit_vector


def _reset_runtime(monkeypatch, store: ReleaseStore, dataset: Path) -> None:
    monkeypatch.setattr(main, "release_store", store)
    monkeypatch.setattr(main, "DATASET_DIR", dataset)
    monkeypatch.setattr(main, "index", None)
    monkeypatch.setattr(main, "metadata", {})
    monkeypatch.setattr(main, "active_release_id", None)
    monkeypatch.setattr(main, "runtime_mode", "unavailable")
    monkeypatch.setattr(main, "runtime_error", None)
    monkeypatch.setattr(main, "path_resolver", None)
    monkeypatch.setattr(main, "embedding_dim", 8)


def test_api_loads_promoted_release_and_returns_safe_relative_paths(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path / "dataset")
    candidate, _ = build_candidate(tmp_path / "candidate")
    store = ReleaseStore(tmp_path / "store")
    plan = store.promote(candidate, confirm=True, dataset_root=dataset)
    _reset_runtime(monkeypatch, store, dataset)
    monkeypatch.setattr(main, "get_image_embedding", lambda source: unit_vector(0))

    main.load_index_and_metadata()
    client = TestClient(main.app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["ready"] is True
    assert health.json()["runtime_mode"] == "release"
    assert health.json()["active_release_id"] == plan.release_id

    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json() == {
        "status": "ready",
        "runtime_mode": "release",
        "active_release_id": plan.release_id,
        "indexed_images": 2,
        "metadata_count": 2,
    }

    response = client.post(
        "/search-image",
        files={"file": ("query.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["id"] == 101
    assert result["path"] == "nested/cat.jpg"
    assert result["image_url"] == "/images/nested/cat.jpg"
    assert str(dataset) not in str(response.json())


def test_legacy_fallback_is_live_but_not_retrieval_ready(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path / "dataset")
    legacy_index_path = tmp_path / "legacy" / "faiss_index.bin"
    legacy_index_path.parent.mkdir()
    legacy_metadata_path = legacy_index_path.parent / "metadata.json"
    index = create_faiss_index(8)
    add_embeddings(index, unit_vector(0).reshape(1, -1), [1])
    save_faiss_index(index, str(legacy_index_path))
    save_metadata({1: "nested/cat.jpg"}, str(legacy_metadata_path))

    store = ReleaseStore(tmp_path / "store")
    _reset_runtime(monkeypatch, store, dataset)
    monkeypatch.setattr(main, "index_path", str(legacy_index_path))
    monkeypatch.setattr(main, "metadata_path", str(legacy_metadata_path))
    client = TestClient(main.app)

    main.load_index_and_metadata()
    assert main.runtime_mode == "legacy"
    assert main.index is not None

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["index_loaded"] is True
    assert health.json()["ready"] is False
    assert health.json()["runtime_mode"] == "legacy"

    ready = client.get("/ready")
    assert ready.status_code == 503
    assert ready.json()["status"] == "not_ready"
    assert ready.json()["runtime_mode"] == "legacy"
    assert "legacy fallback" in ready.json()["error"]
    assert not store.pointer_path.exists()


def test_runtime_fails_closed_when_active_release_is_corrupt(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path / "dataset")
    candidate, _ = build_candidate(tmp_path / "candidate")
    store = ReleaseStore(tmp_path / "store")
    plan = store.promote(candidate, confirm=True, dataset_root=dataset)
    metadata_path = plan.release_dir / "metadata.json"
    metadata_path.write_text(metadata_path.read_text() + "\n", encoding="utf-8")
    _reset_runtime(monkeypatch, store, dataset)
    client = TestClient(main.app)

    with pytest.raises(Exception):
        main.load_index_and_metadata()

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["ready"] is False
    assert health.json()["runtime_mode"] == "unavailable"

    ready = client.get("/ready")
    assert ready.status_code == 503
    assert ready.json()["status"] == "not_ready"

    response = client.post(
        "/search-image",
        files={"file": ("query.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "Artifact runtime unavailable"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model_revision": "0" * 40}, "model revision"),
        ({"model_revision": None}, "model revision"),
        ({"model_name": "openai/clip-vit-large-patch14"}, "built with model"),
    ],
)
def test_runtime_fails_closed_when_release_model_differs_from_query_encoder(
    tmp_path, monkeypatch, overrides, message
):
    dataset = make_dataset(tmp_path / "dataset")
    candidate, _ = build_candidate(tmp_path / "candidate", **overrides)
    store = ReleaseStore(tmp_path / "store")
    store.promote(candidate, confirm=True, dataset_root=dataset)
    _reset_runtime(monkeypatch, store, dataset)
    client = TestClient(main.app)

    with pytest.raises(RuntimeError, match=message):
        main.load_index_and_metadata()

    assert main.runtime_mode == "unavailable"
    assert main.index is None
    assert message in main.runtime_error

    ready = client.get("/ready")
    assert ready.status_code == 503
    assert message in ready.json()["error"]

    response = client.post(
        "/search-image",
        files={"file": ("query.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert response.status_code == 503


def test_runtime_rollback_reloads_previous_release(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path / "dataset")
    candidate_a, _ = build_candidate(tmp_path / "candidate-a")
    candidate_b, _ = build_candidate(
        tmp_path / "candidate-b",
        ids=(303, 404),
        vectors=[unit_vector(2, 8), unit_vector(3, 8)],
    )
    store = ReleaseStore(tmp_path / "store")
    first = store.promote(candidate_a, confirm=True, dataset_root=dataset)
    second = store.promote(candidate_b, confirm=True, dataset_root=dataset)
    _reset_runtime(monkeypatch, store, dataset)

    main.load_index_and_metadata()
    assert main.active_release_id == second.release_id
    assert set(main.metadata) == {303, 404}

    store.rollback(first.release_id, confirm=True)
    main.load_index_and_metadata()
    assert main.active_release_id == first.release_id
    assert set(main.metadata) == {101, 202}

    monkeypatch.setattr(main, "get_image_embedding", lambda source: unit_vector(1))
    client = TestClient(main.app)
    response = client.post(
        "/search-image",
        files={"file": ("query.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["id"] == 202
    assert response.json()["results"][0]["path"] == "dog.png"


def test_legacy_fallback_search_uses_legacy_contract_but_ready_stays_false(
    tmp_path, monkeypatch
):
    dataset = make_dataset(tmp_path / "dataset")
    legacy_index_path = tmp_path / "legacy" / "faiss_index.bin"
    legacy_index_path.parent.mkdir()
    legacy_metadata_path = legacy_index_path.parent / "metadata.json"
    index = create_faiss_index(8)
    add_embeddings(index, unit_vector(0).reshape(1, -1), [1])
    save_faiss_index(index, str(legacy_index_path))
    save_metadata({1: "nested/cat.jpg"}, str(legacy_metadata_path))

    store = ReleaseStore(tmp_path / "store")
    _reset_runtime(monkeypatch, store, dataset)
    monkeypatch.setattr(main, "index_path", str(legacy_index_path))
    monkeypatch.setattr(main, "metadata_path", str(legacy_metadata_path))
    monkeypatch.setattr(main, "get_image_embedding", lambda source: unit_vector(0))
    main.load_index_and_metadata()
    client = TestClient(main.app)

    ready = client.get("/ready")
    assert ready.status_code == 503
    assert ready.json()["runtime_mode"] == "legacy"
    assert ready.json()["status"] == "not_ready"

    response = client.post(
        "/search-image",
        files={"file": ("query.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["id"] == 1
    assert result["path"] == "nested/cat.jpg"
    # Legacy URLs are relative and never hardcode a host or port.
    assert result["image_url"] == "/images/nested/cat.jpg"
