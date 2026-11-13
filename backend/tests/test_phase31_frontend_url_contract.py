"""API URL contract for frontend image rendering.

The React app renders ``result.image_url`` directly. This test verifies that the
API returns relative, encoded URLs that preserve nested paths and duplicate
basenames when a validated release is active.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import main
from artifact_paths import DatasetPathResolver
from release_store import ReleaseStore
from tests.artifact_fixtures import build_candidate, make_dataset, unit_vector

PATHS = (
    "gallery-a/photo.jpg",
    "gallery-b/photo.jpg",
    "nested dir/cat one.jpg",
    "café.jpg",
)


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


def test_release_api_returns_encoded_nested_urls_without_basename_collisions(
    tmp_path, monkeypatch
):
    dataset = make_dataset(tmp_path / "dataset", paths=PATHS)
    candidate, _ = build_candidate(
        tmp_path / "candidate",
        ids=(101, 202, 303, 404),
        paths=PATHS,
        vectors=[unit_vector(index) for index in range(len(PATHS))],
    )
    store = ReleaseStore(tmp_path / "store")
    plan = store.promote(candidate, confirm=True, dataset_root=dataset)

    _reset_runtime(monkeypatch, store, dataset)
    monkeypatch.setattr(main, "get_image_embedding", lambda source: unit_vector(0))
    main.load_index_and_metadata()

    client = TestClient(main.app)
    response = client.post(
        "/search-image",
        files={"file": ("query.jpg", b"fake-image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    results = {item["id"]: item for item in response.json()["results"]}
    assert set(results) == {101, 202, 303, 404}

    resolver = DatasetPathResolver(dataset)
    expected_urls = {image_id: resolver.image_url(path) for image_id, path in zip(
        (101, 202, 303, 404), PATHS
    )}
    for image_id, relative_path in zip((101, 202, 303, 404), PATHS):
        assert results[image_id]["path"] == relative_path
        assert results[image_id]["image_url"] == expected_urls[image_id]
        assert results[image_id]["image_url"].startswith("/images/")
        assert str(dataset) not in results[image_id]["image_url"]
        assert resolver.resolve(relative_path, must_exist=True).is_file()

    assert results[101]["image_url"] != results[202]["image_url"]
    assert results[101]["image_url"].endswith("/gallery-a/photo.jpg")
    assert results[202]["image_url"].endswith("/gallery-b/photo.jpg")
    assert "%20" in results[303]["image_url"]
    assert "caf%C3%A9.jpg" in results[404]["image_url"]

    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["active_release_id"] == plan.release_id
