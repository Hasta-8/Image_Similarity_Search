"""API tests for the search endpoint.

These tests mock the embedding call so they do not require the CLIP model. The
real CLIP + FAISS retrieval path is covered by the integration tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from embeddings import EmbeddingError, EmbeddingModelError
from faiss_index import add_embeddings, create_faiss_index

BACKEND_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture
def main_module(monkeypatch):
    monkeypatch.chdir(BACKEND_DIR)
    import main

    return main


@pytest.fixture
def client(main_module):
    return TestClient(main_module.app)


def _unit_query(dim: int = 512) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    vector[0] = 1.0
    return vector


def test_search_response_contract(client, main_module, monkeypatch):
    query = _unit_query(main_module.embedding_dim)
    index = create_faiss_index(main_module.embedding_dim)
    add_embeddings(index, query.reshape(1, -1), [7])
    main_module.index = index
    # Legacy metadata with an absolute host path: only the basename of a file
    # that exists under the dataset root is exposed, with a relative URL.
    main_module.metadata = {7: "/media/someone/private/_RSB8585.JPG"}
    monkeypatch.setattr(main_module, "get_image_embedding", lambda source: query)

    response = client.post(
        "/search-image",
        files={"file": ("query.jpg", b"fake-image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert list(payload.keys()) == ["results"]
    assert len(payload["results"]) == 1
    result = payload["results"][0]
    assert result["id"] == 7
    assert result["score"] > 0.999
    assert result["path"] == "_RSB8585.JPG"
    assert result["image_url"] == "/images/_RSB8585.JPG"
    assert "/media/" not in response.text
    assert "localhost" not in response.text


def test_legacy_result_missing_file_has_no_image_url(client, main_module, monkeypatch):
    query = _unit_query(main_module.embedding_dim)
    index = create_faiss_index(main_module.embedding_dim)
    add_embeddings(index, query.reshape(1, -1), [7])
    main_module.index = index
    main_module.metadata = {7: "/media/someone/private/does_not_exist.jpg"}
    monkeypatch.setattr(main_module, "get_image_embedding", lambda source: query)

    response = client.post(
        "/search-image",
        files={"file": ("query.jpg", b"fake-image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["path"] == "does_not_exist.jpg"
    assert result["image_url"] is None


def test_invalid_image_returns_400(client, main_module, monkeypatch):
    index = create_faiss_index(main_module.embedding_dim)
    add_embeddings(index, _unit_query().reshape(1, -1), [1])
    main_module.index = index
    main_module.metadata = {1: "image.jpg"}

    def fail_embedding(source):
        raise EmbeddingError("could not decode image")

    monkeypatch.setattr(main_module, "get_image_embedding", fail_embedding)

    response = client.post(
        "/search-image",
        files={"file": ("bad.jpg", b"not an image", "image/jpeg")},
    )

    assert response.status_code == 400
    assert "Invalid or unsupported image" in response.json()["detail"]


def test_embedding_service_failure_returns_503(client, main_module, monkeypatch):
    index = create_faiss_index(main_module.embedding_dim)
    add_embeddings(index, _unit_query().reshape(1, -1), [1])
    main_module.index = index
    main_module.metadata = {1: "image.jpg"}

    def fail_embedding(source):
        raise EmbeddingModelError("model unavailable")

    monkeypatch.setattr(main_module, "get_image_embedding", fail_embedding)

    response = client.post(
        "/search-image",
        files={"file": ("query.jpg", b"fake-image-bytes", "image/jpeg")},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Embedding service unavailable"


def test_empty_index_returns_400(client, main_module):
    main_module.index = create_faiss_index(main_module.embedding_dim)
    main_module.metadata = {}

    response = client.post(
        "/search-image",
        files={"file": ("query.jpg", b"fake-image-bytes", "image/jpeg")},
    )

    assert response.status_code == 400
    assert "No images indexed" in response.json()["detail"]


def test_non_image_content_type_returns_400(client, main_module):
    index = create_faiss_index(main_module.embedding_dim)
    add_embeddings(index, _unit_query().reshape(1, -1), [1])
    main_module.index = index
    main_module.metadata = {1: "image.jpg"}

    response = client.post(
        "/search-image",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert "must be an image" in response.json()["detail"]


def test_empty_file_returns_400(client, main_module):
    index = create_faiss_index(main_module.embedding_dim)
    add_embeddings(index, _unit_query().reshape(1, -1), [1])
    main_module.index = index
    main_module.metadata = {1: "image.jpg"}

    response = client.post(
        "/search-image",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )

    assert response.status_code == 400
    assert "Empty file" in response.json()["detail"]


def test_unexpected_error_returns_500_without_internal_detail(client, main_module, monkeypatch):
    index = create_faiss_index(main_module.embedding_dim)
    add_embeddings(index, _unit_query().reshape(1, -1), [1])
    main_module.index = index
    main_module.metadata = {1: "image.jpg"}

    def fail_embedding(source):
        raise RuntimeError("secret internal failure")

    monkeypatch.setattr(main_module, "get_image_embedding", fail_embedding)

    response = client.post(
        "/search-image",
        files={"file": ("query.jpg", b"fake-image-bytes", "image/jpeg")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Error processing search request"


def test_search_helper_uses_shared_service_output(main_module, monkeypatch):
    vector = np.full(main_module.embedding_dim, 1.0 / np.sqrt(512), dtype=np.float32)
    monkeypatch.setattr(main_module, "generate_embedding", lambda source: vector)

    embedding = main_module.get_image_embedding(b"ignored")

    assert embedding.shape == (main_module.embedding_dim,)
    assert embedding.dtype == np.float32
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)
