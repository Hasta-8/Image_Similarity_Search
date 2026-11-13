"""Real CLIP + FAISS retrieval integration test.

This uses a temporary in-memory/temporary-directory index and never touches the
production ``backend/faiss_index.bin`` or ``backend/metadata.json``.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from embeddings import DEFAULT_EMBEDDING_DIM, generate_embedding
from faiss_index import add_embeddings, create_faiss_index, search_index


@pytest.fixture(scope="session")
def image_paths(tmp_path_factory):
    directory = tmp_path_factory.mktemp("retrieval_images")
    paths = []
    for index in range(2):
        rng = np.random.default_rng(100 + index)
        pixels = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
        image = Image.fromarray(pixels, mode="RGB")
        ImageDraw.Draw(image).rectangle(
            (5, 5, 40 + index * 10, 40), outline=(255, 255, 0), width=3
        )
        path = directory / f"image_{index}.png"
        image.save(path, format="PNG")
        paths.append(str(path))
    return paths


@pytest.fixture(scope="session")
def retrieval_data(image_paths):
    embeddings = [generate_embedding(path) for path in image_paths]
    return image_paths, np.stack(embeddings).astype(np.float32)


def test_exact_image_is_rank_one(retrieval_data):
    image_paths, embeddings = retrieval_data

    index = create_faiss_index(DEFAULT_EMBEDDING_DIM)
    add_embeddings(index, embeddings, [10, 20])

    query = generate_embedding(image_paths[0])
    results = search_index(index, query, top_k=2)

    assert results, "Expected at least one search result"
    top_id, top_score = results[0]
    assert top_id == 10
    assert top_score > 0.999
    assert results[1][0] == 20


def test_exact_image_bytes_match_indexed_path(retrieval_data, image_paths):
    _, embeddings = retrieval_data
    index = create_faiss_index(DEFAULT_EMBEDDING_DIM)
    add_embeddings(index, embeddings, [10, 20])

    with open(image_paths[1], "rb") as handle:
        query_bytes = handle.read()
    query = generate_embedding(query_bytes)
    results = search_index(index, query, top_k=2)

    assert results[0][0] == 20
    assert results[0][1] > 0.999
