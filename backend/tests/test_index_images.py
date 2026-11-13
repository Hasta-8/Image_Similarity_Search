"""Tests for the indexing workflow using deterministic mocked embeddings.

These tests validate batching, ordering, ID assignment, and corrupt-image
handling without loading the real model. Real CLIP inference is covered by
``test_embeddings.py`` and ``test_retrieval_integration.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import index_images as indexing
from embeddings import EmbeddingError
from faiss_index import load_faiss_index, load_metadata, search_index


def _make_images(folder: Path, count: int = 3) -> list[str]:
    paths = []
    for index in range(count):
        image = Image.new("RGB", (32, 32), color=(index * 40, 100, 200 - index * 40))
        path = folder / f"image_{index}.png"
        image.save(path, format="PNG")
        paths.append(str(path))
    return paths


def _unit_vector(index: int, dim: int = 512) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    vector[index % dim] = 1.0
    return vector


def test_index_images_batches_preserves_order_and_ids(tmp_path, monkeypatch):
    image_paths = _make_images(tmp_path, count=3)
    index_path = tmp_path / "test_index.bin"
    metadata_path = tmp_path / "test_metadata.json"
    calls = []

    def fake_batch(paths, batch_size=None):
        calls.append((list(paths), batch_size))
        return np.stack(
            [_unit_vector(int(Path(path).stem.split("_")[-1])) for path in paths]
        )

    def fake_single(path):
        assert False, "single-image fallback should not be used on a valid batch"

    monkeypatch.setattr(indexing, "generate_embeddings_batch", fake_batch)
    monkeypatch.setattr(indexing, "generate_embedding", fake_single)

    indexing.index_images(
        folder_path=str(tmp_path),
        index_path=str(index_path),
        meta_path=str(metadata_path),
        embedding_dim=512,
        batch_size=2,
    )

    assert [len(paths) for paths, _ in calls] == [2, 1]
    assert all(batch_size == 2 for _, batch_size in calls)

    index = load_faiss_index(str(index_path))
    metadata = load_metadata(str(metadata_path))

    assert index.ntotal == 3
    assert sorted(metadata.keys()) == [0, 1, 2]
    assert [metadata[i] for i in range(3)] == [
        str(Path(path).resolve()) for path in image_paths
    ]

    for expected_id, path in enumerate(image_paths):
        embedding = _unit_vector(int(Path(path).stem.split("_")[-1]))
        results = search_index(index, embedding, top_k=1)
        assert results[0][0] == expected_id
        assert results[0][1] > 0.999


def test_index_images_skips_corrupt_images_and_continues(tmp_path, monkeypatch):
    good_one = tmp_path / "a_good.png"
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(good_one, format="PNG")
    bad = tmp_path / "b_bad.png"
    bad.write_bytes(b"not an image")
    good_two = tmp_path / "c_good.png"
    Image.new("RGB", (32, 32), color=(0, 0, 255)).save(good_two, format="PNG")

    index_path = tmp_path / "test_index.bin"
    metadata_path = tmp_path / "test_metadata.json"

    def fake_batch(paths, batch_size=None):
        if any(Path(path).name == "b_bad.png" for path in paths):
            raise EmbeddingError("simulated corrupt image in batch")
        return np.stack([_unit_vector(0) for _ in paths])

    def fake_single(path):
        if Path(path).name == "b_bad.png":
            raise EmbeddingError("simulated corrupt image")
        return _unit_vector(0)

    monkeypatch.setattr(indexing, "generate_embeddings_batch", fake_batch)
    monkeypatch.setattr(indexing, "generate_embedding", fake_single)

    indexing.index_images(
        folder_path=str(tmp_path),
        index_path=str(index_path),
        meta_path=str(metadata_path),
        embedding_dim=512,
        batch_size=2,
    )

    index = load_faiss_index(str(index_path))
    metadata = load_metadata(str(metadata_path))

    assert index.ntotal == 2
    assert len(metadata) == 2
    assert set(Path(path).name for path in metadata.values()) == {
        "a_good.png",
        "c_good.png",
    }


def test_index_images_rejects_dimension_other_than_clip(tmp_path):
    with pytest.raises(ValueError, match="512"):
        indexing.index_images(
            folder_path=str(tmp_path),
            index_path=str(tmp_path / "unused.bin"),
            meta_path=str(tmp_path / "unused.json"),
            embedding_dim=256,
        )


def test_find_image_files_is_recursive_and_extension_aware(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "one.jpg", format="JPEG")
    Image.new("RGB", (8, 8)).save(nested / "two.png", format="PNG")
    (tmp_path / "notes.txt").write_text("ignore me")

    found = indexing.find_image_files(str(tmp_path))

    assert sorted(Path(path).name for path in found) == ["one.jpg", "two.png"]
