"""Shared deterministic artifact fixtures for Phase 3 tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from artifact_manifest import (
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    METADATA_FILENAME,
    build_manifest,
    write_manifest,
)
from faiss_index import add_embeddings, create_faiss_index, save_faiss_index, save_metadata


def make_dataset(
    root: Path,
    paths: tuple[str, ...] = ("nested/cat.jpg", "dog.png"),
) -> Path:
    for index, relative_path in enumerate(paths):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), color=(index * 20, 30, 40)).save(path)
    return root


def unit_vector(index: int, dim: int = 8) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    vector[index % dim] = 1.0
    return vector


def build_candidate(
    directory: Path,
    *,
    dim: int = 8,
    ids: tuple[int, ...] = (101, 202),
    paths: tuple[str, ...] = ("nested/cat.jpg", "dog.png"),
    vectors: np.ndarray | None = None,
    generated_at_utc: str = "2026-01-01T00:00:00Z",
) -> tuple[Path, dict]:
    directory.mkdir()
    if vectors is None:
        vectors = np.stack([unit_vector(index, dim) for index in range(len(ids))])
    vectors = np.asarray(vectors, dtype=np.float32)

    index = create_faiss_index(dim)
    add_embeddings(index, vectors, list(ids))
    save_faiss_index(index, str(directory / INDEX_FILENAME))
    save_metadata(dict(zip(ids, paths)), str(directory / METADATA_FILENAME))

    manifest = build_manifest(
        artifact_dir=directory,
        embedding={
            "model_name": "test-model",
            "model_revision": "test-revision",
            "embedding_dim": dim,
            "normalization": "l2",
            "device": "cpu",
            "preprocessing": {"input_mode": "RGB", "image_size": 224},
        },
        dataset={
            "identifier": "test-dataset",
            "root": "test-dataset",
            "fingerprint": "a" * 64,
            "fingerprint_algorithm": "sha256-path-and-content-v1",
            "file_count": len(ids),
            "total_bytes": 1234,
        },
        id_strategy={
            "name": "sha256-dataset-relative-path",
            "version": 1,
            "identity": "normalized dataset-relative path",
            "id_bits": 63,
        },
        indexing={"mode": "full_candidate_rebuild", "batch_size": 2, "device": "cpu"},
        indexed_images=len(ids),
        failed_images=0,
        metadata_records=len(ids),
        failures=[],
        faiss_type=type(index).__name__,
        generated_at_utc=generated_at_utc,
    )
    write_manifest(manifest, directory / MANIFEST_FILENAME)
    return directory, manifest
