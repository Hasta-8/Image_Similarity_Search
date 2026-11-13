"""Tests for deterministic candidate rebuild orchestration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import rebuild_index
from artifact_manifest import INDEX_FILENAME, MANIFEST_FILENAME, METADATA_FILENAME
from artifact_validation import ValidationResult, validate_candidate_artifacts
from dataset_fingerprint import sha256_file
from embeddings import EmbeddingError, EmbeddingModelError
from faiss_index import load_faiss_index, load_metadata, search_index
from stable_ids import build_stable_ids


def _write_image(root: Path, relative_path: str, color: tuple[int, int, int]) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)
    return path


def _embedding_metadata(dim: int = 8) -> dict:
    return {
        "model_name": "test-model",
        "model_revision": "test-revision",
        "embedding_dim": dim,
        "normalization": "l2",
        "device": "cpu",
        "preprocessing": {"input_mode": "RGB", "image_size": 224},
    }


def _vector_for(path: str | Path, dataset: Path, dim: int = 8) -> np.ndarray:
    relative = Path(path).resolve().relative_to(dataset.resolve()).as_posix()
    digest = hashlib.sha256(relative.encode("utf-8")).digest()
    values = np.frombuffer(digest * ((dim + len(digest) - 1) // len(digest)), dtype=np.uint8)
    vector = values[:dim].astype(np.float32) + 1.0
    return vector / np.linalg.norm(vector)


def _fake_embedders(dataset: Path, dim: int = 8):
    def fake_batch(paths, batch_size=None):
        return np.stack([_vector_for(path, dataset, dim) for path in paths])

    def fake_single(path):
        return _vector_for(path, dataset, dim)

    return fake_batch, fake_single


def _rebuild(dataset: Path, output: Path, **kwargs):
    fake_batch, fake_single = _fake_embedders(dataset)
    batch_size = kwargs.pop("batch_size", 2)
    return rebuild_index.rebuild_candidate_index(
        dataset_dir=dataset,
        output_dir=output,
        batch_size=batch_size,
        embed_batch=fake_batch,
        embed_single=fake_single,
        embedding_metadata=_embedding_metadata(),
        generated_at_utc="2026-01-01T00:00:00Z",
        **kwargs,
    )


def test_rebuild_creates_valid_candidate_with_relative_stable_ids(tmp_path):
    dataset = tmp_path / "dataset"
    _write_image(dataset, "a.jpg", (255, 0, 0))
    _write_image(dataset, "nested/c.png", (0, 255, 0))
    _write_image(dataset, "b.png", (0, 0, 255))
    output = tmp_path / "candidate"

    production_index = rebuild_index.PRODUCTION_INDEX_PATH
    production_metadata = rebuild_index.PRODUCTION_METADATA_PATH
    before_index = sha256_file(production_index)
    before_metadata = sha256_file(production_metadata)

    result = _rebuild(dataset, output)

    assert result.output_dir == output.resolve()
    assert result.index_path == output / INDEX_FILENAME
    assert result.metadata_path == output / METADATA_FILENAME
    assert result.manifest_path == output / MANIFEST_FILENAME
    assert result.indexed_images == 3
    assert result.failed_images == 0
    assert result.validation.valid, result.validation.errors

    metadata = load_metadata(str(result.metadata_path))
    expected_ids = build_stable_ids(["a.jpg", "nested/c.png", "b.png"])
    assert metadata == {expected_ids[path]: path for path in expected_ids}
    assert all(not Path(path).is_absolute() for path in metadata.values())

    index = load_faiss_index(str(result.index_path))
    assert index.ntotal == 3
    assert set(metadata) == set(expected_ids.values())

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_role"] == "candidate"
    assert manifest["promotion_status"] == "not_promoted"
    assert manifest["schema_version"] == "1.0"
    assert manifest["index"]["indexed_images"] == 3
    assert manifest["index"]["failed_images"] == 0
    assert manifest["index"]["metadata_records"] == 3
    assert manifest["failures"] == []
    assert manifest["dataset"]["root"] == "dataset"
    assert manifest["embedding"]["model_revision"] == "test-revision"
    assert str(dataset.resolve()) not in result.manifest_path.read_text(encoding="utf-8")

    assert sha256_file(production_index) == before_index
    assert sha256_file(production_metadata) == before_metadata


def test_rebuild_is_deterministic_for_same_dataset(tmp_path):
    dataset = tmp_path / "dataset"
    _write_image(dataset, "a.jpg", (255, 0, 0))
    _write_image(dataset, "b.png", (0, 0, 255))

    first = _rebuild(dataset, tmp_path / "candidate-one")
    second = _rebuild(dataset, tmp_path / "candidate-two")

    assert first.dataset_fingerprint == second.dataset_fingerprint
    assert sha256_file(first.index_path) == sha256_file(second.index_path)
    assert sha256_file(first.metadata_path) == sha256_file(second.metadata_path)
    assert sha256_file(first.manifest_path) == sha256_file(second.manifest_path)


def test_rebuild_skips_corrupt_images_with_explicit_failure(tmp_path):
    dataset = tmp_path / "dataset"
    _write_image(dataset, "a.png", (255, 0, 0))
    bad = dataset / "b.png"
    bad.write_bytes(b"not an image")
    _write_image(dataset, "c.png", (0, 0, 255))

    def fake_batch(paths, batch_size=None):
        if any(Path(path).name == "b.png" for path in paths):
            raise EmbeddingError("simulated batch failure")
        return np.stack([_vector_for(path, dataset) for path in paths])

    def fake_single(path):
        if Path(path).name == "b.png":
            raise EmbeddingError("simulated corrupt image")
        return _vector_for(path, dataset)

    result = rebuild_index.rebuild_candidate_index(
        dataset_dir=dataset,
        output_dir=tmp_path / "candidate",
        batch_size=2,
        embed_batch=fake_batch,
        embed_single=fake_single,
        embedding_metadata=_embedding_metadata(),
        generated_at_utc="2026-01-01T00:00:00Z",
    )

    assert result.indexed_images == 2
    assert result.failed_images == 1
    assert result.validation.valid, result.validation.errors

    metadata = load_metadata(str(result.metadata_path))
    assert set(metadata.values()) == {"a.png", "c.png"}

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset"]["file_count"] == 3
    assert manifest["index"]["indexed_images"] == 2
    assert manifest["index"]["failed_images"] == 1
    assert manifest["failures"] == [
        {
            "path": "b.png",
            "stage": "embedding",
            "error_type": "EmbeddingError",
            "message": "simulated corrupt image",
        }
    ]


def test_rebuild_fails_without_publishing_when_all_images_fail(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset).mkdir()
    (dataset / "bad.png").write_bytes(b"not an image")
    output = tmp_path / "candidate"

    def fake_batch(paths, batch_size=None):
        raise EmbeddingError("batch failed")

    def fake_single(path):
        raise EmbeddingError("single failed")

    with pytest.raises(rebuild_index.RebuildError, match="No images were successfully embedded"):
        rebuild_index.rebuild_candidate_index(
            dataset_dir=dataset,
            output_dir=output,
            batch_size=2,
            embed_batch=fake_batch,
            embed_single=fake_single,
            embedding_metadata=_embedding_metadata(),
        )

    assert not output.exists()


def test_rebuild_rejects_invalid_embedding_shape_without_publishing(tmp_path):
    dataset = tmp_path / "dataset"
    _write_image(dataset, "a.png", (255, 0, 0))
    output = tmp_path / "candidate"

    with pytest.raises(rebuild_index.RebuildError, match="Embedding shape mismatch"):
        rebuild_index.rebuild_candidate_index(
            dataset_dir=dataset,
            output_dir=output,
            batch_size=2,
            embed_batch=lambda paths, batch_size=None: np.zeros((len(paths), 3), dtype=np.float32),
            embed_single=lambda path: np.ones(8, dtype=np.float32),
            embedding_metadata=_embedding_metadata(),
        )

    assert not output.exists()


def test_rebuild_does_not_swallow_model_failures(tmp_path):
    dataset = tmp_path / "dataset"
    _write_image(dataset, "a.png", (255, 0, 0))

    def fake_batch(paths, batch_size=None):
        raise EmbeddingModelError("model unavailable")

    with pytest.raises(rebuild_index.RebuildError, match="model unavailable"):
        rebuild_index.rebuild_candidate_index(
            dataset_dir=dataset,
            output_dir=tmp_path / "candidate",
            batch_size=2,
            embed_batch=fake_batch,
            embed_single=lambda path: np.ones(8, dtype=np.float32),
            embedding_metadata=_embedding_metadata(),
        )


def test_rebuild_rejects_existing_or_unsafe_output_directory(tmp_path):
    dataset = tmp_path / "dataset"
    _write_image(dataset, "a.png", (255, 0, 0))

    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "marker.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(rebuild_index.RebuildError, match="already exists"):
        _rebuild(dataset, existing)
    assert marker.read_text(encoding="utf-8") == "keep"

    with pytest.raises(rebuild_index.RebuildError, match="production artifact directory"):
        _rebuild(dataset, rebuild_index.BACKEND_DIR)

    with pytest.raises(rebuild_index.RebuildError, match="overlap the dataset"):
        _rebuild(dataset, dataset / "candidate")


def test_rebuild_cleans_staging_when_validation_fails(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    _write_image(dataset, "a.png", (255, 0, 0))
    output = tmp_path / "candidate"

    monkeypatch.setattr(
        rebuild_index,
        "validate_candidate_artifacts",
        lambda *args, **kwargs: ValidationResult(valid=False, errors=["forced failure"]),
    )

    with pytest.raises(rebuild_index.RebuildError, match="forced failure"):
        _rebuild(dataset, output)

    assert not output.exists()
    assert list(tmp_path.glob(".candidate.tmp-*")) == []


def test_rebuild_batch_size_validation(tmp_path):
    dataset = tmp_path / "dataset"
    _write_image(dataset, "a.png", (255, 0, 0))

    for invalid in (0, -1, rebuild_index.MAX_BATCH_SIZE + 1):
        with pytest.raises(rebuild_index.RebuildError, match="batch_size must be"):
            _rebuild(dataset, tmp_path / f"candidate-{invalid}", batch_size=invalid)


def test_embedding_metadata_from_service_is_json_safe():
    class FakeConfig:
        _commit_hash = "revision-123"

    class FakeModel:
        config = FakeConfig()

    class FakeImageProcessor:
        size = {"shortest_edge": 224}
        crop_size = {"height": 224, "width": 224}
        resample = 3
        image_mean = [0.48145466, 0.4578275, 0.40821073]
        image_std = [0.26862954, 0.26130258, 0.27577711]

    class FakeProcessor:
        image_processor = FakeImageProcessor()

    class FakeService:
        model_name = "openai/clip-vit-base-patch32"
        embedding_dim = 512
        device = "cpu"
        model = FakeModel()
        processor = FakeProcessor()

    metadata = rebuild_index.get_embedding_metadata_from_service(FakeService())

    assert metadata["model_revision"] == "revision-123"
    assert metadata["embedding_dim"] == 512
    assert metadata["preprocessing"]["image_size"] == {"shortest_edge": 224}
    json.dumps(metadata)


def test_candidate_cli_validate_only(tmp_path, capsys):
    dataset = tmp_path / "dataset"
    _write_image(dataset, "a.png", (255, 0, 0))
    output = tmp_path / "candidate"
    _rebuild(dataset, output)

    exit_code = rebuild_index.main(["--validate-only", str(output)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "VALID" in captured.out
    assert validate_candidate_artifacts(output).valid

    expected_id = build_stable_ids(["a.png"])["a.png"]
    index = load_faiss_index(str(output / INDEX_FILENAME))
    results = search_index(index, _vector_for(dataset / "a.png", dataset), top_k=1)
    assert results[0][0] == expected_id
    assert results[0][1] > 0.999
