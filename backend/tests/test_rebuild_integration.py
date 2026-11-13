"""End-to-end candidate rebuild using the real CLIP checkpoint."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image, ImageDraw

import rebuild_index
from dataset_fingerprint import sha256_file
from embeddings import DEFAULT_MODEL_NAME, generate_embedding, get_embedding_service
from faiss_index import load_faiss_index, load_metadata, search_index
from stable_ids import build_stable_ids


def _patterned_image(seed: int, size: tuple[int, int] = (96, 96)) -> Image.Image:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, size[0] // 2, size[1] // 2), outline=(255, 255, 255), width=3)
    draw.line((0, size[1] - 1, size[0] - 1, 0), fill=(255, 0, 0), width=2)
    return image


def test_real_clip_candidate_rebuild_and_exact_retrieval(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    relative_paths = ["alpha.jpg", "nested/beta.png", "gamma.jpg"]
    for index, relative_path in enumerate(relative_paths):
        path = dataset / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        _patterned_image(index + 1).save(path)

    service = get_embedding_service(device="cpu", batch_size=2)
    embedding_metadata = rebuild_index.get_embedding_metadata_from_service(service)
    production_index = rebuild_index.PRODUCTION_INDEX_PATH
    production_metadata = rebuild_index.PRODUCTION_METADATA_PATH
    before_index = sha256_file(production_index)
    before_metadata = sha256_file(production_metadata)

    output = tmp_path / "candidate"
    result = rebuild_index.rebuild_candidate_index(
        dataset_dir=dataset,
        output_dir=output,
        batch_size=2,
        embedding_metadata=embedding_metadata,
        generated_at_utc="2026-01-01T00:00:00Z",
    )

    assert result.indexed_images == 3
    assert result.failed_images == 0
    assert result.validation.valid, result.validation.errors

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["embedding"]["model_name"] == DEFAULT_MODEL_NAME
    assert manifest["embedding"]["embedding_dim"] == 512
    assert manifest["embedding"]["model_revision"]
    assert manifest["artifact_role"] == "candidate"
    assert manifest["promotion_status"] == "not_promoted"

    metadata = load_metadata(str(result.metadata_path))
    expected_ids = build_stable_ids(relative_paths)
    assert metadata == {expected_ids[path]: path for path in relative_paths}
    assert all((dataset / path).is_file() for path in metadata.values())

    index = load_faiss_index(str(result.index_path))
    for relative_path in relative_paths:
        query = generate_embedding(dataset / relative_path)
        results = search_index(index, query, top_k=1)
        assert results[0][0] == expected_ids[relative_path]
        assert results[0][1] > 0.999

    assert sha256_file(production_index) == before_index
    assert sha256_file(production_metadata) == before_metadata
