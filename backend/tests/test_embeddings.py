"""Tests for the shared CLIP embedding service.

These tests load the real ``openai/clip-vit-base-patch32`` checkpoint. The first
run downloads the model from Hugging Face and can take a few minutes. No
production FAISS index or metadata file is touched.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from embeddings import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MODEL_REVISION,
    EmbeddingError,
    EmbeddingModelError,
    generate_embedding,
    generate_embeddings_batch,
    get_embedding_service,
    resolve_device,
)


@pytest.fixture(scope="session")
def service():
    """Load the real CLIP service once for the test session."""
    return get_embedding_service(device="cpu", batch_size=4)


def _patterned_image(seed: int, size: tuple[int, int] = (96, 96)) -> Image.Image:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, size[0] // 2, size[1] // 2), outline=(255, 255, 255), width=3)
    draw.line((0, size[1] - 1, size[0] - 1, 0), fill=(255, 0, 0), width=2)
    return image


def _save_image(path, image: Image.Image, format_name: str = "PNG") -> str:
    image.save(path, format=format_name)
    return str(path)


def _assert_valid_embedding(embedding: np.ndarray) -> None:
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (DEFAULT_EMBEDDING_DIM,)
    assert embedding.dtype == np.float32
    assert np.all(np.isfinite(embedding))
    assert np.isclose(float(np.linalg.norm(embedding)), 1.0, atol=1e-4)
    assert embedding.flags["C_CONTIGUOUS"]


def test_jpeg_path(service, tmp_path):
    path = _save_image(tmp_path / "sample.jpg", _patterned_image(1), "JPEG")
    _assert_valid_embedding(generate_embedding(path))


def test_png_path(service, tmp_path):
    path = _save_image(tmp_path / "sample.png", _patterned_image(2))
    _assert_valid_embedding(generate_embedding(path))


def test_grayscale_image(service, tmp_path):
    image = Image.new("L", (96, 96), color=128)
    ImageDraw.Draw(image).rectangle((10, 10, 80, 80), fill=240)
    path = _save_image(tmp_path / "gray.png", image)
    _assert_valid_embedding(generate_embedding(path))


def test_rgba_image(service, tmp_path):
    image = Image.new("RGBA", (96, 96), color=(10, 20, 30, 128))
    path = _save_image(tmp_path / "rgba.png", image)
    _assert_valid_embedding(generate_embedding(path))


def test_pil_image_input(service):
    _assert_valid_embedding(generate_embedding(_patterned_image(3)))


def test_bytes_input(service):
    buffer = io.BytesIO()
    _patterned_image(4).save(buffer, format="JPEG")
    _assert_valid_embedding(generate_embedding(buffer.getvalue()))


def test_oversized_image_is_processed(service):
    image = Image.new("RGB", (1600, 1200), color=(30, 120, 200))
    _assert_valid_embedding(generate_embedding(image))


def test_corrupt_image_raises_embedding_error(service, tmp_path):
    path = tmp_path / "corrupt.jpg"
    path.write_bytes(b"this is definitely not a JPEG image")
    with pytest.raises(EmbeddingError):
        generate_embedding(str(path))


def test_truncated_image_raises_embedding_error(service):
    buffer = io.BytesIO()
    _patterned_image(5).save(buffer, format="JPEG")
    truncated = buffer.getvalue()[:100]
    with pytest.raises(EmbeddingError):
        generate_embedding(truncated)


def test_missing_file_raises_embedding_error(service, tmp_path):
    with pytest.raises(EmbeddingError):
        generate_embedding(str(tmp_path / "does-not-exist.jpg"))


def test_unsupported_input_type_raises_embedding_error(service):
    with pytest.raises(EmbeddingError):
        generate_embedding({"not": "an image"})


def test_empty_bytes_raises_embedding_error(service):
    with pytest.raises(EmbeddingError):
        generate_embedding(b"")


def test_empty_batch_returns_empty_matrix(service):
    embeddings = generate_embeddings_batch([])
    assert embeddings.shape == (0, DEFAULT_EMBEDDING_DIM)
    assert embeddings.dtype == np.float32
    assert embeddings.flags["C_CONTIGUOUS"]


def test_batch_matches_single_and_preserves_order(service, tmp_path):
    paths = []
    for index in range(3):
        paths.append(
            _save_image(tmp_path / f"image_{index}.png", _patterned_image(10 + index))
        )

    batch = generate_embeddings_batch(paths, batch_size=2)
    singles = [generate_embedding(path) for path in paths]

    assert batch.shape == (3, DEFAULT_EMBEDDING_DIM)
    for batch_row, single_row in zip(batch, singles):
        assert np.allclose(batch_row, single_row, atol=1e-5)


def test_repeated_embedding_is_deterministic(service, tmp_path):
    path = _save_image(tmp_path / "repeat.png", _patterned_image(20))
    first = generate_embedding(path)
    second = generate_embedding(path)
    assert np.allclose(first, second, atol=1e-6)


def test_different_images_produce_different_embeddings(service, tmp_path):
    first = generate_embedding(_save_image(tmp_path / "a.png", _patterned_image(30)))
    second = generate_embedding(_save_image(tmp_path / "b.png", _patterned_image(31)))
    assert not np.allclose(first, second, atol=1e-4)


def test_invalid_image_in_batch_raises(service, tmp_path):
    good = _save_image(tmp_path / "good.png", _patterned_image(40))
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(EmbeddingError):
        generate_embeddings_batch([good, str(bad)], batch_size=2)


def test_service_is_cached(service):
    assert get_embedding_service(device="cpu") is service


def test_service_loads_the_pinned_model_revision(service):
    assert service.requested_revision == DEFAULT_MODEL_REVISION
    assert service.model_revision == DEFAULT_MODEL_REVISION
    assert service.model.config._commit_hash == DEFAULT_MODEL_REVISION


def test_service_refuses_to_switch_revision(service):
    with pytest.raises(EmbeddingModelError, match="revision"):
        get_embedding_service(device="cpu", revision="0" * 40)


def test_invalid_device_raises_clear_error():
    with pytest.raises(EmbeddingModelError):
        resolve_device("not-a-real-device")


def test_unavailable_cuda_raises_clear_error():
    if __import__("torch").cuda.is_available():
        pytest.skip("CUDA is available; unavailable-device path not applicable")
    with pytest.raises(EmbeddingModelError, match="CUDA"):
        resolve_device("cuda")
