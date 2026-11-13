"""
Shared CLIP image embedding service.

This module is the single source of truth for image embeddings used by both
dataset indexing and API search. It loads ``openai/clip-vit-base-patch32`` once
per process, applies the same RGB preprocessing for every input type, and
returns L2-normalized, contiguous ``float32`` vectors.

No random or fallback embeddings are produced. Invalid images raise
``EmbeddingError``; model/device problems raise ``EmbeddingModelError``.
"""

from __future__ import annotations

import logging
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor

DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"
DEFAULT_EMBEDDING_DIM = 512
DEFAULT_BATCH_SIZE = 16
MAX_BATCH_SIZE = 128

logger = logging.getLogger(__name__)

ImageSource = Union[str, Path, Image.Image, bytes, bytearray, memoryview]


class EmbeddingError(ValueError):
    """Raised when an image cannot be loaded, decoded, or converted to RGB."""


class EmbeddingModelError(RuntimeError):
    """Raised when the CLIP model, requested device, or inference fails."""


def resolve_device(device: Union[str, torch.device, None] = "auto") -> torch.device:
    """Resolve and validate a requested torch device.

    ``auto`` selects CUDA when available, otherwise CPU. Explicit CUDA requests
    fail clearly when CUDA is unavailable rather than silently falling back.
    """
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        resolved = device if isinstance(device, torch.device) else torch.device(device)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise EmbeddingModelError(f"Invalid torch device {device!r}: {exc}") from exc

    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise EmbeddingModelError(
                "CUDA device requested but CUDA is not available in this environment"
            )
        device_count = torch.cuda.device_count()
        if resolved.index is not None and resolved.index >= device_count:
            raise EmbeddingModelError(
                f"CUDA device index {resolved.index} is invalid; "
                f"{device_count} CUDA device(s) available"
            )
    elif resolved.type != "cpu":
        raise EmbeddingModelError(
            f"Unsupported device type {resolved.type!r}; use 'cpu', 'cuda', or 'auto'"
        )

    return resolved


def _source_label(source: ImageSource, index: Optional[int] = None) -> str:
    """Build a human-readable label without exposing unbounded binary data."""
    if isinstance(source, (str, Path)):
        label = str(source)
    elif isinstance(source, (bytes, bytearray, memoryview)):
        label = f"<bytes:{len(source)}>"
    elif isinstance(source, Image.Image):
        label = f"<PIL {source.mode} {source.size[0]}x{source.size[1]}>"
    else:
        label = f"<{type(source).__name__}>"
    return f"item {index} ({label})" if index is not None else label


def load_image(source: ImageSource, index: Optional[int] = None) -> Image.Image:
    """Load any supported source into a fully decoded RGB PIL image."""
    label = _source_label(source, index)

    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise EmbeddingError(f"Image file not found: {path}")
            if not path.is_file():
                raise EmbeddingError(f"Image path is not a file: {path}")
            with Image.open(path) as image:
                return image.convert("RGB")
        if isinstance(source, (bytes, bytearray, memoryview)):
            raw = bytes(source)
            if not raw:
                raise EmbeddingError("Empty image byte payload")
            with Image.open(BytesIO(raw)) as image:
                return image.convert("RGB")
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        raise EmbeddingError(
            f"Unsupported image input type {type(source).__name__}; "
            "expected path, PIL image, or bytes"
        )
    except EmbeddingError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise EmbeddingError(f"Could not load image {label}: {exc}") from exc


def _validate_batch_size(batch_size: int) -> int:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise ValueError(f"batch_size must be an integer, got {type(batch_size).__name__}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if batch_size > MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be <= {MAX_BATCH_SIZE}, got {batch_size}")
    return batch_size


class ClipEmbeddingService:
    """Process-wide CLIP image encoder with deterministic preprocessing."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Union[str, torch.device, None] = "auto",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = _validate_batch_size(batch_size)
        self._inference_lock = threading.Lock()
        self._model: Optional[CLIPModel] = None
        self._processor: Optional[CLIPProcessor] = None

        logger.info("Loading CLIP model %s on %s", self.model_name, self.device)
        try:
            model = CLIPModel.from_pretrained(self.model_name)
            processor = CLIPProcessor.from_pretrained(self.model_name)
            model.to(self.device)
            model.eval()
        except Exception as exc:
            logger.exception("Failed to load CLIP model %s", self.model_name)
            raise EmbeddingModelError(
                f"Failed to load CLIP model {self.model_name!r}: {exc}"
            ) from exc

        self._model = model
        self._processor = processor
        self.embedding_dim = int(getattr(model.config, "projection_dim", 0))
        if self.embedding_dim <= 0:
            raise EmbeddingModelError(
                f"CLIP model {self.model_name!r} reported an invalid embedding dimension"
            )

        logger.info(
            "CLIP model ready: name=%s device=%s embedding_dim=%d batch_size=%d",
            self.model_name,
            self.device,
            self.embedding_dim,
            self.batch_size,
        )

    @property
    def model(self) -> CLIPModel:
        if self._model is None:
            raise EmbeddingModelError("CLIP model is not loaded")
        return self._model

    @property
    def processor(self) -> CLIPProcessor:
        if self._processor is None:
            raise EmbeddingModelError("CLIP processor is not loaded")
        return self._processor

    def _embed_batch(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Run CLIP inference on an RGB image batch and return unit vectors."""
        if not images:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        started = time.perf_counter()
        try:
            with self._inference_lock:
                with torch.inference_mode():
                    inputs = self.processor(images=list(images), return_tensors="pt")
                    inputs = {
                        key: value.to(self.device) if torch.is_tensor(value) else value
                        for key, value in inputs.items()
                    }
                    features = self.model.get_image_features(**inputs)

                    if not torch.is_tensor(features) or features.ndim != 2:
                        raise EmbeddingModelError(
                            f"CLIP image encoder returned an unexpected output: "
                            f"{type(features).__name__}"
                        )

                    features = torch.nn.functional.normalize(features, p=2, dim=-1)
                    embeddings = (
                        features.detach().to(device="cpu", dtype=torch.float32).numpy()
                    )
        except EmbeddingModelError:
            raise
        except Exception as exc:
            logger.exception("CLIP inference failed for %d image(s)", len(images))
            raise EmbeddingModelError(f"CLIP inference failed: {exc}") from exc

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.shape != (len(images), self.embedding_dim):
            raise EmbeddingModelError(
                f"CLIP returned shape {embeddings.shape}, expected "
                f"{(len(images), self.embedding_dim)}"
            )
        if not np.all(np.isfinite(embeddings)):
            raise EmbeddingModelError("CLIP returned non-finite embedding values")

        norms = np.linalg.norm(embeddings, axis=1)
        if np.any(norms <= 1e-12):
            raise EmbeddingModelError("CLIP returned a zero-norm embedding")
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise EmbeddingModelError(
                f"CLIP embeddings are not unit-normalized (norms {norms.min():.6f}"
                f"..{norms.max():.6f})"
            )

        duration_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "Embedded %d image(s) in %.1f ms (model=%s device=%s)",
            len(images),
            duration_ms,
            self.model_name,
            self.device,
        )
        return embeddings

    def embed_images(
        self,
        sources: Union[ImageSource, Sequence[ImageSource]],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Embed one or more sources in bounded batches, preserving input order."""
        if isinstance(sources, (str, Path, bytes, bytearray, memoryview, Image.Image)):
            source_list: List[ImageSource] = [sources]
        else:
            source_list = list(sources)

        selected_batch_size = _validate_batch_size(
            self.batch_size if batch_size is None else batch_size
        )
        if not source_list:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        batches: List[np.ndarray] = []
        for start in range(0, len(source_list), selected_batch_size):
            chunk = source_list[start : start + selected_batch_size]
            images: List[Image.Image] = []
            for offset, source in enumerate(chunk):
                absolute_index = start + offset
                try:
                    images.append(load_image(source, index=absolute_index))
                except EmbeddingError:
                    logger.warning(
                        "Image preprocessing failed for %s",
                        _source_label(source, index=absolute_index),
                    )
                    raise
            batches.append(self._embed_batch(images))

        result = np.vstack(batches) if len(batches) > 1 else batches[0]
        result = np.ascontiguousarray(result, dtype=np.float32)
        if result.shape != (len(source_list), self.embedding_dim):
            raise EmbeddingModelError(
                f"Batch result shape {result.shape} does not match "
                f"{(len(source_list), self.embedding_dim)}"
            )
        return result

    def embed_image(
        self,
        source: ImageSource,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Embed a single source and return a contiguous ``(dim,)`` vector."""
        return self.embed_images([source], batch_size=batch_size)[0]


_service: Optional[ClipEmbeddingService] = None
_service_lock = threading.Lock()


def get_embedding_service(
    model_name: str = DEFAULT_MODEL_NAME,
    device: Union[str, torch.device, None] = "auto",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> ClipEmbeddingService:
    """Return the process-wide embedding service, loading it on first use.

    The first call fixes the model and device for the process. Later calls with
    a different model or device raise rather than silently mixing embedding
    spaces.
    """
    global _service

    with _service_lock:
        if _service is None:
            _service = ClipEmbeddingService(
                model_name=model_name,
                device=device,
                batch_size=batch_size,
            )
            return _service

        requested_device = resolve_device(device)
        if model_name != _service.model_name:
            raise EmbeddingModelError(
                f"Embedding service already initialized with model "
                f"{_service.model_name!r}; refusing to switch to {model_name!r}"
            )
        if requested_device != _service.device:
            raise EmbeddingModelError(
                f"Embedding service already initialized on {_service.device}; "
                f"refusing to switch to {requested_device}"
            )
        return _service


def _get_default_service() -> ClipEmbeddingService:
    """Return the cached service without re-validating default constructor args."""
    global _service
    with _service_lock:
        if _service is None:
            _service = ClipEmbeddingService()
        return _service


def reset_embedding_service() -> None:
    """Reset the process-wide service. Intended for tests and controlled reloads."""
    global _service
    with _service_lock:
        _service = None


def load_clip_model():
    """Compatibility wrapper returning the cached ``(model, processor)`` pair."""
    service = _get_default_service()
    return service.model, service.processor


def generate_embedding(
    image_input: ImageSource,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """Generate a normalized 512-dimensional CLIP embedding for one image.

    Args:
        image_input: File path, PIL image, or encoded image bytes.
        batch_size: Optional per-call batch size (single-image path uses 1).

    Returns:
        Contiguous ``float32`` array of shape ``(512,)`` with unit L2 norm.
    """
    service = _get_default_service()
    return service.embed_image(image_input, batch_size=batch_size)


def generate_embeddings_batch(
    image_sources: Union[ImageSource, Sequence[ImageSource]],
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """Generate normalized CLIP embeddings for multiple images.

    Returns:
        Contiguous ``float32`` array of shape ``(n_images, 512)``.
    """
    service = _get_default_service()
    return service.embed_images(image_sources, batch_size=batch_size)


def generate_embeddings_batch_from_paths(
    image_paths: Sequence[Union[str, Path]],
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """Backward-compatible helper for path-only batches."""
    return generate_embeddings_batch(image_paths, batch_size=batch_size)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python embeddings.py <image_path> [<image_path> ...]")
        sys.exit(1)

    try:
        sources = sys.argv[1:]
        embeddings = generate_embeddings_batch(sources)
        for source, embedding in zip(sources, embeddings):
            print(f"Embedding: {source}")
            print(f"  Shape: {embedding.shape}")
            print(f"  Dtype: {embedding.dtype}")
            print(f"  Norm:  {np.linalg.norm(embedding):.6f}")
        print(f"Generated {len(embeddings)} embedding(s) of dimension {embeddings.shape[1]}")
    except (EmbeddingError, EmbeddingModelError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)
