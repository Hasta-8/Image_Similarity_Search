"""
FastAPI Backend for Image Similarity Search

Uses FAISS for vector search and CLIP for image embeddings.
"""

import asyncio
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.concurrency import run_in_threadpool

from artifact_paths import DatasetPathError, DatasetPathResolver
from embeddings import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    EmbeddingError,
    EmbeddingModelError,
    generate_embedding,
)
from faiss_index import (
    load_faiss_index,
    search_index,
    load_metadata
)
from release_store import ReleaseStore, ReleaseStoreError
from request_limits import RequestBodyLimitMiddleware

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number, got {raw!r}") from exc


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


BACKEND_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(
    os.environ.get("IMGSIMI_DATASET_DIR", str(BACKEND_DIR.parent / "dataset"))
).expanduser()
RELEASE_STORE_DIR = Path(
    os.environ.get("IMGSIMI_RELEASE_STORE", str(BACKEND_DIR / "artifacts"))
).expanduser()

# Optional built React app served by this process (single public origin).
# Unset means API-only behavior, as in development.
_frontend_env = os.environ.get("IMGSIMI_FRONTEND_DIR", "").strip()
FRONTEND_DIR: Optional[Path] = (
    Path(_frontend_env).expanduser().resolve() if _frontend_env else None
)

# Public-facing limits. Defaults are conservative for an unauthenticated demo.
MAX_UPLOAD_BYTES = _env_int("IMGSIMI_MAX_UPLOAD_BYTES", 16 * 1024 * 1024)
BODY_READ_TIMEOUT_SECONDS = _env_float("IMGSIMI_BODY_READ_TIMEOUT_SECONDS", 30.0)
SEARCH_MAX_CONCURRENT = _env_int("IMGSIMI_MAX_CONCURRENT_SEARCHES", 2)
SEARCH_TIMEOUT_SECONDS = _env_float("IMGSIMI_SEARCH_TIMEOUT_SECONDS", 60.0)
WARMUP_MODEL_ON_STARTUP = _env_flag("IMGSIMI_WARMUP_MODEL", False)
# Interactive API docs are off unless explicitly enabled.
ENABLE_DOCS = _env_flag("IMGSIMI_ENABLE_DOCS", False)

if SEARCH_MAX_CONCURRENT < 1:
    raise RuntimeError("IMGSIMI_MAX_CONCURRENT_SEARCHES must be >= 1")
if SEARCH_TIMEOUT_SECONDS <= 0:
    raise RuntimeError("IMGSIMI_SEARCH_TIMEOUT_SECONDS must be > 0")
if MAX_UPLOAD_BYTES < 1:
    raise RuntimeError("IMGSIMI_MAX_UPLOAD_BYTES must be >= 1")

app = FastAPI(
    title="Image Similarity Search API",
    version="1.0.0",
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_DOCS else None,
)

app.mount("/images", StaticFiles(directory=str(DATASET_DIR)), name="images")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    RequestBodyLimitMiddleware,
    max_body_bytes=MAX_UPLOAD_BYTES,
    read_timeout_seconds=BODY_READ_TIMEOUT_SECONDS,
)

_search_inflight = 0

index = None
metadata: Dict[int, str] = {}
index_path = str(BACKEND_DIR / "faiss_index.bin")
metadata_path = str(BACKEND_DIR / "metadata.json")
embedding_dim = DEFAULT_EMBEDDING_DIM
# The query encoder this process uses. A release built with a different model
# or revision lives in a different embedding space and must not be served.
expected_model_name = DEFAULT_MODEL_NAME
expected_model_revision = DEFAULT_MODEL_REVISION
release_store = ReleaseStore(RELEASE_STORE_DIR)
active_release_id: Optional[str] = None
runtime_mode = "unavailable"
runtime_error: Optional[str] = None
path_resolver: Optional[DatasetPathResolver] = None


def get_image_embedding(file_bytes_or_path: Union[bytes, str]) -> np.ndarray:
    """Compute a normalized CLIP query embedding from bytes or a file path.

    The shared embedding service provides identical preprocessing,
    normalization, and model configuration to the indexing path.
    """
    embedding = generate_embedding(file_bytes_or_path)
    embedding = np.ascontiguousarray(embedding, dtype=np.float32).reshape(-1)

    if embedding.shape != (embedding_dim,):
        raise EmbeddingModelError(
            f"Embedding dimension mismatch: expected {embedding_dim}, "
            f"got {embedding.shape[0]}"
        )
    if not np.all(np.isfinite(embedding)):
        raise EmbeddingModelError("Embedding contains non-finite values")

    norm = float(np.linalg.norm(embedding))
    if not np.isclose(norm, 1.0, atol=1e-4):
        raise EmbeddingModelError(
            f"Embedding is not unit-normalized (norm={norm:.6f})"
        )
    return embedding


def _load_legacy_artifacts() -> None:
    global index, metadata, active_release_id, runtime_mode, runtime_error, path_resolver

    index_file = Path(index_path)
    metadata_file = Path(metadata_path)
    if not index_file.is_file() or not metadata_file.is_file():
        raise RuntimeError(
            "No active release exists and legacy production artifacts are incomplete"
        )

    loaded_index = load_faiss_index(str(index_file))
    if loaded_index.d != embedding_dim:
        raise RuntimeError(
            f"Legacy index dimension is {loaded_index.d}; expected {embedding_dim}"
        )
    loaded_metadata = load_metadata(str(metadata_file))
    if loaded_index.ntotal != len(loaded_metadata):
        raise RuntimeError(
            f"Legacy index contains {loaded_index.ntotal} vectors but metadata "
            f"contains {len(loaded_metadata)} records"
        )

    index = loaded_index
    metadata = loaded_metadata
    active_release_id = None
    runtime_mode = "legacy"
    runtime_error = None
    path_resolver = None


def _verify_release_embedding(manifest) -> None:
    """Reject a release whose embedding model differs from the query encoder."""
    embedding = manifest.get("embedding") if isinstance(manifest, Mapping) else None
    if not isinstance(embedding, Mapping):
        raise RuntimeError("Release manifest has no embedding section")

    release_model = embedding.get("model_name")
    if release_model != expected_model_name:
        raise RuntimeError(
            f"Release was built with model {release_model!r}, but the API encodes "
            f"queries with {expected_model_name!r}"
        )
    release_revision = embedding.get("model_revision")
    if release_revision != expected_model_revision:
        raise RuntimeError(
            f"Release was built with model revision {release_revision!r}, but the "
            f"API encodes queries with revision {expected_model_revision!r}; "
            "rebuild and promote a release with the pinned revision"
        )


def load_index_and_metadata() -> None:
    """Load and validate the active release, or legacy artifacts as a fallback.

    An invalid active release is fail-closed: it never falls back to legacy
    artifacts or an empty index. Startup may remain alive for /health, but
    /ready and search return 503 until the release store is repaired.
    """
    global index, metadata, active_release_id, runtime_mode, runtime_error, path_resolver

    try:
        if release_store.pointer_path.is_file():
            artifacts = release_store.load_active_release(
                DATASET_DIR, expected_embedding_dim=embedding_dim
            )
            _verify_release_embedding(artifacts.manifest)
            index = artifacts.index
            metadata = artifacts.metadata
            active_release_id = artifacts.release_id
            runtime_mode = "release"
            runtime_error = None
            path_resolver = artifacts.path_resolver
            print(
                f"Loaded release {active_release_id} "
                f"({index.ntotal} vectors) from {artifacts.release_dir}"
            )
        else:
            _load_legacy_artifacts()
            print(f"Loaded legacy FAISS index from {index_path} ({index.ntotal} vectors)")
    except Exception as exc:
        index = None
        metadata = {}
        active_release_id = None
        runtime_mode = "unavailable"
        runtime_error = str(exc)
        path_resolver = None
        raise


@app.on_event("startup")
async def startup_event():
    """Load index and metadata on startup without hiding fail-closed errors."""
    try:
        load_index_and_metadata()
    except Exception as exc:
        logger.error("Artifact runtime is not ready: %s", exc)
        print(f"Warning: artifact runtime is not ready: {exc}")

    if WARMUP_MODEL_ON_STARTUP:
        # Load CLIP once before public traffic arrives so the first visitor does
        # not pay model-load latency or hit the search timeout.
        try:
            await run_in_threadpool(
                generate_embedding, Image.new("RGB", (224, 224), (127, 127, 127))
            )
            print("CLIP model warm-up completed")
        except Exception as exc:
            logger.error("CLIP model warm-up failed: %s", exc)
            print(f"Warning: CLIP model warm-up failed: {exc}")


async def root():
    """Root endpoint for API-only mode (the SPA owns "/" when served here)."""
    return {"message": "Welcome to Image Similarity Search API"}


def _retrieval_ready() -> bool:
    """Only a fully validated active release is retrieval-ready."""
    return (
        runtime_mode == "release"
        and active_release_id is not None
        and index is not None
    )


@app.get("/health")
async def health_check():
    """Liveness response with runtime artifact state."""
    return {
        "status": "ok",
        "ready": _retrieval_ready(),
        "index_loaded": index is not None,
        "indexed_images": index.ntotal if index else 0,
        "metadata_count": len(metadata),
        "index_file_exists": Path(index_path).is_file(),
        "metadata_file_exists": Path(metadata_path).is_file(),
        "runtime_mode": runtime_mode,
        "active_release_id": active_release_id,
    }


@app.get("/ready")
async def readiness_check():
    """Readiness response; 503 until a validated active release is loaded."""
    if not _retrieval_ready():
        if index is None:
            error = runtime_error or "Artifact runtime is not loaded"
        else:
            error = (
                "No validated active release is loaded; the legacy fallback is "
                "not release-validated or retrieval-ready"
            )
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "runtime_mode": runtime_mode,
                "error": error,
            },
        )
    return {
        "status": "ready",
        "runtime_mode": runtime_mode,
        "active_release_id": active_release_id,
        "indexed_images": index.ntotal,
        "metadata_count": len(metadata),
    }


def _result_paths(file_path: str) -> tuple[str, Optional[str]]:
    if runtime_mode == "release" and path_resolver is not None:
        if file_path == "Unknown":
            return file_path, None
        try:
            relative = path_resolver.relative_path(file_path)
            path_resolver.resolve(relative, must_exist=True)
            return relative, path_resolver.image_url(relative)
        except DatasetPathError as exc:
            logger.warning("Rejected unsafe or missing result path: %s", exc)
            return "Unknown", None

    # Legacy fallback: metadata may hold machine-specific absolute paths, and the
    # old image URL was hardcoded to localhost. Expose only a dataset-relative
    # path and a relative /images URL, and only when the file exists. Absolute
    # host paths are reduced to their basename, matching the old /images/<name>.
    if file_path == "Unknown":
        return file_path, None
    normalized = file_path.replace("\\", "/")
    candidate = normalized if not normalized.startswith("/") else os.path.basename(normalized)
    try:
        resolver = DatasetPathResolver(DATASET_DIR)
        relative = resolver.relative_path(candidate)
        resolver.resolve(relative, must_exist=True)
    except DatasetPathError:
        return os.path.basename(normalized), None
    return relative, resolver.image_url(relative)


class SearchBusyError(Exception):
    """Raised when all concurrent query-embedding slots are occupied."""


class SearchTimeoutError(Exception):
    """Raised when query embedding does not finish within the timeout."""


def _release_search_slot(task: "asyncio.Future") -> None:
    global _search_inflight
    _search_inflight -= 1
    if not task.cancelled():
        # Mark late exceptions as retrieved after a timeout; the caller has gone.
        task.exception()


async def _embed_query_with_limits(file_bytes: bytes) -> np.ndarray:
    """Run CLIP inference under a concurrency cap and a wall-clock timeout.

    The slot is released only when the worker thread finishes, not when the
    request times out, so abandoned inference still counts against the cap.
    """
    global _search_inflight
    if _search_inflight >= SEARCH_MAX_CONCURRENT:
        raise SearchBusyError()

    _search_inflight += 1
    task = asyncio.ensure_future(run_in_threadpool(get_image_embedding, file_bytes))
    task.add_done_callback(_release_search_slot)
    done, _ = await asyncio.wait({task}, timeout=SEARCH_TIMEOUT_SECONDS)
    if not done:
        raise SearchTimeoutError()
    return task.result()


@app.post("/search-image")
async def search_image(file: UploadFile = File(...)):
    """
    Search for similar images using FAISS.

    Accepts an image file and returns the top 5 most similar images
    with their IDs, similarity scores, file paths, and image URLs.
    """
    global index, metadata

    if index is None:
        try:
            load_index_and_metadata()
        except Exception:
            logger.exception("Search rejected because artifact runtime is unavailable")
            raise HTTPException(
                status_code=503,
                detail="Artifact runtime unavailable",
            )

    if index.ntotal == 0:
        raise HTTPException(
            status_code=400,
            detail="No images indexed yet. Please index some images first."
        )

    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (jpg, png, etc.)"
        )

    try:
        file_bytes = await file.read()

        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file provided")

        try:
            query_embedding = await _embed_query_with_limits(file_bytes)
        except SearchBusyError as exc:
            raise HTTPException(
                status_code=429,
                detail="Search capacity is busy; please retry shortly",
                headers={"Retry-After": "5"},
            ) from exc
        except SearchTimeoutError as exc:
            logger.warning("Query embedding exceeded %.0fs", SEARCH_TIMEOUT_SECONDS)
            raise HTTPException(
                status_code=504,
                detail="Search timed out; please retry",
            ) from exc
        except EmbeddingError as exc:
            logger.info("Rejected invalid query image: %s", exc)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid or unsupported image: {exc}",
            ) from exc
        except EmbeddingModelError as exc:
            logger.exception("Embedding service failed while processing query")
            raise HTTPException(
                status_code=503,
                detail="Embedding service unavailable",
            ) from exc

        search_results = search_index(index, query_embedding, top_k=5)

        results = []
        for image_id, score in search_results:
            file_path = metadata.get(image_id, "Unknown")
            result_path, image_url = _result_paths(file_path)
            results.append({
                "id": int(image_id),
                "score": float(score),
                "path": result_path,
                "image_url": image_url
            })

        return {"results": results}

    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error processing search request")
        raise HTTPException(
            status_code=500,
            detail="Error processing search request",
        )


# Register the root JSON route only in API-only mode. When the built React app
# is configured, "/" is served as the single public origin's index.html.
if FRONTEND_DIR is None:
    app.add_api_route("/", root, methods=["GET"])
else:
    if not (FRONTEND_DIR / "index.html").is_file():
        raise RuntimeError(
            f"IMGSIMI_FRONTEND_DIR does not contain index.html: {FRONTEND_DIR}"
        )
    # Mounted last so every API route above keeps precedence over static files.
    app.mount(
        "/",
        StaticFiles(directory=str(FRONTEND_DIR), html=True),
        name="frontend",
    )
