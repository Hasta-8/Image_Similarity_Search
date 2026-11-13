"""Deterministic full-index candidate rebuild and validation CLI."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from artifact_manifest import (
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    METADATA_FILENAME,
    build_manifest,
    write_manifest,
)
from artifact_validation import ValidationResult, validate_candidate_artifacts
from dataset_fingerprint import compute_dataset_fingerprint, discover_dataset_images
from embeddings import (
    DEFAULT_BATCH_SIZE,
    MAX_BATCH_SIZE,
    EmbeddingError,
    EmbeddingModelError,
    generate_embedding,
    generate_embeddings_batch,
    get_embedding_service,
)
from faiss_index import add_embeddings, create_faiss_index, save_faiss_index, save_metadata
from stable_ids import (
    StableIDError,
    build_stable_ids,
    id_strategy_metadata,
)

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = BACKEND_DIR.parent / "dataset"
PRODUCTION_INDEX_PATH = BACKEND_DIR / INDEX_FILENAME
PRODUCTION_METADATA_PATH = BACKEND_DIR / METADATA_FILENAME
MAX_ERROR_MESSAGE_LENGTH = 1000


class RebuildError(RuntimeError):
    """Raised when a deterministic candidate rebuild cannot complete safely."""


@dataclass
class RebuildResult:
    output_dir: Path
    index_path: Path
    metadata_path: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    validation: ValidationResult
    indexed_images: int
    failed_images: int
    dataset_fingerprint: str


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def get_embedding_metadata_from_service(service) -> dict:
    """Describe the active CLIP service for the candidate manifest."""
    image_processor = service.processor.image_processor
    return {
        "model_name": service.model_name,
        "model_revision": getattr(
            service, "model_revision", getattr(service.model.config, "_commit_hash", None)
        ),
        "embedding_dim": int(service.embedding_dim),
        "normalization": "l2",
        "device": str(service.device),
        "preprocessing": {
            "input_mode": "RGB",
            "processor_class": type(service.processor).__name__,
            "image_size": _json_safe(getattr(image_processor, "size", None)),
            "crop_size": _json_safe(getattr(image_processor, "crop_size", None)),
            "resample": _json_safe(getattr(image_processor, "resample", None)),
            "image_mean": _json_safe(getattr(image_processor, "image_mean", None)),
            "image_std": _json_safe(getattr(image_processor, "image_std", None)),
        },
    }


def _validate_batch_size(batch_size: int) -> int:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise RebuildError(f"batch_size must be an integer, got {type(batch_size).__name__}")
    if batch_size < 1 or batch_size > MAX_BATCH_SIZE:
        raise RebuildError(
            f"batch_size must be between 1 and {MAX_BATCH_SIZE}, got {batch_size}"
        )
    return batch_size


def _validate_embedding_metadata(embedding_metadata: Mapping[str, Any]) -> dict:
    if not isinstance(embedding_metadata, Mapping):
        raise RebuildError("embedding_metadata must be a mapping")
    metadata = dict(embedding_metadata)

    model_name = metadata.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise RebuildError("embedding_metadata.model_name is required")

    embedding_dim = metadata.get("embedding_dim")
    if (
        isinstance(embedding_dim, bool)
        or not isinstance(embedding_dim, int)
        or embedding_dim < 1
    ):
        raise RebuildError("embedding_metadata.embedding_dim must be a positive integer")

    preprocessing = metadata.get("preprocessing")
    if not isinstance(preprocessing, Mapping) or not preprocessing:
        raise RebuildError("embedding_metadata.preprocessing is required")

    normalization = metadata.get("normalization")
    if not isinstance(normalization, str) or not normalization:
        raise RebuildError("embedding_metadata.normalization is required")

    return metadata


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _prepare_output_dir(output_dir: str | os.PathLike, dataset_dir: Path) -> Path:
    output = Path(output_dir).expanduser().resolve()
    backend = BACKEND_DIR.resolve()

    if output == backend:
        raise RebuildError(
            "Refusing to write candidate artifacts into the production artifact "
            f"directory: {output}"
        )
    if output.name in {INDEX_FILENAME, METADATA_FILENAME, MANIFEST_FILENAME}:
        raise RebuildError(
            f"Candidate output directory name {output.name!r} conflicts with an artifact filename"
        )
    if output.exists():
        raise RebuildError(
            f"Candidate output directory already exists: {output}. Choose a new directory."
        )
    if (
        output == dataset_dir
        or _is_within(output, dataset_dir)
        or _is_within(dataset_dir, output)
    ):
        raise RebuildError(
            "Candidate output directory must not overlap the dataset directory: "
            f"output={output}, dataset={dataset_dir}"
        )

    return output


def _dataset_identifier(dataset_dir: Path) -> str:
    name = dataset_dir.name
    return name if name else "dataset"


def _sanitize_error(exc: BaseException, dataset_dir: Path) -> str:
    message = str(exc).replace(str(dataset_dir), "<dataset>")
    message = message.replace(str(dataset_dir.resolve()), "<dataset>")
    message = " ".join(message.split())
    if len(message) > MAX_ERROR_MESSAGE_LENGTH:
        message = message[:MAX_ERROR_MESSAGE_LENGTH] + "..."
    return message


def _failure_record(
    relative_path: str,
    stage: str,
    error_type: str,
    message: str,
) -> dict:
    return {
        "path": relative_path,
        "stage": stage,
        "error_type": error_type,
        "message": message,
    }


def _validate_embedding_matrix(
    embeddings: Any,
    expected_count: int,
    embedding_dim: int,
) -> np.ndarray:
    try:
        array = np.asarray(embeddings)
    except Exception as exc:
        raise RebuildError(f"Could not convert embeddings to an array: {exc}") from exc

    expected_shape = (expected_count, embedding_dim)
    if array.shape != expected_shape:
        raise RebuildError(
            f"Embedding shape mismatch: expected {expected_shape}, got {array.shape}"
        )

    array = np.ascontiguousarray(array, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise RebuildError("Embedding batch contains non-finite values")

    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 1e-12):
        raise RebuildError("Embedding batch contains a zero-norm vector")
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise RebuildError(
            f"Embedding batch is not unit-normalized "
            f"(norms {norms.min():.6f}..{norms.max():.6f})"
        )
    return array


def _validate_single_embedding(embedding: Any, embedding_dim: int) -> np.ndarray:
    array = np.asarray(embedding).reshape(1, -1)
    return _validate_embedding_matrix(array, 1, embedding_dim)[0]


def _add_failure(
    failures: Dict[str, dict],
    relative_path: str,
    stage: str,
    exc: BaseException | str,
    dataset_dir: Path,
) -> None:
    if relative_path in failures:
        return
    if isinstance(exc, str):
        error_type = exc
        message = f"Could not read file for dataset fingerprinting ({exc})"
    else:
        error_type = type(exc).__name__
        message = _sanitize_error(exc, dataset_dir)
    failures[relative_path] = _failure_record(
        relative_path, stage, error_type, message
    )


def rebuild_candidate_index(
    dataset_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    batch_size: int = DEFAULT_BATCH_SIZE,
    embed_batch: Callable[..., Any] = generate_embeddings_batch,
    embed_single: Callable[..., Any] = generate_embedding,
    embedding_metadata: Optional[Mapping[str, Any]] = None,
    generated_at_utc: Optional[str] = None,
) -> RebuildResult:
    """Create a validated candidate index, metadata, and manifest.

    Existing production artifacts are never read for writing and are never
    replaced. The output directory must not already exist.
    """
    batch_size = _validate_batch_size(batch_size)
    dataset_path = Path(dataset_dir).expanduser().resolve()
    output_path = _prepare_output_dir(output_dir, dataset_path)

    if embedding_metadata is None:
        embedding_metadata = get_embedding_metadata_from_service(get_embedding_service())
    embedding_metadata = _validate_embedding_metadata(embedding_metadata)
    embedding_dim = int(embedding_metadata["embedding_dim"])

    discovered = discover_dataset_images(dataset_path)
    if not discovered:
        raise RebuildError(f"No supported images found in dataset: {dataset_path}")

    relative_paths = [relative_path for relative_path, _ in discovered]
    absolute_paths = {relative_path: absolute_path for relative_path, absolute_path in discovered}

    try:
        stable_ids = build_stable_ids(relative_paths)
    except StableIDError as exc:
        raise RebuildError(f"Could not assign stable image IDs: {exc}") from exc

    try:
        fingerprint = compute_dataset_fingerprint(dataset_path, relative_paths)
    except Exception as exc:
        raise RebuildError(f"Could not fingerprint dataset: {exc}") from exc

    failures: Dict[str, dict] = {}
    for relative_path, error_type in fingerprint.unreadable_files.items():
        _add_failure(failures, relative_path, "fingerprint", error_type, dataset_path)

    eligible_paths = [
        relative_path
        for relative_path in relative_paths
        if relative_path not in fingerprint.unreadable_files
    ]

    embeddings: List[np.ndarray] = []
    metadata: Dict[int, str] = {}

    logger.info(
        "Rebuilding candidate: dataset=%s images=%d batch_size=%d embedding_dim=%d",
        dataset_path,
        len(relative_paths),
        batch_size,
        embedding_dim,
    )

    for start in range(0, len(eligible_paths), batch_size):
        batch_paths = eligible_paths[start : start + batch_size]
        batch_absolute_paths = [absolute_paths[path] for path in batch_paths]
        successful_paths: Sequence[str]
        batch_vectors: np.ndarray

        try:
            batch_vectors = _validate_embedding_matrix(
                embed_batch(batch_absolute_paths, batch_size=batch_size),
                len(batch_paths),
                embedding_dim,
            )
            successful_paths = batch_paths
        except EmbeddingError as batch_error:
            logger.warning(
                "Batch at image %d failed (%s); retrying individually",
                start + 1,
                batch_error,
            )
            individual_vectors: List[np.ndarray] = []
            successful_paths = []
            for relative_path in batch_paths:
                try:
                    vector = _validate_single_embedding(
                        embed_single(absolute_paths[relative_path]), embedding_dim
                    )
                except EmbeddingError as exc:
                    _add_failure(
                        failures, relative_path, "embedding", exc, dataset_path
                    )
                    continue
                except EmbeddingModelError as exc:
                    raise RebuildError(
                        f"Embedding model failed during candidate rebuild: {exc}"
                    ) from exc
                except Exception as exc:
                    raise RebuildError(
                        f"Unexpected embedding failure for {relative_path}: {exc}"
                    ) from exc
                individual_vectors.append(vector)
                successful_paths.append(relative_path)

            if not individual_vectors:
                continue
            batch_vectors = np.stack(individual_vectors).astype(np.float32)
        except EmbeddingModelError as exc:
            raise RebuildError(
                f"Embedding model failed during candidate rebuild: {exc}"
            ) from exc
        except RebuildError:
            raise
        except Exception as exc:
            raise RebuildError(f"Unexpected embedding batch failure: {exc}") from exc

        for relative_path, vector in zip(successful_paths, batch_vectors):
            image_id = stable_ids[relative_path]
            if image_id in metadata:
                raise RebuildError(
                    f"Duplicate stable ID {image_id} while assigning metadata"
                )
            metadata[image_id] = relative_path
            embeddings.append(vector)

        logger.info(
            "Processed %d/%d eligible image(s); %d failed so far",
            min(start + len(batch_paths), len(eligible_paths)),
            len(eligible_paths),
            len(failures),
        )

    if not embeddings:
        raise RebuildError(
            "No images were successfully embedded; no candidate artifacts were written. "
            f"Failures: {len(failures)}"
        )

    embeddings_array = np.stack(embeddings).astype(np.float32, copy=False)
    successful_ids = list(metadata.keys())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=str(output_path.parent))
    )

    try:
        index = create_faiss_index(embedding_dim)
        add_embeddings(index, embeddings_array, successful_ids)

        index_path = staging_path / INDEX_FILENAME
        metadata_path = staging_path / METADATA_FILENAME
        manifest_path = staging_path / MANIFEST_FILENAME

        save_faiss_index(index, str(index_path))
        save_metadata(dict(sorted(metadata.items())), str(metadata_path))

        dataset_summary = {
            "identifier": _dataset_identifier(dataset_path),
            "root": _dataset_identifier(dataset_path),
            "fingerprint": fingerprint.fingerprint,
            "fingerprint_algorithm": fingerprint.algorithm,
            "file_count": fingerprint.file_count,
            "readable_file_count": len(fingerprint.content_hashes),
            "unreadable_file_count": len(fingerprint.unreadable_files),
            "total_bytes": fingerprint.total_bytes,
        }
        indexing_summary = {
            "mode": "full_candidate_rebuild",
            "batch_size": batch_size,
            "device": embedding_metadata.get("device"),
        }
        failure_list = [failures[path] for path in sorted(failures)]

        manifest = build_manifest(
            artifact_dir=staging_path,
            embedding=embedding_metadata,
            dataset=dataset_summary,
            id_strategy=id_strategy_metadata(),
            indexing=indexing_summary,
            indexed_images=len(metadata),
            failed_images=len(failures),
            metadata_records=len(metadata),
            failures=failure_list,
            faiss_type=type(index).__name__,
            generated_at_utc=generated_at_utc,
        )
        write_manifest(manifest, manifest_path)

        validation = validate_candidate_artifacts(
            staging_path,
            expected_embedding_dim=embedding_dim,
            expected_count=len(metadata),
        )
        if not validation.valid:
            raise RebuildError(
                "Candidate artifact validation failed:\n"
                + "\n".join(f"- {error}" for error in validation.errors)
            )

        if output_path.exists():
            raise RebuildError(
                f"Candidate output directory appeared during rebuild: {output_path}"
            )

        try:
            os.replace(staging_path, output_path)
        except OSError as exc:
            raise RebuildError(
                f"Could not publish candidate directory {output_path}: {exc}"
            ) from exc

        final_validation = validate_candidate_artifacts(
            output_path,
            expected_embedding_dim=embedding_dim,
            expected_count=len(metadata),
        )
        if not final_validation.valid:
            raise RebuildError(
                "Published candidate failed validation:\n"
                + "\n".join(f"- {error}" for error in final_validation.errors)
            )

        return RebuildResult(
            output_dir=output_path,
            index_path=output_path / INDEX_FILENAME,
            metadata_path=output_path / METADATA_FILENAME,
            manifest_path=output_path / MANIFEST_FILENAME,
            manifest=manifest,
            validation=final_validation,
            indexed_images=len(metadata),
            failed_images=len(failures),
            dataset_fingerprint=fingerprint.fingerprint,
        )
    except Exception:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise


def _print_validation(validation: ValidationResult) -> None:
    print("Candidate artifact validation: " + ("VALID" if validation.valid else "INVALID"))
    for warning in validation.warnings:
        print(f"WARNING: {warning}")
    for error in validation.errors:
        print(f"ERROR: {error}")
    details = validation.details
    if details:
        print(f"  Index dimension: {details.get('index_dimension')}")
        print(f"  Index vectors:   {details.get('index_ntotal')}")
        print(f"  FAISS IDs:       {details.get('index_id_count')}")
        print(f"  Metadata records:{details.get('metadata_count')}")
        print(f"  Dataset SHA-256: {details.get('dataset_fingerprint')}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or validate a deterministic candidate ImgSimiSearch index. "
            "Production artifacts are never replaced."
        )
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_DIR),
        help=f"Dataset directory (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        help="New candidate artifact directory; must not already exist",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"CLIP embedding batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for embedding: auto, cpu, cuda, or cuda:N (default: auto)",
    )
    parser.add_argument(
        "--validate-only",
        metavar="DIR",
        help="Validate an existing candidate artifact directory and exit",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.validate_only:
        validation = validate_candidate_artifacts(args.validate_only)
        _print_validation(validation)
        return 0 if validation.valid else 1

    if not args.output_dir:
        parser.error("--output-dir is required unless --validate-only is used")

    try:
        service = get_embedding_service(device=args.device, batch_size=args.batch_size)
        embedding_metadata = get_embedding_metadata_from_service(service)
        result = rebuild_candidate_index(
            dataset_dir=args.dataset,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            embedding_metadata=embedding_metadata,
        )
    except RebuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected candidate rebuild failure")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("=" * 68)
    print("CANDIDATE INDEX REBUILD COMPLETE")
    print("=" * 68)
    print(f"Dataset:             {Path(args.dataset).expanduser().resolve()}")
    print(f"Candidate directory: {result.output_dir}")
    print(f"Indexed images:      {result.indexed_images}")
    print(f"Failed images:       {result.failed_images}")
    print(f"Dataset fingerprint: {result.dataset_fingerprint}")
    print(f"Index:               {result.index_path}")
    print(f"Metadata:            {result.metadata_path}")
    print(f"Manifest:            {result.manifest_path}")
    print()
    _print_validation(result.validation)
    print()
    print("Production artifacts were not read for writing, replaced, or deleted.")
    print("Candidate artifacts have not been promoted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
