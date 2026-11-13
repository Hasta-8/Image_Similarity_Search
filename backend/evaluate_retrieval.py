"""Evaluate a candidate index with human-labeled query/relevance pairs.

This tool intentionally requires a non-empty ground-truth JSONL file. It does
not infer or manufacture semantic labels from exact-image self-retrieval.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from artifact_manifest import load_manifest
from artifact_paths import DatasetPathError, DatasetPathResolver
from artifact_validation import validate_candidate_artifacts
from dataset_fingerprint import compute_dataset_fingerprint, discover_dataset_images
from embeddings import DEFAULT_BATCH_SIZE, get_embedding_service, generate_embeddings_batch
from faiss_index import load_faiss_index, load_metadata, search_index
from retrieval_metrics import metrics_at_k
from stable_ids import build_stable_ids

DEFAULT_CANDIDATE_DIR = Path(__file__).resolve().parent / "candidate_index"
DEFAULT_DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
EXPECTED_MODEL = "openai/clip-vit-base-patch32"
EXPECTED_DIMENSION = 512


class EvaluationError(ValueError):
    """Raised for invalid evaluation data or incompatible candidate assets."""


def parse_ks(value: str) -> list[int]:
    try:
        ks = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("K values must be comma-separated integers") from exc
    if not ks or any(k < 1 for k in ks):
        raise argparse.ArgumentTypeError("At least one positive K value is required")
    return ks


def load_ground_truth(
    path: str | Path,
    resolver: DatasetPathResolver,
    indexed_paths: set[str],
) -> list[dict[str, Any]]:
    records = []
    seen_queries = set()
    path = Path(path)
    if not path.is_file():
        raise EvaluationError(f"Ground-truth JSONL file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(
                    f"Invalid JSON on line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(item, dict) or set(item) != {"query_path", "relevant_paths"}:
                raise EvaluationError(
                    f"Line {line_number} must contain only query_path and relevant_paths"
                )

            try:
                query_path = resolver.relative_path(item["query_path"])
                resolver.resolve(query_path, must_exist=True)
            except (DatasetPathError, TypeError) as exc:
                raise EvaluationError(f"Invalid query path on line {line_number}: {exc}") from exc

            if query_path in seen_queries:
                raise EvaluationError(f"Duplicate query_path on line {line_number}: {query_path}")
            seen_queries.add(query_path)

            raw_relevant = item["relevant_paths"]
            if not isinstance(raw_relevant, list) or not raw_relevant:
                raise EvaluationError(
                    f"Line {line_number} relevant_paths must be a non-empty array"
                )
            relevant_paths = []
            for raw_path in raw_relevant:
                try:
                    relative_path = resolver.relative_path(raw_path)
                    resolver.resolve(relative_path, must_exist=True)
                except (DatasetPathError, TypeError) as exc:
                    raise EvaluationError(
                        f"Invalid relevant path on line {line_number}: {exc}"
                    ) from exc
                if relative_path == query_path:
                    raise EvaluationError(
                        f"Line {line_number} labels the query itself relevant; omit self matches"
                    )
                if relative_path not in indexed_paths:
                    raise EvaluationError(
                        f"Relevant path is not present in candidate metadata: {relative_path}"
                    )
                relevant_paths.append(relative_path)

            if len(set(relevant_paths)) != len(relevant_paths):
                raise EvaluationError(f"Duplicate relevant path on line {line_number}")
            records.append({"query_path": query_path, "relevant_paths": relevant_paths})

    if not records:
        raise EvaluationError("Ground-truth file has no labeled query records")
    return records


def evaluate(
    candidate_dir: str | Path,
    dataset_dir: str | Path,
    ground_truth_path: str | Path,
    ks: list[int],
) -> dict[str, Any]:
    validation = validate_candidate_artifacts(
        candidate_dir, expected_embedding_dim=EXPECTED_DIMENSION
    )
    if not validation.valid:
        raise EvaluationError("Candidate validation failed: " + "; ".join(validation.errors))

    manifest = load_manifest(Path(candidate_dir) / "manifest.json")
    embedding = manifest["embedding"]
    if embedding.get("model_name") != EXPECTED_MODEL:
        raise EvaluationError(
            f"Expected model {EXPECTED_MODEL!r}, got {embedding.get('model_name')!r}"
        )
    if embedding.get("embedding_dim") != EXPECTED_DIMENSION:
        raise EvaluationError("Candidate embedding dimension is not 512")

    discovered = discover_dataset_images(dataset_dir)
    fingerprint = compute_dataset_fingerprint(
        dataset_dir, [relative_path for relative_path, _ in discovered]
    )
    expected_fingerprint = manifest["dataset"]["fingerprint"]
    if fingerprint.fingerprint != expected_fingerprint:
        raise EvaluationError(
            "Dataset fingerprint differs from candidate manifest: "
            f"actual={fingerprint.fingerprint}, expected={expected_fingerprint}"
        )

    resolver = DatasetPathResolver(dataset_dir)
    metadata = load_metadata(str(Path(candidate_dir) / "metadata.json"))
    resolver.metadata_paths(metadata, must_exist=True)
    stable_ids = build_stable_ids(metadata.values())
    expected_mapping = {image_id: path for path, image_id in stable_ids.items()}
    if metadata != expected_mapping:
        raise EvaluationError("Candidate metadata IDs do not match stable path IDs")

    indexed_paths = set(metadata.values())
    records = load_ground_truth(ground_truth_path, resolver, indexed_paths)
    index = load_faiss_index(str(Path(candidate_dir) / "faiss_index.bin"))

    service = get_embedding_service(
        model_name=embedding["model_name"], device=embedding.get("device", "cpu")
    )
    expected_revision = embedding.get("model_revision")
    actual_revision = getattr(service.model.config, "_commit_hash", None)
    if expected_revision and actual_revision != expected_revision:
        raise EvaluationError(
            f"Loaded CLIP revision {actual_revision!r} differs from candidate "
            f"revision {expected_revision!r}"
        )

    query_paths = [record["query_path"] for record in records]
    query_sources = [str(resolver.resolve(path, must_exist=True)) for path in query_paths]
    query_vectors = generate_embeddings_batch(
        query_sources, batch_size=min(DEFAULT_BATCH_SIZE, len(query_sources))
    )

    per_query = []
    max_k = max(ks)
    for record, vector in zip(records, query_vectors):
        query_path = record["query_path"]
        requested = min(index.ntotal, max_k + 1)
        search_results = search_index(index, vector, top_k=requested)
        ranked_paths = [
            metadata[image_id]
            for image_id, _score in search_results
            if image_id in metadata and metadata[image_id] != query_path
        ][:max_k]
        metrics = {
            str(k): metrics_at_k(ranked_paths, record["relevant_paths"], k)
            for k in ks
        }
        per_query.append({"query_path": query_path, "metrics": metrics})

    aggregate = {}
    for k in ks:
        fields = ("precision", "recall", "average_precision", "reciprocal_rank")
        aggregate[str(k)] = {
            metric: sum(row["metrics"][str(k)][metric] for row in per_query)
            / len(per_query)
            for metric in fields
        }

    return {
        "queries": len(records),
        "model": embedding["model_name"],
        "model_revision": actual_revision or expected_revision,
        "dataset_fingerprint": fingerprint.fingerprint,
        "k_values": ks,
        "macro_average": aggregate,
        "per_query": per_query,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate candidate retrieval using human-labeled JSONL relevance pairs."
    )
    parser.add_argument("--candidate-dir", default=str(DEFAULT_CANDIDATE_DIR))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--ks", type=parse_ks, default=parse_ks("1,5,10"))
    args = parser.parse_args(argv)

    try:
        result = evaluate(args.candidate_dir, args.dataset, args.ground_truth, args.ks)
    except (EvaluationError, OSError, ValueError) as exc:
        print(f"Evaluation blocked: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
