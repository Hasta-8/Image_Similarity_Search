"""Tests for evaluation input validation; no relevance benchmark is implied."""

import argparse
import json

import pytest

from artifact_paths import DatasetPathResolver
from evaluate_retrieval import EvaluationError, load_ground_truth, parse_ks


def _resolver_with_images(root):
    root.mkdir()
    (root / "nested").mkdir()
    (root / "query.jpg").write_bytes(b"query fixture")
    (root / "nested" / "relevant.jpg").write_bytes(b"relevance fixture")
    return DatasetPathResolver(root)


def test_load_ground_truth_accepts_dataset_relative_paths(tmp_path):
    resolver = _resolver_with_images(tmp_path / "dataset")
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps({
            "query_path": "query.jpg",
            "relevant_paths": ["nested/relevant.jpg"],
        }) + "\n",
        encoding="utf-8",
    )

    records = load_ground_truth(
        labels,
        resolver,
        {"query.jpg", "nested/relevant.jpg"},
    )

    assert records == [{
        "query_path": "query.jpg",
        "relevant_paths": ["nested/relevant.jpg"],
    }]


def test_load_ground_truth_rejects_self_and_unindexed_relevance(tmp_path):
    resolver = _resolver_with_images(tmp_path / "dataset")
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps({"query_path": "query.jpg", "relevant_paths": ["query.jpg"]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="query itself"):
        load_ground_truth(labels, resolver, {"query.jpg", "nested/relevant.jpg"})

    labels.write_text(
        json.dumps({
            "query_path": "query.jpg",
            "relevant_paths": ["nested/relevant.jpg"],
        }) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="not present in candidate metadata"):
        load_ground_truth(labels, resolver, {"query.jpg"})


def test_empty_label_file_is_not_reported_as_a_benchmark(tmp_path):
    resolver = _resolver_with_images(tmp_path / "dataset")
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n", encoding="utf-8")

    with pytest.raises(EvaluationError, match="no labeled query records"):
        load_ground_truth(labels, resolver, {"query.jpg", "nested/relevant.jpg"})


def test_parse_ks_sorts_and_deduplicates():
    assert parse_ks("10,1,5,5") == [1, 5, 10]

    with pytest.raises(argparse.ArgumentTypeError):
        parse_ks("0,5")
