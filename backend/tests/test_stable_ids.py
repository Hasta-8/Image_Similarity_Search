"""Tests for deterministic dataset-relative image IDs."""

from __future__ import annotations

import pytest

import stable_ids
from stable_ids import (
    ID_BITS,
    IDCollisionError,
    StableIDError,
    build_stable_ids,
    dataset_relative_path,
    id_strategy_metadata,
    normalize_dataset_relative_path,
    stable_image_id,
)


def test_stable_id_is_deterministic_and_int64_compatible():
    first = stable_image_id("cat.jpg")
    second = stable_image_id("cat.jpg")
    assert first == second
    assert first == 1484942834709677640
    assert 0 <= first <= (1 << ID_BITS) - 1


def test_stable_id_normalizes_relative_path_forms():
    assert stable_image_id("nested\\dog.png") == stable_image_id("./nested/dog.png")
    assert stable_image_id("nested//dog.png") == stable_image_id("nested/dog.png")


def test_different_paths_have_different_ids():
    assert stable_image_id("cat.jpg") != stable_image_id("dog.jpg")
    assert stable_image_id("nested/dog.png") == 246783455577582185


def test_normalize_rejects_absolute_and_parent_paths():
    with pytest.raises(StableIDError):
        normalize_dataset_relative_path("/absolute/image.jpg")
    with pytest.raises(StableIDError):
        normalize_dataset_relative_path("../image.jpg")
    with pytest.raises(StableIDError):
        normalize_dataset_relative_path("nested/../../image.jpg")
    with pytest.raises(StableIDError):
        normalize_dataset_relative_path("")


def test_dataset_relative_path_requires_image_inside_dataset(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    image = dataset / "nested" / "cat.jpg"
    assert dataset_relative_path(dataset, image) == "nested/cat.jpg"

    outside = tmp_path / "outside.jpg"
    with pytest.raises(StableIDError):
        dataset_relative_path(dataset, outside)


def test_dataset_relative_path_does_not_require_file_to_exist(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    assert dataset_relative_path(dataset, dataset / "missing.png") == "missing.png"


def test_build_stable_ids_is_sorted_and_order_independent():
    forward = build_stable_ids(["b.jpg", "a.jpg", "nested/c.jpg"])
    reverse = build_stable_ids(["nested/c.jpg", "a.jpg", "b.jpg"])
    assert forward == reverse
    assert list(forward.keys()) == ["a.jpg", "b.jpg", "nested/c.jpg"]


def test_build_stable_ids_rejects_duplicate_identity():
    with pytest.raises(StableIDError, match="Duplicate image identity"):
        build_stable_ids(["a.jpg", "./a.jpg"])


def test_build_stable_ids_rejects_collision(monkeypatch):
    monkeypatch.setattr(stable_ids, "stable_image_id", lambda path: 42)
    with pytest.raises(IDCollisionError, match="collision"):
        build_stable_ids(["a.jpg", "b.jpg"])


def test_id_strategy_metadata_describes_algorithm():
    metadata = id_strategy_metadata()
    assert metadata["name"] == "sha256-dataset-relative-path"
    assert metadata["version"] == 1
    assert metadata["id_bits"] == 63
    assert metadata["hash"] == "sha256"
