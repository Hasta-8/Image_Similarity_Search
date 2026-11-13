"""Tests for dataset-relative path normalization and safe resolution."""

from __future__ import annotations

import os

import pytest

from artifact_paths import DatasetPathError, DatasetPathResolver, normalize_relative_path


@pytest.mark.parametrize(
    "value",
    [
        "",
        "/absolute.jpg",
        "../escape.jpg",
        "nested/../escape.jpg",
        "nested//double.jpg",
        ".\\windows.jpg",
        "nested\\image.jpg",
        "C:/image.jpg",
        "nested/file:name.jpg",
        "bad\x00name.jpg",
        "./image.jpg",
    ],
)
def test_invalid_relative_paths_are_rejected(value):
    with pytest.raises(DatasetPathError):
        normalize_relative_path(value)


def test_nested_and_unicode_paths_are_normalized():
    assert normalize_relative_path("nested/photo.jpg") == "nested/photo.jpg"
    assert normalize_relative_path("cafe\u0301/photo.jpg") == "café/photo.jpg"


def test_paths_resolve_inside_dataset_and_preserve_duplicate_basenames(tmp_path):
    resolver = DatasetPathResolver(tmp_path)
    first = tmp_path / "a" / "same.jpg"
    second = tmp_path / "b" / "same.jpg"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    assert resolver.resolve("a/same.jpg", must_exist=True) == first.resolve()
    assert resolver.resolve("b/same.jpg", must_exist=True) == second.resolve()


def test_image_url_encodes_each_nested_segment(tmp_path):
    resolver = DatasetPathResolver(tmp_path)

    assert resolver.image_url("nested/photo 1.jpg") == "/images/nested/photo%201.jpg"
    assert resolver.image_url("café/photo.jpg") == "/images/caf%C3%A9/photo.jpg"


def test_missing_file_can_be_resolved_syntactically_or_rejected(tmp_path):
    resolver = DatasetPathResolver(tmp_path)

    assert resolver.resolve("missing.jpg") == (tmp_path / "missing.jpg").resolve()
    with pytest.raises(DatasetPathError, match="does not exist"):
        resolver.resolve("missing.jpg", must_exist=True)


def test_symlink_escape_is_rejected(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"secret")
    (dataset / "link.jpg").symlink_to(outside)

    resolver = DatasetPathResolver(dataset)

    with pytest.raises(DatasetPathError, match="escapes|is not a regular file|does not exist"):
        resolver.resolve("link.jpg", must_exist=True)


def test_dataset_root_must_be_a_directory(tmp_path):
    with pytest.raises(DatasetPathError, match="not a directory"):
        DatasetPathResolver(tmp_path / "missing")


def test_metadata_paths_validation_is_all_or_nothing(tmp_path):
    resolver = DatasetPathResolver(tmp_path)
    (tmp_path / "ok.jpg").write_bytes(b"ok")

    with pytest.raises(DatasetPathError):
        resolver.metadata_paths({1: "ok.jpg", 2: "../bad.jpg"}, must_exist=True)
