"""Tests for result-grid thumbnails: generation, caching, and the HTTP route."""

from __future__ import annotations

import io
import os

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main
import thumbnails
from thumbnails import ThumbnailCache, ThumbnailError, make_thumbnail

EXIF_ORIENTATION_TAG = 0x0112


def _save_jpeg(path, size=(1600, 800), color=(200, 40, 40), orientation=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color)
    if orientation is None:
        image.save(path, format="JPEG")
    else:
        exif = Image.Exif()
        exif[EXIF_ORIENTATION_TAG] = orientation
        image.save(path, format="JPEG", exif=exif)
    return path


def _decoded(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    image.load()
    return image


def test_thumbnail_is_a_bounded_jpeg_preserving_aspect_ratio(tmp_path):
    source = _save_jpeg(tmp_path / "wide.jpg", size=(1600, 800))

    thumb = _decoded(make_thumbnail(source, 384))

    assert thumb.format == "JPEG"
    assert thumb.mode == "RGB"
    assert max(thumb.size) == 384
    assert thumb.size == (384, 192)


def test_thumbnail_does_not_upscale_small_images(tmp_path):
    source = _save_jpeg(tmp_path / "small.jpg", size=(120, 80))

    assert _decoded(make_thumbnail(source, 384)).size == (120, 80)


def test_thumbnail_applies_exif_orientation_for_display(tmp_path):
    # Orientation 6 means "rotate 90 degrees clockwise to display": a landscape
    # pixel buffer that browsers show as portrait.
    source = _save_jpeg(tmp_path / "rotated.jpg", size=(400, 200), orientation=6)

    width, height = _decoded(make_thumbnail(source, 384)).size

    assert height > width


def test_png_sources_are_converted_to_jpeg(tmp_path):
    source = tmp_path / "alpha.png"
    Image.new("RGBA", (500, 500), (0, 128, 255, 100)).save(source)

    thumb = _decoded(make_thumbnail(source, 100))

    assert thumb.format == "JPEG"
    assert thumb.size == (100, 100)


def test_corrupt_source_raises_thumbnail_error(tmp_path):
    source = tmp_path / "broken.jpg"
    source.write_bytes(b"not an image")

    with pytest.raises(ThumbnailError):
        make_thumbnail(source, 384)


def test_cache_reuses_thumbnails_and_regenerates_after_source_changes(tmp_path, monkeypatch):
    source = _save_jpeg(tmp_path / "a.jpg")
    calls = []
    real = thumbnails.make_thumbnail
    monkeypatch.setattr(
        thumbnails, "make_thumbnail", lambda path, side: calls.append(path) or real(path, side)
    )
    cache = ThumbnailCache(max_side=64, max_bytes=10_000_000)

    first = cache.get(source)
    assert cache.get(source) is first
    assert len(calls) == 1

    _save_jpeg(source, size=(900, 900), color=(0, 0, 255))
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
    changed = cache.get(source)

    assert len(calls) == 2
    assert _decoded(changed).size == (64, 64)


def test_cache_evicts_least_recently_used_entries_within_budget(tmp_path):
    sources = [_save_jpeg(tmp_path / f"{i}.jpg", color=(i * 60, 0, 0)) for i in range(3)]
    one_thumb = len(make_thumbnail(sources[0], 64))
    cache = ThumbnailCache(max_side=64, max_bytes=one_thumb * 2 + one_thumb // 2)

    for source in sources:
        cache.get(source)

    assert len(cache) == 2
    assert cache.size_bytes <= cache.max_bytes


@pytest.fixture
def thumbnail_client(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    _save_jpeg(dataset / "nested dir" / "café photo.jpg")
    _save_jpeg(dataset / "corrupt.jpg").write_bytes(b"not an image")
    (dataset / "notes.txt").write_text("not an image")
    outside = _save_jpeg(tmp_path / "outside.jpg")
    (dataset / "escape.jpg").symlink_to(outside)

    monkeypatch.setattr(main, "DATASET_DIR", dataset)
    monkeypatch.setattr(main, "path_resolver", None)
    monkeypatch.setattr(main, "thumbnail_cache", ThumbnailCache(384, 10_000_000))
    return TestClient(main.app)


def test_thumbnail_route_serves_cacheable_jpeg_for_encoded_nested_path(thumbnail_client):
    response = thumbnail_client.get("/thumbnails/nested%20dir/caf%C3%A9%20photo.jpg")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert "max-age" in response.headers["cache-control"]
    assert max(_decoded(response.content).size) == 384


@pytest.mark.parametrize(
    "path",
    [
        "missing.jpg",
        "notes.txt",
        "corrupt.jpg",
        "escape.jpg",
        "%2E%2E/outside.jpg",
        "nested%20dir/%2E%2E/%2E%2E/outside.jpg",
    ],
)
def test_thumbnail_route_returns_404_for_unsafe_missing_or_invalid_images(
    thumbnail_client, path
):
    response = thumbnail_client.get(f"/thumbnails/{path}")

    assert response.status_code == 404
    assert "outside" not in response.text
