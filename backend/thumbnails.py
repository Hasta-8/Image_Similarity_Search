"""Small JPEG previews of dataset images for the results grid.

Dataset originals are often 4-14 MB camera JPEGs. The UI only needs a few
hundred pixels per result, so the API serves downscaled JPEGs instead. They are
generated on first request and kept in a bounded in-memory LRU cache.

Previews apply the EXIF orientation tag so they display the same way browsers
display the originals. This affects display only; query and index embeddings
are unchanged.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

THUMBNAIL_SUFFIXES = {".jpg", ".jpeg", ".png"}
JPEG_QUALITY = 82


class ThumbnailError(ValueError):
    """Raised when a source image cannot be turned into a thumbnail."""


def make_thumbnail(source: Path, max_side: int) -> bytes:
    """Return JPEG bytes whose longest side is at most ``max_side`` pixels."""
    if max_side < 1:
        raise ValueError("max_side must be >= 1")
    try:
        with Image.open(source) as image:
            # For JPEGs, let the decoder downscale while reading. This is much
            # faster and lighter than decoding a full-resolution camera image.
            image.draft("RGB", (max_side, max_side))
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ThumbnailError(f"Cannot create thumbnail for {source.name}: {exc}") from exc
    return buffer.getvalue()


class ThumbnailCache:
    """Thread-safe LRU cache of thumbnail bytes, bounded by total size.

    Keys include the file's modification time and size, so an edited source
    image gets a fresh thumbnail.
    """

    def __init__(self, max_side: int, max_bytes: int) -> None:
        if max_side < 1:
            raise ValueError("max_side must be >= 1")
        if max_bytes < 0:
            raise ValueError("max_bytes must be >= 0")
        self.max_side = max_side
        self.max_bytes = max_bytes
        self._items: OrderedDict[tuple, bytes] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, source: Path) -> bytes:
        stat = source.stat()
        key = (str(source), stat.st_mtime_ns, stat.st_size, self.max_side)
        with self._lock:
            cached = self._items.get(key)
            if cached is not None:
                self._items.move_to_end(key)
                return cached

        data = make_thumbnail(source, self.max_side)

        with self._lock:
            if key not in self._items and len(data) <= self.max_bytes:
                self._items[key] = data
                self._bytes += len(data)
                while self._bytes > self.max_bytes:
                    _, evicted = self._items.popitem(last=False)
                    self._bytes -= len(evicted)
        return data

    @property
    def size_bytes(self) -> int:
        with self._lock:
            return self._bytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
