"""Live single-origin test for the local demo runtime.

Starts the real Uvicorn process (one worker, as the demo script does) against:

* a temporary dataset with nested folders, spaces, NFC Unicode names and
  duplicate basenames;
* a candidate built with the real shared CLIP service and promoted into a
  temporary release store (never the production store);
* the production React build served by the same process.

Requires ``frontend/build`` (``REACT_APP_API_BASE_URL= npm run build``) and the
CLIP checkpoint in the local Hugging Face cache. Skipped when the build is
absent. Production artifacts and ``backend/artifacts`` are never touched.
"""

from __future__ import annotations

import io
import os
import socket
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote

import httpx
import numpy as np
import pytest
from PIL import Image, ImageDraw

import rebuild_index
from embeddings import get_embedding_service
from release_store import ReleaseStore

BACKEND_DIR = Path(__file__).resolve().parents[1]
FRONTEND_BUILD = BACKEND_DIR.parent / "frontend" / "build"
UPLOAD_LIMIT_BYTES = 1024 * 1024  # small limit so the oversize test stays cheap

pytestmark = pytest.mark.skipif(
    not (FRONTEND_BUILD / "index.html").is_file(),
    reason="frontend production build missing; run REACT_APP_API_BASE_URL= npm run build",
)

# (dataset-relative path as stored, seed). "gallery-a/photo.jpg" and
# "gallery-b/photo.jpg" are duplicate basenames in different folders.
DATASET_FILES = [
    ("gallery-a/photo.jpg", 1),
    ("gallery-b/photo.jpg", 2),
    ("nested dir/cat one.jpg", 3),
    ("caf\u00e9.jpg", 4),
    ("\u00dcn\u00efcode dir/\u65e5\u672c scene.png", 5),
]


def _patterned_image(seed: int, size=(128, 96)) -> Image.Image:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, size[0] // 2, size[1] // 2), outline=(255, 255, 255), width=3)
    draw.line((0, size[1] - 1, size[0] - 1, 0), fill=(255, 0, 0), width=2)
    return image


def _encoded_image_url(relative_path: str) -> str:
    return "/images/" + "/".join(quote(part, safe="") for part in relative_path.split("/"))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_ready(base: str, proc: subprocess.Popen, log_path: Path, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited early ({proc.returncode}):\n{log_path.read_text()[-3000:]}"
            )
        try:
            if httpx.get(f"{base}/ready", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"server not ready after {timeout}s:\n{log_path.read_text()[-3000:]}")


@pytest.fixture(scope="module")
def live_demo(tmp_path_factory):
    root = tmp_path_factory.mktemp("demo-live")
    dataset = root / "dataset"
    for relative_path, seed in DATASET_FILES:
        path = dataset / unicodedata.normalize("NFC", relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fmt = "PNG" if path.suffix.lower() == ".png" else "JPEG"
        _patterned_image(seed).save(path, format=fmt)

    service = get_embedding_service(device="cpu", batch_size=4)
    embedding_metadata = rebuild_index.get_embedding_metadata_from_service(service)
    candidate = root / "candidate"
    rebuild_index.rebuild_candidate_index(
        dataset_dir=dataset,
        output_dir=candidate,
        batch_size=4,
        embedding_metadata=embedding_metadata,
        generated_at_utc="2026-01-01T00:00:00Z",
    )
    store_root = root / "release-store"
    plan = ReleaseStore(store_root).promote(candidate, confirm=True, dataset_root=dataset)

    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    env = {
        **os.environ,
        "IMGSIMI_DATASET_DIR": str(dataset),
        "IMGSIMI_RELEASE_STORE": str(store_root),
        "IMGSIMI_FRONTEND_DIR": str(FRONTEND_BUILD),
        "IMGSIMI_ENABLE_DOCS": "0",
        "IMGSIMI_MAX_UPLOAD_BYTES": str(UPLOAD_LIMIT_BYTES),
        "IMGSIMI_WARMUP_MODEL": "1",
        "PYTHONUNBUFFERED": "1",
    }
    log_path = root / "server.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn", "main:app",
                "--host", "127.0.0.1", "--port", str(port), "--workers", "1",
            ],
            cwd=BACKEND_DIR,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:
        _wait_ready(base, proc, log_path, timeout=300)
        yield {
            "base": base,
            "dataset": dataset,
            "release_id": plan.release_id,
            "root": root,
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def test_root_serves_built_react_app(live_demo):
    response = httpx.get(f"{live_demo['base']}/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<div id="root">' in response.text


def test_built_bundle_has_no_localhost_api_urls():
    scanned = list((FRONTEND_BUILD / "static").rglob("*.js"))
    assert scanned, "no built JavaScript found"
    for path in scanned:
        text = path.read_text(encoding="utf-8", errors="replace")
        assert "localhost:8000" not in text, path.name
        assert "http://localhost" not in text, path.name
        assert "127.0.0.1:8000" not in text, path.name
    bundle = next(p for p in scanned if p.name.startswith("main."))
    assert "/search-image" in bundle.read_text(encoding="utf-8")


def test_health_and_ready_report_release_mode(live_demo):
    health = httpx.get(f"{live_demo['base']}/health").json()
    assert health["ready"] is True
    assert health["runtime_mode"] == "release"
    assert health["active_release_id"] == live_demo["release_id"]

    ready = httpx.get(f"{live_demo['base']}/ready")
    assert ready.status_code == 200
    assert ready.json()["indexed_images"] == len(DATASET_FILES)


def test_api_docs_and_openapi_are_not_exposed(live_demo):
    assert httpx.get(f"{live_demo['base']}/docs").status_code == 404
    assert httpx.get(f"{live_demo['base']}/openapi.json").status_code == 404


@pytest.mark.parametrize("relative_path,seed", DATASET_FILES)
def test_exact_image_search_and_image_url_round_trip(live_demo, relative_path, seed):
    stored = unicodedata.normalize("NFC", relative_path)
    source = live_demo["dataset"] / stored
    mime = "image/png" if source.suffix.lower() == ".png" else "image/jpeg"
    response = httpx.post(
        f"{live_demo['base']}/search-image",
        files={"file": (source.name, source.read_bytes(), mime)},
        timeout=120,
    )
    assert response.status_code == 200, response.text
    body_text = response.text
    assert "localhost" not in body_text
    assert str(live_demo["root"]) not in body_text

    top = response.json()["results"][0]
    assert top["path"] == stored
    assert top["image_url"] == _encoded_image_url(stored)
    assert top["score"] > 0.99

    image = httpx.get(f"{live_demo['base']}{top['image_url']}")
    assert image.status_code == 200
    assert image.content == source.read_bytes()


def test_duplicate_basenames_resolve_to_their_own_files(live_demo):
    for relative in ("gallery-a/photo.jpg", "gallery-b/photo.jpg"):
        source = live_demo["dataset"] / relative
        response = httpx.post(
            f"{live_demo['base']}/search-image",
            files={"file": ("q.jpg", source.read_bytes(), "image/jpeg")},
            timeout=120,
        )
        top = response.json()["results"][0]
        assert top["path"] == relative
        image = httpx.get(f"{live_demo['base']}{top['image_url']}")
        assert image.content == source.read_bytes()


def test_content_length_over_limit_is_rejected(live_demo):
    payload = b"\xff" * (UPLOAD_LIMIT_BYTES + 1024)
    response = httpx.post(
        f"{live_demo['base']}/search-image",
        content=payload,
        headers={"content-type": "image/jpeg"},
        timeout=60,
    )
    assert response.status_code == 413


def test_chunked_body_over_limit_is_rejected(live_demo):
    # A multipart body without Content-Length is sent chunked; Starlette only
    # reads multipart bodies, so the limit must be enforced while streaming.
    boundary = "imgsimi-boundary"
    head = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="big.jpg"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()

    def chunks():
        yield head
        for _ in range(4):
            yield b"\xff" * (UPLOAD_LIMIT_BYTES // 2)
        yield tail

    try:
        response = httpx.post(
            f"{live_demo['base']}/search-image",
            content=chunks(),
            headers={"content-type": f"multipart/form-data; boundary={boundary}"},
            timeout=60,
        )
    except httpx.TransportError:
        # The server may close the connection once the limit is crossed. The
        # body is still rejected; no search result is produced.
        return
    assert response.status_code == 413, response.text


def test_non_image_upload_is_rejected(live_demo):
    response = httpx.post(
        f"{live_demo['base']}/search-image",
        files={"file": ("notes.txt", b"hello", "text/plain")},
        timeout=60,
    )
    assert response.status_code == 400


def test_small_valid_upload_within_limit_succeeds(live_demo):
    buffer = io.BytesIO()
    _patterned_image(99, size=(64, 64)).save(buffer, format="JPEG")
    assert len(buffer.getvalue()) < UPLOAD_LIMIT_BYTES
    response = httpx.post(
        f"{live_demo['base']}/search-image",
        files={"file": ("small.jpg", buffer.getvalue(), "image/jpeg")},
        timeout=120,
    )
    assert response.status_code == 200
    assert len(response.json()["results"]) == min(5, len(DATASET_FILES))
