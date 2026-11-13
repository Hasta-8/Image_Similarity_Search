"""Unit tests for the public-demo runtime limits.

Covers request-body limits (413/408), search concurrency (429), inference
timeout (504) and API docs exposure. No CLIP model is required: inference is
replaced with deterministic fakes. The live single-origin test lives in
``test_demo_single_origin_live.py``.
"""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from faiss_index import add_embeddings, create_faiss_index
from request_limits import RequestBodyLimitMiddleware


def _unit_vector(dim: int = 512) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    vector[0] = 1.0
    return vector


@pytest.fixture
def main_module(monkeypatch):
    import main

    monkeypatch.setattr(main, "_search_inflight", 0)
    return main


# --- Request body limits -----------------------------------------------------


def _limited_echo_app(max_body_bytes: int, read_timeout: float = 5.0) -> FastAPI:
    app = FastAPI()

    @app.post("/echo")
    async def echo(request: Request):
        body = await request.body()
        return {"size": len(body)}

    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_body_bytes=max_body_bytes,
        read_timeout_seconds=read_timeout,
    )
    return app


def test_body_within_limit_is_accepted():
    client = TestClient(_limited_echo_app(max_body_bytes=1024))
    response = client.post("/echo", content=b"x" * 1024)
    assert response.status_code == 200
    assert response.json() == {"size": 1024}


def test_content_length_over_limit_is_rejected_with_413():
    client = TestClient(_limited_echo_app(max_body_bytes=1024))
    response = client.post("/echo", content=b"x" * 1025)
    assert response.status_code == 413
    assert "detail" in response.json()


def _run_asgi(app, *, headers, chunks, stall=False):
    """Drive an ASGI app directly, optionally with a chunked or stalled body."""
    sent = []
    pending = list(chunks)

    async def receive():
        if stall:
            await asyncio.Event().wait()
        if pending:
            body = pending.pop(0)
            return {"type": "http.request", "body": body, "more_body": bool(pending)}
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/echo",
        "headers": headers,
        "query_string": b"",
    }
    asyncio.run(app(scope, receive, send))
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    return status, sent


async def _drain_app(scope, receive, send):
    while True:
        message = await receive()
        if not message.get("more_body"):
            break
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok", "more_body": False})


def test_chunked_body_over_limit_is_rejected_with_413():
    app = RequestBodyLimitMiddleware(
        _drain_app, max_body_bytes=1000, read_timeout_seconds=5.0
    )
    chunks = [b"a" * 600, b"b" * 600]  # no Content-Length: counted as it arrives
    status, _ = _run_asgi(app, headers=[], chunks=chunks)
    assert status == 413


def test_chunked_body_within_limit_is_accepted():
    app = RequestBodyLimitMiddleware(
        _drain_app, max_body_bytes=1000, read_timeout_seconds=5.0
    )
    status, _ = _run_asgi(app, headers=[], chunks=[b"a" * 400, b"b" * 400])
    assert status == 200


def test_stalled_body_is_aborted_with_408():
    app = RequestBodyLimitMiddleware(
        _drain_app, max_body_bytes=1000, read_timeout_seconds=0.2
    )
    status, _ = _run_asgi(app, headers=[], chunks=[], stall=True)
    assert status == 408


def test_invalid_limit_configuration_is_rejected():
    with pytest.raises(ValueError):
        RequestBodyLimitMiddleware(_drain_app, max_body_bytes=0, read_timeout_seconds=1)
    with pytest.raises(ValueError):
        RequestBodyLimitMiddleware(_drain_app, max_body_bytes=1, read_timeout_seconds=0)


def test_get_requests_are_not_body_limited(main_module):
    client = TestClient(main_module.app)
    response = client.get("/health")
    assert response.status_code == 200


# --- Search concurrency and timeout ------------------------------------------


def test_search_concurrency_cap_returns_busy(main_module, monkeypatch):
    monkeypatch.setattr(main_module, "SEARCH_MAX_CONCURRENT", 1)
    monkeypatch.setattr(main_module, "SEARCH_TIMEOUT_SECONDS", 10.0)
    gate = threading.Event()

    def slow_embedding(source):
        gate.wait(10)
        return _unit_vector()

    monkeypatch.setattr(main_module, "get_image_embedding", slow_embedding)

    async def scenario():
        first = asyncio.ensure_future(main_module._embed_query_with_limits(b"one"))
        await asyncio.sleep(0.05)
        with pytest.raises(main_module.SearchBusyError):
            await main_module._embed_query_with_limits(b"two")
        gate.set()
        result = await first
        await asyncio.sleep(0)
        return result

    result = asyncio.run(scenario())
    assert result.shape == (512,)
    assert main_module._search_inflight == 0


def test_search_timeout_keeps_slot_until_worker_finishes(main_module, monkeypatch):
    monkeypatch.setattr(main_module, "SEARCH_MAX_CONCURRENT", 1)
    monkeypatch.setattr(main_module, "SEARCH_TIMEOUT_SECONDS", 0.1)

    def slow_embedding(source):
        time.sleep(0.6)
        return _unit_vector()

    monkeypatch.setattr(main_module, "get_image_embedding", slow_embedding)

    async def scenario():
        with pytest.raises(main_module.SearchTimeoutError):
            await main_module._embed_query_with_limits(b"slow")
        # The abandoned worker still occupies the slot, so a new query is shed.
        assert main_module._search_inflight == 1
        with pytest.raises(main_module.SearchBusyError):
            await main_module._embed_query_with_limits(b"next")
        await asyncio.sleep(0.8)

    asyncio.run(scenario())
    assert main_module._search_inflight == 0


def test_search_endpoint_maps_busy_to_429(main_module, monkeypatch):
    client = TestClient(main_module.app)
    index = create_faiss_index(main_module.embedding_dim)
    add_embeddings(index, _unit_vector().reshape(1, -1), [1])
    main_module.index = index
    main_module.metadata = {1: "x.jpg"}
    monkeypatch.setattr(main_module, "_search_inflight", 99)
    response = client.post(
        "/search-image",
        files={"file": ("q.jpg", b"fake", "image/jpeg")},
    )
    assert response.status_code == 429
    assert response.headers.get("retry-after") == "5"


def test_search_endpoint_maps_timeout_to_504(main_module, monkeypatch):
    client = TestClient(main_module.app)
    index = create_faiss_index(main_module.embedding_dim)
    add_embeddings(index, _unit_vector().reshape(1, -1), [1])
    main_module.index = index
    main_module.metadata = {1: "x.jpg"}
    monkeypatch.setattr(main_module, "SEARCH_TIMEOUT_SECONDS", 0.05)

    def slow_embedding(source):
        time.sleep(0.5)
        return _unit_vector()

    monkeypatch.setattr(main_module, "get_image_embedding", slow_embedding)
    response = client.post(
        "/search-image",
        files={"file": ("q.jpg", b"fake", "image/jpeg")},
    )
    assert response.status_code == 504
    # Let the abandoned worker finish so later tests see a clean slot count.
    deadline = time.time() + 5
    while main_module._search_inflight and time.time() < deadline:
        time.sleep(0.05)


# --- Exposure ----------------------------------------------------------------


def test_api_docs_are_disabled_by_default(main_module):
    assert main_module.ENABLE_DOCS is False
    assert main_module.app.docs_url is None
    assert main_module.app.openapi_url is None
    client = TestClient(main_module.app)
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_upload_limit_default_is_16_mib(main_module):
    # Dataset photos reach ~14 MB, so the default must admit them.
    if "IMGSIMI_MAX_UPLOAD_BYTES" not in __import__("os").environ:
        assert main_module.MAX_UPLOAD_BYTES == 16 * 1024 * 1024
