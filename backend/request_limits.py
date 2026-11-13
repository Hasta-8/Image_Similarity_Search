"""Request-body limits for public demo deployments.

``RequestBodyLimitMiddleware`` is a pure ASGI middleware that bounds how much
request body a client may send and how long the server waits for it:

* a ``Content-Length`` above the limit is rejected with HTTP 413 before any
  body is read;
* chunked bodies are counted as they arrive and rejected with HTTP 413 once the
  limit is crossed, even when no ``Content-Length`` header is present;
* a body that stops arriving for longer than the read timeout is aborted with
  HTTP 408.

Only methods that carry bodies are checked. The middleware sits outside the
FastAPI exception handling, so it can send its own response for these cases.
"""

from __future__ import annotations

import asyncio
import json
from typing import Awaitable, Callable, Optional

Scope = dict
Message = dict
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]

BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})


class _BodyAborted(Exception):
    """Internal signal that the request body was rejected mid-stream."""


class RequestBodyLimitMiddleware:
    def __init__(
        self,
        app,
        *,
        max_body_bytes: int,
        read_timeout_seconds: float,
    ) -> None:
        if max_body_bytes < 1:
            raise ValueError("max_body_bytes must be >= 1")
        if read_timeout_seconds <= 0:
            raise ValueError("read_timeout_seconds must be > 0")
        self.app = app
        self.max_body_bytes = int(max_body_bytes)
        self.read_timeout_seconds = float(read_timeout_seconds)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") not in BODY_METHODS:
            await self.app(scope, receive, send)
            return

        content_length = _content_length(scope)
        if content_length is not None and content_length > self.max_body_bytes:
            await _send_error(send, 413, _too_large_detail(self.max_body_bytes))
            return

        # The multipart parser inside the app swallows errors raised by
        # ``receive`` and turns them into its own 400. The abort is therefore
        # recorded here and enforced by this middleware, which discards any
        # response the app produces after the abort.
        abort: dict = {"status": None, "detail": ""}
        forwarded_start = False
        received = 0

        async def tracking_send(message: Message) -> None:
            nonlocal forwarded_start
            if abort["status"] is not None:
                return
            if message["type"] == "http.response.start":
                forwarded_start = True
            await send(message)

        def _abort(status: int, detail: str) -> None:
            abort["status"] = status
            abort["detail"] = detail

        async def limited_receive() -> Message:
            nonlocal received
            if abort["status"] is not None:
                raise _BodyAborted()
            try:
                message = await asyncio.wait_for(
                    receive(), timeout=self.read_timeout_seconds
                )
            except asyncio.TimeoutError:
                _abort(408, "Request body was not received in time")
                raise _BodyAborted()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    _abort(413, _too_large_detail(self.max_body_bytes))
                    raise _BodyAborted()
            return message

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyAborted:
            pass
        except Exception:
            if abort["status"] is None:
                raise

        if abort["status"] is not None and not forwarded_start:
            await _send_error(send, abort["status"], abort["detail"])


def _content_length(scope: Scope) -> Optional[int]:
    for name, value in scope.get("headers", []):
        if name == b"content-length":
            try:
                return int(value.decode("latin-1"))
            except ValueError:
                return None
    return None


def _too_large_detail(limit_bytes: int) -> str:
    limit_mib = limit_bytes / (1024 * 1024)
    return f"Upload exceeds the {limit_mib:g} MiB request limit"


async def _send_error(send: Send, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"connection", b"close"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})
