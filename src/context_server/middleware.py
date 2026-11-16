from __future__ import annotations

import math
import time
from collections import deque
from urllib.parse import urlsplit

from starlette.types import ASGIApp, Message, Receive, Scope, Send

MAX_REQUEST_BODY_BYTES = 1024 * 1024
MAX_REQUESTS_PER_MINUTE = 600
RATE_LIMIT_WINDOW_SECONDS = 60.0


class RequestBoundaryMiddleware:
    """Enforce request boundaries before MCP and OAuth routes run."""

    def __init__(
        self,
        app: ASGIApp,
        allowed_hosts: tuple[str, ...],
        allowed_origins: tuple[str, ...],
    ) -> None:
        self.app = app
        self.allowed_hosts = frozenset(host.lower().rstrip(".") for host in allowed_hosts)
        self.allowed_origins = frozenset(origin.lower() for origin in allowed_origins)
        self._request_timestamps: deque[float] = deque()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = scope.get("headers", [])
        hosts = [value for name, value in headers if name.lower() == b"host"]
        origins = [value for name, value in headers if name.lower() == b"origin"]
        if len(hosts) != 1 or len(origins) > 1:
            await self._forbidden(send)
            return

        try:
            host = hosts[0].decode("ascii")
            parsed_host = urlsplit(f"//{host}")
            hostname = parsed_host.hostname
            _ = parsed_host.port
        except (UnicodeDecodeError, ValueError):
            await self._forbidden(send)
            return

        if (
            hostname is None
            or parsed_host.path
            or parsed_host.query
            or parsed_host.fragment
            or parsed_host.username is not None
            or parsed_host.password is not None
            or hostname.lower().rstrip(".") not in self.allowed_hosts
        ):
            await self._forbidden(send)
            return

        if origins:
            try:
                origin = origins[0].decode("ascii").lower()
                parsed_origin = urlsplit(origin)
                _ = parsed_origin.port
            except (UnicodeDecodeError, ValueError):
                await self._forbidden(send)
                return
            if (
                parsed_origin.scheme not in {"http", "https"}
                or not parsed_origin.netloc
                or parsed_origin.path
                or parsed_origin.query
                or parsed_origin.fragment
                or parsed_origin.username is not None
                or parsed_origin.password is not None
                or origin not in self.allowed_origins
            ):
                await self._forbidden(send)
                return

        if not self._allow_request():
            retry_after = max(
                1,
                math.ceil(
                    RATE_LIMIT_WINDOW_SECONDS - (time.monotonic() - self._request_timestamps[0])
                ),
            )
            await self._respond(
                send,
                429,
                b"Too many requests",
                [(b"retry-after", str(retry_after).encode("ascii"))],
            )
            return

        content_lengths = [value for name, value in headers if name.lower() == b"content-length"]
        if len(content_lengths) > 1:
            await self._respond(send, 400, b"Invalid Content-Length")
            return
        if content_lengths:
            try:
                content_length = int(content_lengths[0])
            except ValueError:
                await self._respond(send, 400, b"Invalid Content-Length")
                return
            if content_length < 0:
                await self._respond(send, 400, b"Invalid Content-Length")
                return
            if content_length > MAX_REQUEST_BODY_BYTES:
                await self._respond(send, 413, b"Request body too large")
                return

        body = bytearray()
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            body.extend(message.get("body", b""))
            if len(body) > MAX_REQUEST_BODY_BYTES:
                await self._respond(send, 413, b"Request body too large")
                return
            more_body = message.get("more_body", False)

        body_message: Message = {
            "type": "http.request",
            "body": bytes(body),
            "more_body": False,
        }
        body_pending = True

        async def replay_receive() -> Message:
            nonlocal body_pending
            if body_pending:
                body_pending = False
                return body_message
            return await receive()

        await self.app(scope, replay_receive, send)

    def _allow_request(self) -> bool:
        now = time.monotonic()
        while (
            self._request_timestamps
            and now - self._request_timestamps[0] >= RATE_LIMIT_WINDOW_SECONDS
        ):
            self._request_timestamps.popleft()
        if len(self._request_timestamps) >= MAX_REQUESTS_PER_MINUTE:
            return False
        self._request_timestamps.append(now)
        return True

    @staticmethod
    async def _forbidden(send: Send) -> None:
        await RequestBoundaryMiddleware._respond(send, 403, b"Forbidden")

    @staticmethod
    async def _respond(
        send: Send, status: int, body: bytes, extra_headers: list[tuple[bytes, bytes]] | None = None
    ) -> None:
        headers = [
            (b"content-type", b"text/plain; charset=utf-8"),
            (b"content-length", str(len(body)).encode("ascii")),
            (b"cache-control", b"no-store"),
        ]
        response_start: Message = {
            "type": "http.response.start",
            "status": status,
            "headers": headers + (extra_headers or []),
        }
        await send(response_start)
        response_body: Message = {"type": "http.response.body", "body": body}
        await send(response_body)
