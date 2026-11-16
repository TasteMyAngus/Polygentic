from collections.abc import AsyncIterator

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from context_server.middleware import (
    MAX_REQUEST_BODY_BYTES,
    MAX_REQUESTS_PER_MINUTE,
    RequestBoundaryMiddleware,
)


def _app() -> ASGIApp:
    async def application(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    return application


async def _body_echo(request: Request) -> Response:
    return Response(str(len(await request.body())))


def _body_app() -> ASGIApp:
    return Starlette(routes=[Route("/", _body_echo, methods=["GET", "POST"])])


@pytest.mark.asyncio
async def test_boundary_allows_configured_host_and_origin() -> None:
    app = RequestBoundaryMiddleware(
        _app(),
        allowed_hosts=("127.0.0.1",),
        allowed_origins=("http://localhost:3000",),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1:8000",
    ) as client:
        response = await client.get("/", headers={"Origin": "http://localhost:3000"})

    assert response.status_code == 204


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "headers"),
    [
        ("http://attacker.example/", {}),
        ("http://127.0.0.1:8000/", {"Origin": "https://attacker.example"}),
        ("http://127.0.0.1:8000/", {"Origin": "null"}),
    ],
)
async def test_boundary_rejects_untrusted_host_or_origin(url: str, headers: dict[str, str]) -> None:
    app = RequestBoundaryMiddleware(
        _app(),
        allowed_hosts=("127.0.0.1",),
        allowed_origins=("http://localhost:3000",),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1:8000",
    ) as client:
        response = await client.get(url, headers=headers)

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_boundary_rejects_oversized_content_length() -> None:
    app = RequestBoundaryMiddleware(_body_app(), ("localhost",), ())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://localhost",
    ) as client:
        response = await client.post("/", content=b"x" * (MAX_REQUEST_BODY_BYTES + 1))

    assert response.status_code == 413
    assert response.text == "Request body too large"


@pytest.mark.asyncio
async def test_boundary_rejects_oversized_streamed_body() -> None:
    app = RequestBoundaryMiddleware(_body_app(), ("localhost",), ())

    async def stream_body() -> AsyncIterator[bytes]:
        yield b"x" * MAX_REQUEST_BODY_BYTES
        yield b"x"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://localhost",
    ) as client:
        response = await client.post("/", content=stream_body())

    assert response.status_code == 413
    assert response.text == "Request body too large"


@pytest.mark.asyncio
async def test_boundary_rejects_oversized_streamed_get_body() -> None:
    app = RequestBoundaryMiddleware(_body_app(), ("localhost",), ())

    async def stream_body() -> AsyncIterator[bytes]:
        yield b"x" * MAX_REQUEST_BODY_BYTES
        yield b"x"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://localhost",
    ) as client:
        response = await client.request("GET", "/", content=stream_body())

    assert response.status_code == 413
    assert response.text == "Request body too large"


@pytest.mark.asyncio
async def test_boundary_applies_process_rate_limit() -> None:
    app = RequestBoundaryMiddleware(_app(), ("localhost",), ())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://localhost",
    ) as client:
        for _ in range(MAX_REQUESTS_PER_MINUTE):
            response = await client.get("/")
            assert response.status_code == 204
        limited = await client.get("/")

    assert limited.status_code == 429
    assert limited.headers["retry-after"].isdigit()
