from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from context_server.config import ProviderCredential, Settings
from context_server.server import create_app


class FakeEmbedder:
    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [tuple([1.0] + [0.0] * 383) for _ in texts]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--token-sha256", required=True)
    parser.add_argument("--claude-token-sha256", required=True)
    args = parser.parse_args()
    app = create_app(
        Settings(
            credentials=(
                ProviderCredential("codex", args.token_sha256),
                ProviderCredential("claude", args.claude_token_sha256),
            ),
            data_dir=Path(args.data_dir),
        ),
        embedder=FakeEmbedder(),
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="error")


if __name__ == "__main__":
    main()
