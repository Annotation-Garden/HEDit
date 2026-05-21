"""Real-LSP integration tests for HedLspClient over stdio.

No mocks. Spawns the actual `node server.js --stdio` child, runs the
LSP initialize handshake, issues `hed/suggest` requests, and asserts
the server answered with the documented `{query: [tag, ...]}` shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.lsp import HedLspClient

pytestmark = pytest.mark.integration


async def test_initialize_and_suggest_single_query(hed_lsp_server_js: Path) -> None:
    client = await HedLspClient.spawn_stdio(hed_lsp_server_js)
    try:
        result = await client.suggest("red square")
        assert result.success, result.error
        assert "red square" in result.raw
        tags = result.raw["red square"]
        assert tags, "expected at least one suggestion for 'red square'"
        assert "Red" in tags
    finally:
        await client.shutdown()


async def test_batched_suggest_returns_one_entry_per_query(hed_lsp_server_js: Path) -> None:
    client = await HedLspClient.spawn_stdio(hed_lsp_server_js)
    try:
        queries = ("red square", "button press", "200ms duration")
        result = await client.suggest(*queries)
        assert result.success, result.error
        assert set(result.raw.keys()) == set(queries)
        for query in queries:
            assert result.raw[query], f"expected at least one suggestion for {query!r}"
        # Flat list deduplicates across queries.
        flat_tags = [s.tag for s in result.suggestions]
        assert len(flat_tags) == len(set(flat_tags)), "flat suggestion list should be deduplicated"
    finally:
        await client.shutdown()


async def test_empty_query_list_returns_failure(hed_lsp_server_js: Path) -> None:
    client = await HedLspClient.spawn_stdio(hed_lsp_server_js)
    try:
        result = await client.suggest()
        assert result.success is False
        assert result.error is not None
    finally:
        await client.shutdown()


async def test_concurrent_suggest_calls_are_demuxed(hed_lsp_server_js: Path) -> None:
    """Multiple in-flight requests should all resolve with their own results."""
    import asyncio

    client = await HedLspClient.spawn_stdio(hed_lsp_server_js)
    try:
        results = await asyncio.gather(
            client.suggest("red"),
            client.suggest("button"),
            client.suggest("flash"),
        )
        for result in results:
            assert result.success, result.error
            assert len(result.raw) == 1
    finally:
        await client.shutdown()


async def test_shutdown_is_idempotent(hed_lsp_server_js: Path) -> None:
    client = await HedLspClient.spawn_stdio(hed_lsp_server_js)
    await client.shutdown()
    await client.shutdown()  # must not raise


async def test_spawn_with_missing_server_js_raises() -> None:
    with pytest.raises(RuntimeError, match="not found"):
        await HedLspClient.spawn_stdio(Path("/nonexistent/server.js"))
