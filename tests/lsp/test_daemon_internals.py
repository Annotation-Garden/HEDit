"""In-process tests for the hedit-lspd daemon's internals.

The end-to-end test in test_daemon_lifecycle.py launches the daemon
in a subprocess, which exercises the SIGTERM/exit-code path but leaves
no coverage in the parent test process. This module exercises the same
classes directly so coverage measurement sees them, while still using
the real Node hed-lsp child and a real Unix socket -- no mocks.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.lsp import HedLspClient
from src.lsp.daemon import (
    _Daemon,
    _NodeChild,
    default_runtime_dir,
    meta_file_path,
    pid_file_path,
    socket_file_path,
)


@pytest.fixture
def short_runtime_dir() -> Iterator[Path]:
    """Daemon runtime under /tmp so the Unix socket path stays under 104 bytes."""
    with tempfile.TemporaryDirectory(prefix="hedit-lspd-int-", dir="/tmp") as path:
        yield Path(path)


def test_runtime_path_helpers_use_runtime_dir(tmp_path: Path) -> None:
    """The helper functions derive every path from the runtime dir argument."""
    assert pid_file_path(tmp_path).parent == tmp_path
    assert socket_file_path(tmp_path).parent == tmp_path
    assert meta_file_path(tmp_path).parent == tmp_path
    assert pid_file_path(tmp_path).name == "lspd.pid"
    assert socket_file_path(tmp_path).name == "lspd.sock"
    assert meta_file_path(tmp_path).name == "lspd.meta.json"


def test_default_runtime_dir_is_per_user() -> None:
    """default_runtime_dir() returns a per-user path with the hedit subdir."""
    path = default_runtime_dir()
    assert path.name == "hedit"


async def test_node_child_request_response(hed_lsp_server_js: Path) -> None:
    """Spawn _NodeChild directly, drive an initialize+hed/suggest cycle, shut down."""
    node = await _NodeChild.spawn(hed_lsp_server_js)
    try:
        init_queue: asyncio.Queue = asyncio.Queue()
        suggest_queue: asyncio.Queue = asyncio.Queue()

        # LSP initialize handshake. claim_response_for routes the matching
        # response into a dedicated queue, so unrelated server-side
        # notifications (window/showMessage, hed/modelProgress, ...) don't
        # collide with the response we're waiting on.
        node.claim_response_for(1, init_queue)
        await node.send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"processId": None, "rootUri": None, "capabilities": {}},
            }
        )
        init_resp = await asyncio.wait_for(init_queue.get(), timeout=15.0)
        assert init_resp["id"] == 1
        assert "capabilities" in init_resp.get("result", {})

        await node.send({"jsonrpc": "2.0", "method": "initialized", "params": {}})

        # Real hed/suggest round-trip through the spawned Node child.
        node.claim_response_for(2, suggest_queue)
        await node.send(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "hed/suggest",
                "params": {"queries": ["red square"], "schema": "8.4.0", "top": 5},
            }
        )
        suggest_resp = await asyncio.wait_for(suggest_queue.get(), timeout=10.0)
        assert suggest_resp["id"] == 2
        assert "Red" in suggest_resp["result"]["red square"]
    finally:
        await node.shutdown()


async def test_node_child_broadcast_queue_receives_notifications(
    hed_lsp_server_js: Path,
) -> None:
    """Server notifications (no JSON-RPC id) fan out to broadcast queues."""
    node = await _NodeChild.spawn(hed_lsp_server_js)
    try:
        bcast: asyncio.Queue = asyncio.Queue()
        node.add_broadcast_queue(bcast)
        try:
            await node.send(
                {
                    "jsonrpc": "2.0",
                    "id": 99,
                    "method": "initialize",
                    "params": {"processId": None, "rootUri": None, "capabilities": {}},
                }
            )
            # Drain until we see at least one unclaimed message (the
            # initialize response with id=99 also routes here since we did
            # not claim that id).
            seen = await asyncio.wait_for(bcast.get(), timeout=15.0)
            assert isinstance(seen, dict)
        finally:
            node.remove_broadcast_queue(bcast)
    finally:
        await node.shutdown()


async def test_daemon_in_process_lifecycle(
    hed_lsp_server_js: Path, short_runtime_dir: Path
) -> None:
    """Build a _Daemon in the test process, connect a client through it, then stop.

    This exercises the socket-bind, PID/meta-write, peer-connection
    handling, and stop() cleanup paths inside the same process the
    coverage tool measures, lifting coverage on src/lsp/daemon.py above
    what the subprocess-based lifecycle test alone can reach.
    """
    daemon = _Daemon(
        server_js=hed_lsp_server_js,
        runtime_dir=short_runtime_dir,
        node_path="node",
    )
    await daemon.start()
    try:
        socket = socket_file_path(short_runtime_dir)
        assert socket.exists()
        assert pid_file_path(short_runtime_dir).exists()
        assert meta_file_path(short_runtime_dir).exists()

        client = await HedLspClient.connect_unix(socket)
        try:
            result = await client.suggest("button press")
            assert result.success, result.error
            assert result.raw["button press"], "expected at least one suggestion"
        finally:
            await client.shutdown()
    finally:
        await daemon.stop()

    # stop() must clean up the runtime files so a subsequent start
    # doesn't see stale state.
    assert not socket_file_path(short_runtime_dir).exists()
    assert not pid_file_path(short_runtime_dir).exists()
    assert not meta_file_path(short_runtime_dir).exists()
