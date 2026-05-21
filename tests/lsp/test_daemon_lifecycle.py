"""Real-LSP integration tests for the hedit-lspd daemon.

Spawns the actual daemon supervisor in a child process bound to a
temporary runtime dir, connects to it over the Unix socket, issues a
real `hed/suggest` query, then shuts the daemon down. No mocks.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.lsp import HedLspClient
from src.lsp.daemon import meta_file_path, pid_file_path, socket_file_path

pytestmark = pytest.mark.integration


@pytest.fixture
def short_runtime_dir() -> Iterator[Path]:
    """A short tmp dir for the daemon socket.

    macOS caps `AF_UNIX` paths at 104 bytes; pytest's `tmp_path` lives
    under `/private/var/folders/...` which blows that limit. We allocate
    under `/tmp` with a short prefix so `lspd.sock` fits.
    """
    with tempfile.TemporaryDirectory(prefix="hedit-lspd-", dir="/tmp") as path:
        yield Path(path)


async def _wait_for_socket(socket_path: Path, timeout: float) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if socket_path.exists():
            return
        await asyncio.sleep(0.05)
    raise TimeoutError(f"Daemon never created socket at {socket_path}")


async def test_daemon_lifecycle_end_to_end(
    hed_lsp_server_js: Path,
    short_runtime_dir: Path,
) -> None:
    runtime_dir = short_runtime_dir
    env = {**os.environ, "HED_LSP_SERVER_JS": str(hed_lsp_server_js)}
    daemon_proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "src.lsp.daemon",
        "--runtime-dir",
        str(runtime_dir),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        socket_path = socket_file_path(runtime_dir)
        await _wait_for_socket(socket_path, timeout=20.0)

        # PID and meta files must be present once the daemon is ready.
        assert pid_file_path(runtime_dir).exists()
        assert meta_file_path(runtime_dir).exists()

        client = await HedLspClient.connect_unix(socket_path)
        try:
            result = await client.suggest("red square")
            assert result.success, result.error
            assert "Red" in result.raw["red square"]
        finally:
            await client.shutdown()
    finally:
        if daemon_proc.returncode is None:
            daemon_proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(daemon_proc.wait(), timeout=10.0)
            except TimeoutError:
                daemon_proc.kill()
                await daemon_proc.wait()
        else:
            stdout, stderr = await daemon_proc.communicate()
            pytest.fail(
                f"daemon exited early with code {daemon_proc.returncode}\n"
                f"stdout: {stdout.decode(errors='replace')}\n"
                f"stderr: {stderr.decode(errors='replace')}"
            )

    # After SIGTERM the daemon should remove its socket + PID/meta files.
    assert not socket_file_path(runtime_dir).exists()
    assert not pid_file_path(runtime_dir).exists()
    assert not meta_file_path(runtime_dir).exists()
