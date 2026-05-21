"""hedit-lspd: persistent hed-lsp daemon for opt-in CLI use.

Spawns one `node server.js --stdio` child, exposes a Unix socket, and
proxies LSP-framed messages bidirectionally between socket peers and
the child's stdio. All connected clients share the same Node process,
which keeps the HED schema and embeddings loaded in memory across
many `hedit annotate` invocations.

This is opt-in only. Users run `hedit lsp start` to launch the daemon;
nothing auto-spawns it. Stop with `hedit lsp stop`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

from src.lsp.protocol import LspProtocolError, read_message, write_message

logger = logging.getLogger(__name__)

DEFAULT_RUNTIME_SUBDIR = "hedit"
DEFAULT_PID_FILENAME = "lspd.pid"
DEFAULT_SOCKET_FILENAME = "lspd.sock"
DEFAULT_META_FILENAME = "lspd.meta.json"


def default_runtime_dir() -> Path:
    """Pick a per-user runtime directory across macOS/Linux."""
    env = os.environ.get("HEDIT_LSP_RUNTIME_DIR")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        return Path(xdg) / DEFAULT_RUNTIME_SUBDIR
    # macOS has no XDG_RUNTIME_DIR; fall back to user cache.
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / DEFAULT_RUNTIME_SUBDIR
    return Path.home() / ".cache" / DEFAULT_RUNTIME_SUBDIR


def pid_file_path(runtime_dir: Path | None = None) -> Path:
    return (runtime_dir or default_runtime_dir()) / DEFAULT_PID_FILENAME


def socket_file_path(runtime_dir: Path | None = None) -> Path:
    return (runtime_dir or default_runtime_dir()) / DEFAULT_SOCKET_FILENAME


def meta_file_path(runtime_dir: Path | None = None) -> Path:
    return (runtime_dir or default_runtime_dir()) / DEFAULT_META_FILENAME


class _NodeChild:
    """Owns the spawned `node server.js --stdio` process and serializes writes to its stdin."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("node child missing stdio pipes")
        self._process = process
        self._stdin = process.stdin
        self._stdout = process.stdout
        self._stderr = process.stderr
        self._write_lock = asyncio.Lock()
        self._pending: dict[int | str, asyncio.Queue[dict]] = {}
        self._broadcast: list[asyncio.Queue[dict]] = []
        self._reader_task = asyncio.create_task(self._read_loop())
        self._stderr_task = (
            asyncio.create_task(self._drain_stderr()) if process.stderr is not None else None
        )

    @classmethod
    async def spawn(cls, server_js: Path, node_path: str = "node") -> _NodeChild:
        if shutil.which(node_path) is None:
            raise RuntimeError(f"node binary not found on PATH (looked for {node_path!r})")
        if not server_js.exists():
            raise RuntimeError(f"LSP server entrypoint not found at {server_js}")
        process = await asyncio.create_subprocess_exec(
            node_path,
            str(server_js),
            "--stdio",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return cls(process)

    @property
    def pid(self) -> int:
        return self._process.pid

    def add_broadcast_queue(self, queue: asyncio.Queue[dict]) -> None:
        """Register a peer queue that should receive every server-originated notification."""
        self._broadcast.append(queue)

    def remove_broadcast_queue(self, queue: asyncio.Queue[dict]) -> None:
        try:
            self._broadcast.remove(queue)
        except ValueError:
            pass

    def claim_response_for(self, request_id: int | str, queue: asyncio.Queue[dict]) -> None:
        """Route the response with this id to the given peer's queue."""
        self._pending[request_id] = queue

    async def send(self, msg: dict) -> None:
        async with self._write_lock:
            await write_message(self._stdin, msg)  # type: ignore[arg-type]

    async def _read_loop(self) -> None:
        try:
            while True:
                try:
                    msg = await read_message(self._stdout)
                except LspProtocolError as exc:
                    logger.warning("Node child protocol error: %s", exc)
                    break
                if msg is None:
                    break
                msg_id = msg.get("id")
                if msg_id is not None and msg_id in self._pending:
                    queue = self._pending.pop(msg_id)
                    await queue.put(msg)
                else:
                    # Notification or unclaimed message; broadcast.
                    for queue in list(self._broadcast):
                        await queue.put(msg)
        finally:
            # Drop any waiters whose responses will never come.
            for queue in list(self._pending.values()):
                await queue.put(
                    {"jsonrpc": "2.0", "error": {"code": -32603, "message": "Node child exited"}}
                )
            self._pending.clear()

    async def _drain_stderr(self) -> None:
        assert self._stderr is not None
        while True:
            line = await self._stderr.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                logger.debug("[hed-lsp stderr] %s", text)

    async def shutdown(self) -> None:
        if self._process.returncode is None:
            try:
                self._stdin.close()
            except Exception:
                pass
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except TimeoutError:
                self._process.kill()
                await self._process.wait()
        for task in (self._reader_task, self._stderr_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass


class _Daemon:
    def __init__(self, server_js: Path, runtime_dir: Path, node_path: str) -> None:
        self._server_js = server_js
        self._runtime_dir = runtime_dir
        self._node_path = node_path
        self._socket_path = socket_file_path(runtime_dir)
        self._pid_path = pid_file_path(runtime_dir)
        self._meta_path = meta_file_path(runtime_dir)
        self._node: _NodeChild | None = None
        self._server: asyncio.AbstractServer | None = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._connections: set[asyncio.Task[None]] = set()
        self._started_at: datetime | None = None

    async def start(self) -> None:
        self._runtime_dir.mkdir(parents=True, exist_ok=True)
        if self._socket_path.exists():
            # Stale socket from a crashed prior daemon.
            self._socket_path.unlink()
        self._node = await _NodeChild.spawn(self._server_js, self._node_path)
        self._server = await asyncio.start_unix_server(
            self._handle_connection,
            path=str(self._socket_path),
        )
        os.chmod(self._socket_path, 0o600)
        self._started_at = datetime.now(UTC)
        self._write_pid_and_meta()
        logger.info(
            "hedit-lspd ready (pid=%s, socket=%s, node_pid=%s)",
            os.getpid(),
            self._socket_path,
            self._node.pid,
        )

    def _write_pid_and_meta(self) -> None:
        self._pid_path.write_text(str(os.getpid()))
        assert self._node is not None and self._started_at is not None
        self._meta_path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "node_pid": self._node.pid,
                    "socket": str(self._socket_path),
                    "server_js": str(self._server_js),
                    "started_at": self._started_at.isoformat(),
                }
            )
        )

    async def serve_until_signal(self) -> None:
        assert self._server is not None
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._stop_event.set)
        async with self._server:
            stop_task = asyncio.create_task(self._stop_event.wait())
            serve_task = asyncio.create_task(self._server.serve_forever())
            done, pending = await asyncio.wait(
                {stop_task, serve_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in done:
                exc = task.exception()
                if exc is not None and not isinstance(exc, asyncio.CancelledError):
                    logger.warning("daemon task ended with %s", exc)
        await self.stop()

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        for task in list(self._connections):
            task.cancel()
        for task in list(self._connections):
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        if self._node is not None:
            await self._node.shutdown()
            self._node = None
        for path in (self._socket_path, self._pid_path, self._meta_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer_queue: asyncio.Queue[dict] = asyncio.Queue()
        assert self._node is not None
        node = self._node
        node.add_broadcast_queue(peer_queue)
        task = asyncio.current_task()
        if task is not None:
            self._connections.add(task)

        async def forward_to_peer() -> None:
            while True:
                msg = await peer_queue.get()
                try:
                    await write_message(writer, msg)
                except (ConnectionResetError, BrokenPipeError):
                    break

        forward_task = asyncio.create_task(forward_to_peer())

        try:
            while True:
                try:
                    msg = await read_message(reader)
                except LspProtocolError as exc:
                    logger.warning("peer protocol error: %s", exc)
                    break
                if msg is None:
                    break

                method = msg.get("method")
                msg_id = msg.get("id")

                # Intercept per-peer lifecycle messages so they don't kill
                # the shared Node child. Each peer's `shutdown`/`exit` only
                # means "I'm done with this connection".
                if method == "shutdown":
                    if msg_id is not None:
                        await peer_queue.put({"jsonrpc": "2.0", "id": msg_id, "result": None})
                    continue
                if method == "exit":
                    break

                if msg_id is not None:
                    node.claim_response_for(msg_id, peer_queue)
                await node.send(msg)
        finally:
            forward_task.cancel()
            try:
                await forward_task
            except (asyncio.CancelledError, Exception):
                pass
            node.remove_broadcast_queue(peer_queue)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            if task is not None:
                self._connections.discard(task)


def _resolve_server_js() -> Path:
    env = os.environ.get("HED_LSP_SERVER_JS")
    if env:
        return Path(env)
    candidates = [
        Path("/app/hed-lsp/server/out/server.js"),
        Path.home() / "Documents/git/hed/hed-lsp/server/out/server.js",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(
        "Could not locate hed-lsp server.js. Set HED_LSP_SERVER_JS or install "
        "the hed-lsp Node server."
    )


def main(argv: list[str] | None = None) -> NoReturn:
    parser = argparse.ArgumentParser(description="HEDit LSP daemon (foreground)")
    parser.add_argument(
        "--server-js",
        type=Path,
        default=None,
        help="Path to hed-lsp server.js (overrides HED_LSP_SERVER_JS).",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=None,
        help="Directory for socket + PID file (default: per-user runtime dir).",
    )
    parser.add_argument(
        "--node",
        default=os.environ.get("HEDIT_NODE", "node"),
        help="Path to node binary (default: 'node' on PATH).",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("HEDIT_LSP_LOG_LEVEL", "INFO"),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s"
    )
    server_js = args.server_js or _resolve_server_js()
    runtime_dir = args.runtime_dir or default_runtime_dir()

    daemon = _Daemon(server_js=server_js, runtime_dir=runtime_dir, node_path=args.node)

    async def _run() -> None:
        await daemon.start()
        await daemon.serve_until_signal()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    sys.exit(0)


if __name__ == "__main__":
    main()
