"""Async LSP/JSON-RPC client for the hed-lsp server.

Owns either a spawned Node child (`spawn_stdio`) or a Unix-socket
connection to a running daemon (`connect_unix`). Both transports speak
the same LSP wire protocol; the client demultiplexes responses by
JSON-RPC id so multiple `suggest()` calls can be in flight at once.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.lsp.protocol import LspProtocolError, read_message, write_message

logger = logging.getLogger(__name__)


@dataclass
class HedSuggestion:
    """One tag suggestion returned by the LSP server."""

    tag: str
    score: float | None = None
    description: str | None = None


@dataclass
class HedSuggestResult:
    """Result of a `suggest()` call.

    `success` is False only when the request itself failed (transport
    error, server error response, malformed payload). An empty
    `suggestions` list with `success=True` means the server answered
    but had no matches.
    """

    success: bool
    suggestions: list[HedSuggestion] = field(default_factory=list)
    error: str | None = None
    raw: dict[str, list[str]] = field(default_factory=dict)


class HedLspClient:
    """Persistent connection to a hed-lsp server."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        schema_version: str = "8.4.0",
        max_results: int = 10,
        process: asyncio.subprocess.Process | None = None,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._schema_version = schema_version
        self._max_results = max_results
        self._process = process
        self._next_id = 1
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()
        self._shutdown_started = False

    @classmethod
    async def spawn_stdio(
        cls,
        server_js: Path | str,
        *,
        node_path: str = "node",
        schema_version: str = "8.4.0",
        max_results: int = 10,
        startup_timeout: float = 30.0,
    ) -> HedLspClient:
        """Spawn `node <server_js> --stdio` and run the initialize handshake."""
        server_js_path = Path(server_js)
        if shutil.which(node_path) is None:
            raise RuntimeError(f"node binary not found on PATH (looked for {node_path!r})")
        if not server_js_path.exists():
            raise RuntimeError(f"LSP server entrypoint not found at {server_js_path}")

        process = await asyncio.create_subprocess_exec(
            node_path,
            str(server_js_path),
            "--stdio",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("Failed to attach to subprocess stdio")

        client = cls(
            reader=process.stdout,
            writer=process.stdin,
            schema_version=schema_version,
            max_results=max_results,
            process=process,
        )
        client._stderr_task = asyncio.create_task(_drain(process.stderr, "hed-lsp"))
        try:
            await client._start(startup_timeout)
        except BaseException:
            await client.shutdown()
            raise
        return client

    @classmethod
    async def connect_unix(
        cls,
        socket_path: Path | str,
        *,
        schema_version: str = "8.4.0",
        max_results: int = 10,
        startup_timeout: float = 30.0,
    ) -> HedLspClient:
        """Connect to a running hedit-lspd daemon via Unix socket."""
        reader, writer = await asyncio.open_unix_connection(str(socket_path))
        client = cls(
            reader=reader,
            writer=writer,
            schema_version=schema_version,
            max_results=max_results,
        )
        try:
            await client._start(startup_timeout)
        except BaseException:
            await client.shutdown()
            raise
        return client

    async def _start(self, startup_timeout: float) -> None:
        self._reader_task = asyncio.create_task(self._read_loop())
        await asyncio.wait_for(self._initialize(), timeout=startup_timeout)

    async def _initialize(self) -> None:
        result = await self._request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": None,
                "capabilities": {},
            },
        )
        caps = list((result or {}).get("capabilities", {}).keys())
        logger.debug("LSP initialize OK; server capabilities: %s", caps)
        await self._notify("initialized", {})

    async def _read_loop(self) -> None:
        transport = "stdio" if self._process is not None else "unix-socket"
        try:
            while True:
                try:
                    msg = await read_message(self._reader)
                except LspProtocolError as exc:
                    logger.warning("LSP protocol error on %s transport: %s", transport, exc)
                    break
                if msg is None:
                    break
                self._dispatch(msg)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("LSP reader loop crashed on %s transport", transport)
        finally:
            # Once the reader exits we can never resolve future responses;
            # mark the client as shutting down so subsequent _request calls
            # fail fast with a clear error instead of hanging on a future
            # that will never complete.
            self._shutdown_started = True
            self._fail_pending(RuntimeError("LSP connection closed"))

    def _dispatch(self, msg: dict[str, Any]) -> None:
        msg_id = msg.get("id")
        if msg_id is None:
            method = msg.get("method")
            if method is not None:
                logger.debug("LSP server notification (unhandled): method=%s", method)
            return
        fut = self._pending.pop(msg_id, None)
        if fut is None:
            logger.warning(
                "LSP response for unknown id=%r; client may have already given up", msg_id
            )
            return
        if fut.done():
            logger.warning(
                "LSP response for id=%r arrived after future was already resolved", msg_id
            )
            return
        if "error" in msg:
            err = msg["error"]
            fut.set_exception(RuntimeError(f"LSP server error: {err}"))
        else:
            fut.set_result(msg.get("result"))

    def _fail_pending(self, exc: Exception) -> None:
        pending = self._pending
        self._pending = {}
        for fut in pending.values():
            if not fut.done():
                fut.set_exception(exc)

    async def _request(self, method: str, params: Any) -> Any:
        if self._shutdown_started:
            raise RuntimeError("LSP client is shutting down")
        msg_id = self._next_id
        self._next_id += 1
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.create_future()
        self._pending[msg_id] = fut
        msg = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
        async with self._write_lock:
            try:
                await write_message(self._writer, msg)
            except Exception as exc:
                self._pending.pop(msg_id, None)
                raise RuntimeError(f"Failed to send LSP request {method}: {exc}") from exc
        return await fut

    async def _notify(self, method: str, params: Any) -> None:
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        async with self._write_lock:
            await write_message(self._writer, msg)

    async def suggest(
        self,
        *queries: str,
        use_semantic: bool | None = None,
    ) -> HedSuggestResult:
        """Issue one batched `hed/suggest` request and return all suggestions.

        All queries share the same schema version and `max_results`
        configured on this client. The returned `raw` mapping preserves
        the per-query grouping the server emits; the flat `suggestions`
        list deduplicates within the result for callers that don't
        care which query produced which tag.
        """
        if not queries:
            return HedSuggestResult(success=False, error="No queries provided")

        params: dict[str, Any] = {
            "queries": list(queries),
            "schema": self._schema_version,
            "top": self._max_results,
        }
        if use_semantic is not None:
            params["semantic"] = use_semantic

        try:
            raw = await self._request("hed/suggest", params)
        except Exception as exc:
            return HedSuggestResult(success=False, error=str(exc))

        if not isinstance(raw, dict):
            return HedSuggestResult(
                success=False,
                error=f"Unexpected hed/suggest response shape: {type(raw).__name__}",
            )

        raw_clean: dict[str, list[str]] = {}
        suggestions: list[HedSuggestion] = []
        seen: set[str] = set()
        for query, tags in raw.items():
            if not isinstance(tags, list):
                continue
            cleaned = [t for t in tags if isinstance(t, str) and t]
            raw_clean[query] = cleaned
            for tag in cleaned:
                if tag not in seen:
                    seen.add(tag)
                    suggestions.append(HedSuggestion(tag=tag))
        return HedSuggestResult(success=True, suggestions=suggestions, raw=raw_clean)

    async def shutdown(self) -> None:
        """Send LSP shutdown+exit, close streams, reap the child."""
        if self._shutdown_started:
            return
        self._shutdown_started = True

        try:
            await asyncio.wait_for(self._request_unshutdownable("shutdown", None), timeout=5.0)
        except Exception as exc:
            logger.debug("LSP shutdown request failed: %s", exc)

        try:
            await self._notify("exit", None)
        except Exception as exc:
            logger.debug("LSP exit notification failed: %s", exc)

        # Cancel the reader first so it doesn't see a half-closed writer
        # and report a spurious "LSP connection closed" before shutdown
        # finishes; the canceller waits on the reader task to drain.
        for task in (self._reader_task, self._stderr_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        try:
            self._writer.close()
            await self._writer.wait_closed()
        except Exception as exc:
            logger.debug("LSP writer close error during shutdown: %s", exc)

        if self._process is not None and self._process.returncode is None:
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except TimeoutError:
                self._process.kill()
                await self._process.wait()

        self._fail_pending(RuntimeError("LSP client shut down"))

    async def _request_unshutdownable(self, method: str, params: Any) -> Any:
        """Send a request even after `_shutdown_started` was set.

        Used for the `shutdown` request itself, since `_request` rejects
        calls once the shutdown flag is set. Mirrors `_request`'s
        write-error handling so a failed write doesn't leave a stranded
        future in `_pending`.
        """
        msg_id = self._next_id
        self._next_id += 1
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.create_future()
        self._pending[msg_id] = fut
        msg = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
        async with self._write_lock:
            try:
                await write_message(self._writer, msg)
            except Exception as exc:
                self._pending.pop(msg_id, None)
                raise RuntimeError(f"Failed to send LSP request {method}: {exc}") from exc
        return await fut


async def _drain(stream: asyncio.StreamReader | None, tag: str) -> None:
    """Forward subprocess stderr lines to the Python logger."""
    if stream is None:
        return
    while True:
        line = await stream.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace").rstrip()
        if text:
            logger.debug("[%s stderr] %s", tag, text)
