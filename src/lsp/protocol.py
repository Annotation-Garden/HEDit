"""LSP wire-protocol framing.

JSON-RPC messages on an LSP connection are prefixed with an HTTP-like
header block terminated by a blank line; the only required header is
`Content-Length`, giving the byte length of the UTF-8 JSON body that
follows. This module reads and writes those frames over any asyncio
stream pair (stdio of a subprocess, a Unix socket, a TCP socket).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any


class LspProtocolError(Exception):
    """Raised when a malformed LSP frame is received."""


async def read_message(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    """Read one framed JSON-RPC message.

    Returns the parsed message dict, or None on clean EOF (peer closed
    the stream before sending any bytes of the next frame).
    """
    headers: dict[str, str] = {}
    first_line = True
    while True:
        line = await reader.readline()
        if not line:
            if first_line and not headers:
                return None
            raise LspProtocolError("Unexpected EOF inside LSP header block")
        first_line = False
        stripped = line.rstrip(b"\r\n")
        if not stripped:
            break
        try:
            key, _, value = stripped.decode("ascii").partition(":")
        except UnicodeDecodeError as exc:
            raise LspProtocolError(f"Non-ASCII byte in LSP header: {stripped!r}") from exc
        headers[key.strip().lower()] = value.strip()

    length_str = headers.get("content-length")
    if length_str is None:
        raise LspProtocolError(f"LSP message missing Content-Length header: {headers!r}")
    try:
        length = int(length_str)
    except ValueError as exc:
        raise LspProtocolError(f"Invalid Content-Length value: {length_str!r}") from exc

    body = await reader.readexactly(length)
    try:
        return json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise LspProtocolError(f"Could not decode LSP message body: {exc}") from exc


def encode_message(msg: dict[str, Any]) -> bytes:
    """Encode a JSON-RPC message with proper LSP framing."""
    body = json.dumps(msg, separators=(",", ":")).encode("utf-8")
    return f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body


async def write_message(writer: asyncio.StreamWriter, msg: dict[str, Any]) -> None:
    """Send one framed JSON-RPC message and flush."""
    writer.write(encode_message(msg))
    await writer.drain()
