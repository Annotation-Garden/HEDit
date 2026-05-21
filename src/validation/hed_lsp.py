"""Legacy import shim for the hed-lsp Python integration.

Historically this module wrapped the `hed-suggest` CLI as a synchronous
subprocess. That path was retired in favor of `src.lsp.HedLspClient`,
which holds one persistent JSON-RPC connection to the hed-lsp server
and answers batched queries in milliseconds. This file now re-exports
the small set of helpers we still use and locates a server.js install
for downstream callers (FastAPI lifespan, CLI executors).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from src.lsp import HedLspClient, HedSuggestion, HedSuggestResult

__all__ = [
    "HedLspClient",
    "HedSuggestion",
    "HedSuggestResult",
    "find_lsp_server_js",
    "is_hed_lsp_available",
]


def _server_js_candidates() -> list[Path]:
    env = os.environ.get("HED_LSP_SERVER_JS")
    if env:
        return [Path(env)]
    return [
        Path("/app/hed-lsp/server/out/server.js"),
        Path.home() / "Documents/git/hed/hed-lsp/server/out/server.js",
        Path.home() / "git/hed/hed-lsp/server/out/server.js",
    ]


def find_lsp_server_js() -> Path | None:
    """Return the first existing `hed-lsp` server.js path, or None."""
    for candidate in _server_js_candidates():
        if candidate.exists():
            return candidate
    return None


def is_hed_lsp_available() -> bool:
    """Whether a `node` interpreter and the hed-lsp server.js are reachable.

    Cheap check used by callers (CLI, tests) that want to decide whether
    to spawn a `HedLspClient` without actually attempting to spawn one.
    """
    return shutil.which("node") is not None and find_lsp_server_js() is not None
