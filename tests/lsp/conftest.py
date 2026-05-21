"""Shared fixtures for LSP integration tests.

Locates a local hed-lsp build via env var or a small set of candidate
paths, and skips tests when node + server.js + the new hed/suggest
endpoint aren't all available.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest


def _candidate_server_js_paths() -> list[Path]:
    env = os.environ.get("HED_LSP_SERVER_JS")
    if env:
        return [Path(env)]
    return [
        Path("/app/hed-lsp/server/out/server.js"),
        Path.home() / "Documents/git/hed/hed-lsp/server/out/server.js",
        Path.home() / "git/hed/hed-lsp/server/out/server.js",
    ]


@pytest.fixture(scope="session")
def hed_lsp_server_js() -> Path:
    """Locate the local hed-lsp build, skipping if missing."""
    if shutil.which("node") is None:
        pytest.skip("node binary not found on PATH; skipping LSP tests")
    for candidate in _candidate_server_js_paths():
        if candidate.exists():
            return candidate
    pytest.skip("hed-lsp server.js not found. Build hed-lsp locally or set HED_LSP_SERVER_JS.")
    raise AssertionError("unreachable: pytest.skip never returns")
