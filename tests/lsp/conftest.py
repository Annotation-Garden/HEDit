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


def _supports_hed_suggest(server_js: Path) -> bool:
    """Whether a built server bundle handles the hed/suggest request.

    hed-lsp added hed/suggest in 0.4.0. An older bundle answers
    "Unhandled method hed/suggest" (JSON-RPC -32601), which would surface as
    a wall of assertion failures rather than the real problem: a stale local
    build. The check reads the bundle because that is the artifact the tests
    actually run.
    """
    try:
        return "hed/suggest" in server_js.read_text(errors="ignore")
    except OSError:
        return False


@pytest.fixture(scope="session")
def hed_lsp_server_js() -> Path:
    """Locate a usable local hed-lsp build, skipping if missing or too old."""
    if shutil.which("node") is None:
        pytest.skip("node binary not found on PATH; skipping LSP tests")
    for candidate in _candidate_server_js_paths():
        if not candidate.exists():
            continue
        if not _supports_hed_suggest(candidate):
            pytest.skip(
                f"hed-lsp build at {candidate} predates the hed/suggest request "
                "(added in hed-lsp 0.4.0). Rebuild it from a current checkout "
                "(bun install && node esbuild.config.mjs), or point "
                "HED_LSP_SERVER_JS at a newer build."
            )
        return candidate
    pytest.skip("hed-lsp server.js not found. Build hed-lsp locally or set HED_LSP_SERVER_JS.")
