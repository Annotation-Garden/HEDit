"""Persistent client for the hed-lsp Node server.

The hed-lsp server is a long-running LSP process that loads the HED
schema and embedding model once at boot. This module speaks the LSP
wire protocol (JSON-RPC over Content-Length framing) so the Python
backend can hold one connection for the lifetime of the process and
issue batched `hed/suggest` requests in milliseconds, instead of
spawning the `hed-suggest` CLI per query (~6 s per spawn).
"""

from src.lsp.client import HedLspClient, HedSuggestion, HedSuggestResult

__all__ = ["HedLspClient", "HedSuggestion", "HedSuggestResult"]
