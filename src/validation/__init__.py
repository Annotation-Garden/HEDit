"""HED validation integration.

This module provides HED validation and tag suggestion capabilities:

- HedPythonValidator: Validates HED strings using hed-python library
- HedJavaScriptValidator: Validates HED strings using hed-javascript (legacy)
- HedLspClient: Persistent async client for the hed-lsp Node server
- get_validator: Factory function to get the appropriate validator
- is_hed_lsp_available: Whether node + hed-lsp server.js are reachable

Note: Imports are lazy to avoid requiring hed-python for hed-lsp functionality.
"""

from typing import TYPE_CHECKING

# HED-LSP imports (no heavy dependencies)
from src.validation.hed_lsp import (
    HedLspClient,
    HedSuggestion,
    HedSuggestResult,
    find_lsp_server_js,
    is_hed_lsp_available,
)

# Type hints only - actual imports happen at runtime when used via __getattr__
if TYPE_CHECKING:
    from src.validation.hed_validator import (
        HedJavaScriptValidator,
        HedPythonValidator,
        ValidationIssue,
        ValidationResult,
        get_validator,
        is_js_validator_available,
    )

__all__ = [
    # Validators (requires hed-python)
    "HedPythonValidator",
    "HedJavaScriptValidator",
    "get_validator",
    "is_js_validator_available",
    "ValidationIssue",
    "ValidationResult",
    # HED-LSP (lightweight)
    "HedLspClient",
    "HedSuggestion",
    "HedSuggestResult",
    "find_lsp_server_js",
    "is_hed_lsp_available",
]


def __getattr__(name: str) -> object:
    """Lazy import for hed_validator components that require hed-python."""
    if name in (
        "HedPythonValidator",
        "HedJavaScriptValidator",
        "get_validator",
        "is_js_validator_available",
        "ValidationIssue",
        "ValidationResult",
    ):
        from src.validation import hed_validator

        return getattr(hed_validator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
