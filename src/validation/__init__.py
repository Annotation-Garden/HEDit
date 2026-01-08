"""HED validation integration.

This module provides HED validation and tag suggestion capabilities:

- HedPythonValidator: Validates HED strings using hedtools Python library
- HedJavaScriptValidator: Validates HED strings using hed-javascript (legacy)
- HedLspClient: Suggests HED tags using hed-lsp CLI
- get_validator: Factory function to get the appropriate validator
- is_hed_lsp_available: Check if hed-lsp CLI is installed

Note: Imports are lazy to avoid requiring hedtools for hed-lsp functionality.
"""

from typing import TYPE_CHECKING

# HED-LSP imports (no heavy dependencies)
from src.validation.hed_lsp import (
    HedLspClient,
    HedSuggestion,
    HedSuggestResult,
    get_hed_suggestions,
    is_hed_lsp_available,
    suggest_tags_for_keywords,
)

# Type hints only - actual imports happen at runtime when used
if TYPE_CHECKING:
    from src.validation.hed_validator import (
        HedJavaScriptValidator,
        HedPythonValidator,
        ValidationIssue,
        ValidationResult,
    )

__all__ = [
    # Validators (requires hedtools)
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
    "get_hed_suggestions",
    "is_hed_lsp_available",
    "suggest_tags_for_keywords",
]


def __getattr__(name: str):
    """Lazy import for hed_validator components that require hedtools."""
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
