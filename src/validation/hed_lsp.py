"""HED-LSP CLI integration for HED tag suggestions and validation.

This module provides integration with the hed-lsp CLI tool for:
- Suggesting HED tags from natural language descriptions
- Schema-aware tag completion
- Semantic search for relevant tags

Requires hed-lsp to be installed and available in PATH.
Installation: https://github.com/hed-standard/hed-lsp
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


def is_hed_lsp_available() -> bool:
    """Check if hed-suggest CLI is available in PATH.

    Returns:
        True if hed-suggest command is available.
    """
    return shutil.which("hed-suggest") is not None


def get_default_schema_version() -> str:
    """Get default HED schema version from environment.

    Returns:
        Schema version string (default: "8.4.0")
    """
    return os.environ.get("HED_SCHEMA_VERSION", "8.4.0")


def get_default_use_semantic() -> bool:
    """Get default semantic search setting from environment.

    Returns:
        True if semantic search should be enabled by default
    """
    return os.environ.get("HED_LSP_USE_SEMANTIC", "false").lower() == "true"


def get_default_max_results() -> int:
    """Get default max results setting from environment.

    Returns:
        Maximum number of results to return (default: 10)
    """
    value = os.environ.get("HED_LSP_MAX_RESULTS", "10")
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid HED_LSP_MAX_RESULTS value '{value}', using default of 10")
        return 10


@dataclass
class HedSuggestion:
    """A suggested HED tag from the hed-lsp CLI.

    Attributes:
        tag: The suggested HED tag path
        score: Relevance score (higher is better)
        description: Tag description from schema
    """

    tag: str
    score: float | None = None
    description: str | None = None


@dataclass
class HedSuggestResult:
    """Result from hed-suggest CLI command.

    Attributes:
        success: Whether the command succeeded
        suggestions: List of suggested HED tags
        error: Error message if command failed
    """

    success: bool
    suggestions: list[HedSuggestion]
    error: str | None = None


class HedLspClient:
    """Client for interacting with hed-lsp CLI tools.

    This class provides a Python interface to the hed-lsp command-line tools,
    enabling HED tag suggestions from natural language descriptions.

    Example:
        >>> client = HedLspClient(schema_version="8.3.0")
        >>> result = client.suggest("button press")
        >>> for suggestion in result.suggestions:
        ...     print(f"{suggestion.tag}: {suggestion.score}")
    """

    def __init__(
        self,
        schema_version: str | None = None,
        use_semantic: bool | None = None,
        max_results: int | None = None,
    ) -> None:
        """Initialize the HED-LSP client.

        Args:
            schema_version: HED schema version to use. Defaults to HED_SCHEMA_VERSION env var.
            use_semantic: Enable semantic search. Defaults to HED_LSP_USE_SEMANTIC env var.
            max_results: Maximum suggestions. Defaults to HED_LSP_MAX_RESULTS env var.

        Raises:
            RuntimeError: If hed-suggest CLI is not available
        """
        if not is_hed_lsp_available():
            raise RuntimeError(
                "hed-suggest CLI not found in PATH. "
                "Install hed-lsp from https://github.com/hed-standard/hed-lsp "
                "and run 'npm link' in the server directory."
            )

        self.schema_version = schema_version or get_default_schema_version()
        self.use_semantic = use_semantic if use_semantic is not None else get_default_use_semantic()
        self.max_results = max_results if max_results is not None else get_default_max_results()

    def suggest(self, *queries: str) -> HedSuggestResult:
        """Suggest HED tags for one or more natural language descriptions.

        Args:
            *queries: One or more natural language descriptions to convert to HED tags

        Returns:
            HedSuggestResult with suggested tags or error information
        """
        if not queries:
            return HedSuggestResult(
                success=False,
                suggestions=[],
                error="No queries provided",
            )

        # Build command
        cmd = [
            "hed-suggest",
            "--json",
            "--schema",
            self.schema_version,
            "--top",
            str(self.max_results),
        ]

        if self.use_semantic:
            cmd.append("--semantic")

        # Add all query terms
        cmd.extend(queries)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return HedSuggestResult(
                    success=False,
                    suggestions=[],
                    error=result.stderr or f"Command failed with exit code {result.returncode}",
                )

            # Parse JSON output
            output = json.loads(result.stdout)
            suggestions = []

            # Handle different output formats
            if isinstance(output, list):
                # List of suggestions
                for item in output:
                    if isinstance(item, str):
                        suggestions.append(HedSuggestion(tag=item))
                    elif isinstance(item, dict):
                        tag = item.get("tag") or item.get("name") or ""
                        suggestions.append(
                            HedSuggestion(
                                tag=tag,
                                score=item.get("score"),
                                description=item.get("description"),
                            )
                        )
            elif isinstance(output, dict):
                # Handle hed-suggest output format: {"query": ["tag1", "tag2", ...]}
                # First check for explicit suggestions/results keys
                items = output.get("suggestions") or output.get("results")
                if items is not None:
                    for item in items:
                        if isinstance(item, str):
                            suggestions.append(HedSuggestion(tag=item))
                        elif isinstance(item, dict):
                            tag = item.get("tag") or item.get("name") or ""
                            suggestions.append(
                                HedSuggestion(
                                    tag=tag,
                                    score=item.get("score"),
                                    description=item.get("description"),
                                )
                            )
                else:
                    # Handle format where keys are query terms
                    # e.g., {"button press": ["Button", "Response-button", ...]}
                    for _query_key, tag_list in output.items():
                        if isinstance(tag_list, list):
                            for item in tag_list:
                                if isinstance(item, str):
                                    suggestions.append(HedSuggestion(tag=item))
                                elif isinstance(item, dict):
                                    tag = item.get("tag") or item.get("name") or ""
                                    suggestions.append(
                                        HedSuggestion(
                                            tag=tag,
                                            score=item.get("score"),
                                            description=item.get("description"),
                                        )
                                    )

            return HedSuggestResult(
                success=True,
                suggestions=suggestions,
            )

        except subprocess.TimeoutExpired:
            return HedSuggestResult(
                success=False,
                suggestions=[],
                error="Command timed out after 30 seconds",
            )
        except json.JSONDecodeError as e:
            return HedSuggestResult(
                success=False,
                suggestions=[],
                error=f"Failed to parse JSON output: {e}",
            )
        except Exception as e:
            return HedSuggestResult(
                success=False,
                suggestions=[],
                error=f"Command failed: {e}",
            )

    def suggest_for_description(
        self,
        description: str,
        mode: Literal["basic", "semantic"] | None = None,
    ) -> HedSuggestResult:
        """Suggest HED tags for a natural language event description.

        This is a convenience method that processes a full event description,
        potentially breaking it down into components for better suggestions.

        Args:
            description: Natural language description of an event
            mode: Override the default search mode ("basic" or "semantic")

        Returns:
            HedSuggestResult with suggested tags
        """
        # Temporarily override semantic mode if specified
        original_semantic = self.use_semantic
        if mode == "semantic":
            self.use_semantic = True
        elif mode == "basic":
            self.use_semantic = False

        try:
            # Split description into keywords for better matching
            # Simple approach: use the full description as a query
            return self.suggest(description)
        finally:
            # Restore original setting
            self.use_semantic = original_semantic


def get_hed_suggestions(
    description: str,
    schema_version: str | None = None,
    use_semantic: bool | None = None,
    max_results: int | None = None,
) -> list[str]:
    """Get HED tag suggestions for a natural language description.

    This is a convenience function that creates a client and returns
    just the tag strings.

    Args:
        description: Natural language description to convert to HED tags
        schema_version: HED schema version (defaults to HED_SCHEMA_VERSION env var)
        use_semantic: Enable semantic search (defaults to HED_LSP_USE_SEMANTIC env var)
        max_results: Maximum suggestions (defaults to HED_LSP_MAX_RESULTS env var)

    Returns:
        List of suggested HED tag strings

    Raises:
        RuntimeError: If hed-suggest CLI is not available
    """
    client = HedLspClient(
        schema_version=schema_version,
        use_semantic=use_semantic,
        max_results=max_results,
    )
    result = client.suggest(description)

    if not result.success:
        raise RuntimeError(f"HED suggestion failed: {result.error}")

    return [s.tag for s in result.suggestions]


def suggest_tags_for_keywords(
    keywords: list[str],
    schema_version: str | None = None,
    use_semantic: bool | None = None,
    max_results: int | None = None,
) -> dict[str, list[str]]:
    """Get HED tag suggestions for a list of keywords.

    This is useful for batch processing of keywords extracted from
    event descriptions.

    Args:
        keywords: List of keywords to convert to HED tags
        schema_version: HED schema version (defaults to HED_SCHEMA_VERSION env var)
        use_semantic: Enable semantic search (defaults to HED_LSP_USE_SEMANTIC env var)
        max_results: Maximum suggestions per keyword (defaults to HED_LSP_MAX_RESULTS env var)

    Returns:
        Dictionary mapping keywords to their suggested HED tags

    Raises:
        RuntimeError: If hed-suggest CLI is not available
    """
    if not keywords:
        return {}

    client = HedLspClient(
        schema_version=schema_version,
        use_semantic=use_semantic,
        max_results=max_results,
    )

    results = {}
    failed_keywords = []
    for keyword in keywords:
        result = client.suggest(keyword)
        if result.success:
            results[keyword] = [s.tag for s in result.suggestions]
        else:
            logger.warning(f"hed-lsp suggestion failed for keyword '{keyword}': {result.error}")
            results[keyword] = []
            failed_keywords.append(keyword)

    if failed_keywords:
        logger.warning(f"Failed to get suggestions for {len(failed_keywords)} keywords")

    return results
