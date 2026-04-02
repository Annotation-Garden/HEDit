"""Loader for official HED documentation bundled with the package.

Reads processed HED documentation from src/data/hed-docs/ and provides
it for inclusion in the annotation agent's system prompt. The docs are
fetched and processed offline by scripts/fetch_hed_docs.py, then
bundled with the package via setuptools package-data.

No runtime fetching occurs here; the system prompt stays deterministic.
"""

import logging
from importlib.resources import files
from pathlib import Path

logger = logging.getLogger(__name__)

# Module-level cache
_cached_docs: dict[str, str] | None = None

# Maximum characters per document (safety limit)
MAX_DOC_CHARS = 50_000

# Document identifiers mapped to filenames
_DOC_FILES = {
    "annotation_semantics": "HedAnnotationSemantics.md",
    "terminology": "02_Terminology.md",
}


def _get_docs_dir() -> Path:
    """Get path to bundled HED docs directory.

    Uses importlib.resources to locate the 'src.data' package (matching
    the setuptools find-packages configuration), falling back to a
    __file__-relative path for editable installs or development.
    """
    try:
        docs_dir = files("src.data") / "hed-docs"
        # Verify it's a real directory path
        if hasattr(docs_dir, "is_dir") and docs_dir.is_dir():
            return Path(str(docs_dir))
        # For Traversable objects, try resolving to a path
        resolved = Path(str(docs_dir))
        if resolved.is_dir():
            return resolved
    except (TypeError, ModuleNotFoundError) as e:
        logger.debug(
            "importlib.resources lookup for 'src.data' failed (%s: %s); "
            "falling back to __file__-relative path",
            type(e).__name__,
            e,
        )

    # Fallback: relative to this file
    fallback = Path(__file__).parent.parent / "data" / "hed-docs"
    if fallback.is_dir():
        return fallback

    return fallback  # Return anyway; caller handles missing files


def load_hed_docs(*, docs_dir: Path | None = None) -> dict[str, str]:
    """Load official HED documentation content.

    Returns a dict mapping doc identifiers to processed markdown content.
    Results are cached after the first successful load (non-empty).
    Empty results are not cached, allowing retry in long-running processes.

    When docs_dir is provided, bypasses both the cache and default path
    resolution, loading directly from the given directory. This is used
    by tests to point at real temporary directories.

    Args:
        docs_dir: Optional override for the docs directory path.
                  Bypasses cache when provided.

    Returns:
        Dict with keys "annotation_semantics" and/or "terminology".
        Partial dict if only some docs are available, empty if none found.
    """
    global _cached_docs

    # When an explicit directory is provided, skip cache entirely
    if docs_dir is not None:
        return _load_from_dir(docs_dir)

    if _cached_docs is not None:
        return _cached_docs

    result = _load_from_dir(_get_docs_dir())

    if not result:
        logger.warning(
            "No bundled HED docs found. System prompt will use "
            "HEDit-specific sections only. Will retry on next call."
        )
        # Do NOT cache empty results; allow retry in long-running processes
        return result

    _cached_docs = result
    return result


def _load_from_dir(docs_dir: Path) -> dict[str, str]:
    """Load docs from a specific directory.

    Args:
        docs_dir: Directory containing the processed markdown files.

    Returns:
        Dict mapping doc identifiers to content strings.
    """
    result: dict[str, str] = {}

    for doc_id, filename in _DOC_FILES.items():
        doc_path = docs_dir / filename
        try:
            content = doc_path.read_text(encoding="utf-8")
            truncation_msg = "\n\n... [truncated for length]"
            if len(content) > MAX_DOC_CHARS:
                content = content[: MAX_DOC_CHARS - len(truncation_msg)] + truncation_msg
                logger.info("Truncated %s to %d chars", filename, MAX_DOC_CHARS)
            result[doc_id] = content
        except FileNotFoundError:
            logger.warning(
                "Bundled HED doc not found: %s. Run 'python scripts/fetch_hed_docs.py' to fetch.",
                doc_path,
            )
        except OSError as e:
            logger.warning("Error reading %s: %s", doc_path, e)

    return result


def clear_cache() -> None:
    """Clear the cached docs. Useful for testing."""
    global _cached_docs
    _cached_docs = None
