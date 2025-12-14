"""HED schema loading and caching utilities.

This module handles loading HED schemas from various sources and formats,
with session-based caching to improve performance.
"""

from pathlib import Path

from hed import load_schema, load_schema_version
from hed.schema import HedSchema


class HedSchemaLoader:
    """Loads and caches HED schemas for use in annotation and validation.

    The loader supports loading from:
    - Version strings (e.g., "8.3.0")
    - Local XML files
    - Schema names with library support (e.g., "score_2.0.0")

    Schemas are cached per session to avoid repeated loading.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        """Initialize the schema loader.

        Args:
            cache_dir: Optional directory for caching downloaded schemas.
                      If None, uses default HEDTools cache location.
        """
        self._cache: dict[str, HedSchema] = {}
        self._cache_dir = cache_dir

    def load_schema(
        self,
        schema_spec: str = "8.3.0",
        library_name: str | None = None,
    ) -> HedSchema:
        """Load a HED schema from version string or file path.

        Args:
            schema_spec: Schema version (e.g., "8.3.0") or path to XML file
            library_name: Optional library name (e.g., "score")

        Returns:
            Loaded HedSchema object

        Raises:
            ValueError: If schema cannot be loaded
        """
        cache_key = f"{library_name}:{schema_spec}" if library_name else schema_spec

        # Return cached schema if available
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Load schema based on input type
        try:
            if Path(schema_spec).exists():
                # Load from file path
                schema = load_schema(schema_spec)
            else:
                # Load from version string
                # Note: library_name is not directly supported by load_schema_version
                # For library schemas, use format "library:version" (e.g., "score:2.0.0")
                schema = load_schema_version(xml_version=schema_spec)

            # Cache the loaded schema
            self._cache[cache_key] = schema
            return schema

        except Exception as e:
            raise ValueError(f"Failed to load HED schema '{schema_spec}': {e}") from e

    def get_schema_vocabulary(self, schema: HedSchema) -> list[str]:
        """Extract all tag names from a schema as a vocabulary list.

        Args:
            schema: HedSchema object

        Returns:
            List of all tag names (short form) in the schema
        """
        tags = []
        for _name, entry in schema.tags.items():
            # Get short form tag name
            tags.append(entry.short_tag_name)
        return sorted(tags)

    def clear_cache(self) -> None:
        """Clear all cached schemas."""
        self._cache.clear()


# Global schema loader instance for shared use
_default_loader: HedSchemaLoader | None = None


def get_schema_loader() -> HedSchemaLoader:
    """Get the default global schema loader instance.

    Returns:
        Global HedSchemaLoader instance
    """
    global _default_loader
    if _default_loader is None:
        _default_loader = HedSchemaLoader()
    return _default_loader
