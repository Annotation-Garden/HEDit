"""JSON-based HED schema loading and vocabulary extraction.

This module loads HED schemas from JSON format and extracts vocabulary
with support for tag extension rules.
"""

import json
from pathlib import Path
from typing import Any


class HedJsonSchemaLoader:
    """Loads HED schemas from JSON format.

    Provides vocabulary extraction with proper short-form tags
    and information about extensionAllowed tags.
    """

    def __init__(self, schema_path: Path | str) -> None:
        """Initialize with path to JSON schema file.

        Args:
            schema_path: Path to HED JSON schema file
        """
        self.schema_path = Path(schema_path)
        self.schema_data: dict[str, Any] = {}
        self.tags: dict[str, Any] = {}
        self.vocabulary: list[str] = []
        self.extendable_tags: dict[str, str] = {}  # short_form -> description
        self._load_schema()

    def _load_schema(self) -> None:
        """Load and parse the JSON schema."""
        with open(self.schema_path) as f:
            self.schema_data = json.load(f)

        self.tags = self.schema_data.get("tags", {})
        self._extract_vocabulary()

    def _extract_vocabulary(self) -> None:
        """Extract short-form vocabulary and extensionAllowed tags."""
        self.vocabulary = []
        self.extendable_tags = {}

        def traverse_tags(tags_dict: dict[str, Any]) -> None:
            """Recursively traverse tag tree."""
            for tag_name, tag_data in tags_dict.items():
                if isinstance(tag_data, dict):
                    short_form = tag_data.get("short_form", tag_name)

                    # Add to vocabulary (prefer short-form)
                    if short_form not in self.vocabulary:
                        self.vocabulary.append(short_form)

                    # Check if extensionAllowed
                    attrs = tag_data.get("attributes", {})
                    if attrs.get("extensionAllowed"):
                        desc = tag_data.get("description", "")
                        self.extendable_tags[short_form] = desc

        traverse_tags(self.tags)
        self.vocabulary.sort()

    def get_vocabulary(self) -> list[str]:
        """Get list of all short-form tags.

        Returns:
            Sorted list of short-form tag names
        """
        return self.vocabulary.copy()

    def get_extendable_tags(self) -> dict[str, str]:
        """Get dictionary of tags that allow extension.

        Returns:
            Dict mapping short_form tag to description
        """
        return self.extendable_tags.copy()

    def is_extendable(self, tag: str) -> bool:
        """Check if a tag allows extension.

        Args:
            tag: Tag name (short or long form)

        Returns:
            True if tag allows extension with /
        """
        # Check both the tag itself and its path components
        return tag in self.extendable_tags

    def get_schema_version(self) -> str:
        """Get schema version.

        Returns:
            Schema version string
        """
        return self.schema_data.get("version", "unknown")

    def find_closest_match(self, invalid_tag: str) -> list[str]:
        """Find closest vocabulary matches for an invalid tag.

        Uses simple string matching to suggest alternatives.

        Args:
            invalid_tag: Tag that's not in vocabulary

        Returns:
            List of suggested tags (up to 5)
        """
        invalid_lower = invalid_tag.lower()
        suggestions = []

        # Exact case-insensitive match
        for tag in self.vocabulary:
            if tag.lower() == invalid_lower:
                return [tag]

        # Contains match
        for tag in self.vocabulary:
            if invalid_lower in tag.lower() or tag.lower() in invalid_lower:
                suggestions.append(tag)
                if len(suggestions) >= 5:
                    break

        return suggestions[:5]


def load_latest_schema(schema_dir: Path | str) -> HedJsonSchemaLoader:
    """Load the latest HED standard schema.

    Args:
        schema_dir: Directory containing JSON schemas (required)

    Returns:
        Loaded HedJsonSchemaLoader

    Raises:
        ValueError: If schema_dir is None or invalid
    """
    if schema_dir is None:
        raise ValueError(
            "schema_dir is required for load_latest_schema. "
            "Use HedSchemaLoader for dynamic schema fetching from GitHub."
        )

    schema_dir = Path(schema_dir)
    latest_schema = schema_dir / "HEDLatest.json"
    return HedJsonSchemaLoader(latest_schema)
