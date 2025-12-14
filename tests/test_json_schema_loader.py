"""Tests for JSON schema loader functionality."""

import json

import pytest

from src.utils.json_schema_loader import HedJsonSchemaLoader, load_latest_schema


@pytest.fixture
def sample_schema_json():
    """Create a minimal HED-like JSON schema for testing."""
    return {
        "version": "8.3.0",
        "tags": {
            "Event": {
                "short_form": "Event",
                "description": "Something that happens",
                "attributes": {"extensionAllowed": True},
            },
            "Sensory-event": {
                "short_form": "Sensory-event",
                "description": "A sensory event",
                "attributes": {},
            },
            "Item": {
                "short_form": "Item",
                "description": "An item",
                "attributes": {"extensionAllowed": True},
            },
            "Red": {
                "short_form": "Red",
                "description": "Red color",
                "attributes": {},
            },
            "Circle": {
                "short_form": "Circle",
                "description": "A circle shape",
                "attributes": {},
            },
            "Agent-action": {
                "short_form": "Agent-action",
                "description": "An action by an agent",
                "attributes": {},
            },
        },
    }


@pytest.fixture
def schema_file(sample_schema_json, tmp_path):
    """Create a temporary schema file."""
    schema_path = tmp_path / "HEDTest.json"
    with open(schema_path, "w") as f:
        json.dump(sample_schema_json, f)
    return schema_path


@pytest.fixture
def schema_dir(sample_schema_json, tmp_path):
    """Create a temporary schema directory with HEDLatest.json."""
    schema_path = tmp_path / "HEDLatest.json"
    with open(schema_path, "w") as f:
        json.dump(sample_schema_json, f)
    return tmp_path


class TestHedJsonSchemaLoader:
    """Tests for HedJsonSchemaLoader class."""

    def test_init_loads_schema(self, schema_file):
        """Test that initialization loads the schema correctly."""
        loader = HedJsonSchemaLoader(schema_file)
        assert loader.schema_path == schema_file
        assert loader.schema_data is not None
        assert "tags" in loader.schema_data

    def test_init_with_string_path(self, schema_file):
        """Test initialization with string path."""
        loader = HedJsonSchemaLoader(str(schema_file))
        assert loader.schema_path == schema_file

    def test_get_vocabulary(self, schema_file):
        """Test vocabulary extraction."""
        loader = HedJsonSchemaLoader(schema_file)
        vocab = loader.get_vocabulary()

        assert isinstance(vocab, list)
        assert len(vocab) == 6
        assert "Event" in vocab
        assert "Red" in vocab
        assert "Circle" in vocab
        # Verify sorted
        assert vocab == sorted(vocab)

    def test_get_vocabulary_returns_copy(self, schema_file):
        """Test that get_vocabulary returns a copy."""
        loader = HedJsonSchemaLoader(schema_file)
        vocab1 = loader.get_vocabulary()
        vocab2 = loader.get_vocabulary()
        vocab1.append("TestTag")
        assert "TestTag" not in vocab2

    def test_get_extendable_tags(self, schema_file):
        """Test extraction of extensionAllowed tags."""
        loader = HedJsonSchemaLoader(schema_file)
        extendable = loader.get_extendable_tags()

        assert isinstance(extendable, dict)
        assert "Event" in extendable
        assert "Item" in extendable
        assert "Red" not in extendable  # Red is not extendable
        assert extendable["Event"] == "Something that happens"

    def test_get_extendable_tags_returns_copy(self, schema_file):
        """Test that get_extendable_tags returns a copy."""
        loader = HedJsonSchemaLoader(schema_file)
        ext1 = loader.get_extendable_tags()
        ext2 = loader.get_extendable_tags()
        ext1["NewTag"] = "New description"
        assert "NewTag" not in ext2

    def test_is_extendable(self, schema_file):
        """Test is_extendable method."""
        loader = HedJsonSchemaLoader(schema_file)

        assert loader.is_extendable("Event") is True
        assert loader.is_extendable("Item") is True
        assert loader.is_extendable("Red") is False
        assert loader.is_extendable("NonExistent") is False

    def test_get_schema_version(self, schema_file):
        """Test schema version extraction."""
        loader = HedJsonSchemaLoader(schema_file)
        version = loader.get_schema_version()
        assert version == "8.3.0"

    def test_get_schema_version_unknown(self, tmp_path):
        """Test schema version when not present."""
        schema_path = tmp_path / "no_version.json"
        with open(schema_path, "w") as f:
            json.dump({"tags": {}}, f)

        loader = HedJsonSchemaLoader(schema_path)
        assert loader.get_schema_version() == "unknown"

    def test_find_closest_match_exact(self, schema_file):
        """Test finding exact case-insensitive match."""
        loader = HedJsonSchemaLoader(schema_file)

        # Exact match with different case
        matches = loader.find_closest_match("red")
        assert matches == ["Red"]

        matches = loader.find_closest_match("EVENT")
        assert matches == ["Event"]

    def test_find_closest_match_contains(self, schema_file):
        """Test finding matches by containment."""
        loader = HedJsonSchemaLoader(schema_file)

        # Partial match
        matches = loader.find_closest_match("Sens")
        assert "Sensory-event" in matches

    def test_find_closest_match_no_match(self, schema_file):
        """Test when no matches found."""
        loader = HedJsonSchemaLoader(schema_file)
        matches = loader.find_closest_match("XYZNonExistent")
        assert matches == []

    def test_find_closest_match_limit(self, schema_file):
        """Test that matches are limited to 5."""
        loader = HedJsonSchemaLoader(schema_file)
        # "e" appears in many tags
        matches = loader.find_closest_match("e")
        assert len(matches) <= 5


class TestLoadLatestSchema:
    """Tests for load_latest_schema function."""

    def test_load_latest_schema(self, schema_dir):
        """Test loading latest schema from directory."""
        loader = load_latest_schema(schema_dir)
        assert isinstance(loader, HedJsonSchemaLoader)
        assert loader.get_schema_version() == "8.3.0"

    def test_load_latest_schema_with_string_path(self, schema_dir):
        """Test loading with string path."""
        loader = load_latest_schema(str(schema_dir))
        assert isinstance(loader, HedJsonSchemaLoader)

    def test_load_latest_schema_none_raises(self):
        """Test that None schema_dir raises ValueError."""
        with pytest.raises(ValueError, match="schema_dir is required"):
            load_latest_schema(None)

    def test_load_latest_schema_missing_file(self, tmp_path):
        """Test error when HEDLatest.json is missing."""
        with pytest.raises(FileNotFoundError):
            load_latest_schema(tmp_path)


class TestSchemaEdgeCases:
    """Tests for edge cases in schema loading."""

    def test_empty_tags(self, tmp_path):
        """Test schema with empty tags."""
        schema_path = tmp_path / "empty.json"
        with open(schema_path, "w") as f:
            json.dump({"version": "1.0", "tags": {}}, f)

        loader = HedJsonSchemaLoader(schema_path)
        assert loader.get_vocabulary() == []
        assert loader.get_extendable_tags() == {}

    def test_nested_tag_data_non_dict(self, tmp_path):
        """Test handling of non-dict tag data."""
        schema_path = tmp_path / "mixed.json"
        with open(schema_path, "w") as f:
            json.dump(
                {
                    "version": "1.0",
                    "tags": {
                        "ValidTag": {"short_form": "ValidTag"},
                        "InvalidTag": "just a string",  # Non-dict value
                    },
                },
                f,
            )

        loader = HedJsonSchemaLoader(schema_path)
        vocab = loader.get_vocabulary()
        # Only ValidTag should be in vocabulary
        assert "ValidTag" in vocab

    def test_tag_without_short_form(self, tmp_path):
        """Test tag that uses tag name as short form."""
        schema_path = tmp_path / "no_short.json"
        with open(schema_path, "w") as f:
            json.dump(
                {
                    "version": "1.0",
                    "tags": {"MyTag": {"description": "A tag without explicit short_form"}},
                },
                f,
            )

        loader = HedJsonSchemaLoader(schema_path)
        vocab = loader.get_vocabulary()
        # Should use tag name as short form
        assert "MyTag" in vocab
