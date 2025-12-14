"""Tests for HED schema loader."""

import pytest

from src.utils.schema_loader import HedSchemaLoader, get_schema_loader


def test_schema_loader_init():
    """Test schema loader initialization."""
    loader = HedSchemaLoader()
    assert loader is not None
    assert loader._cache == {}


def test_load_schema_by_version():
    """Test loading schema by version string."""
    loader = HedSchemaLoader()
    schema = loader.load_schema("8.3.0")

    assert schema is not None
    assert schema.version == "8.3.0"


def test_schema_caching():
    """Test that schemas are cached after first load."""
    loader = HedSchemaLoader()

    # Load schema twice
    schema1 = loader.load_schema("8.3.0")
    schema2 = loader.load_schema("8.3.0")

    # Should be same object (cached)
    assert schema1 is schema2


@pytest.mark.skip(reason="HED library API changed - get_all_schema_tags no longer available")
def test_get_schema_vocabulary():
    """Test extracting vocabulary from schema."""
    loader = HedSchemaLoader()
    schema = loader.load_schema("8.3.0")

    vocabulary = loader.get_schema_vocabulary(schema)

    assert isinstance(vocabulary, list)
    assert len(vocabulary) > 0
    assert "Sensory-event" in vocabulary or "Event" in vocabulary


def test_clear_cache():
    """Test clearing schema cache."""
    loader = HedSchemaLoader()

    # Load and cache a schema
    loader.load_schema("8.3.0")
    assert len(loader._cache) > 0

    # Clear cache
    loader.clear_cache()
    assert len(loader._cache) == 0


def test_get_default_loader():
    """Test getting the default global loader."""
    loader1 = get_schema_loader()
    loader2 = get_schema_loader()

    # Should be the same instance
    assert loader1 is loader2


def test_invalid_schema_version():
    """Test handling of invalid schema version."""
    loader = HedSchemaLoader()

    with pytest.raises(ValueError):
        loader.load_schema("999.999.999")
