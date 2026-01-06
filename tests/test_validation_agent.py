"""Tests for validation agent functionality."""

import pytest

from src.agents.validation_agent import ValidationAgent, strip_extensions
from src.utils.schema_loader import HedSchemaLoader
from src.validation.hed_validator import HedPythonValidator, ValidationIssue, ValidationResult


class TestStripExtensions:
    """Tests for the strip_extensions utility function."""

    def test_strip_single_extension(self):
        """Strip a single extension from annotation."""
        annotation = "(Animal-agent, Animal/Marmoset)"
        extended_tags = ["Animal/Marmoset"]

        result = strip_extensions(annotation, extended_tags)

        assert result == "(Animal-agent, Animal)"
        assert "Marmoset" not in result

    def test_strip_multiple_extensions(self):
        """Strip multiple extensions from annotation."""
        annotation = "((Animal-agent, Animal/Dolphin), (Move, Building/Maze))"
        extended_tags = ["Animal/Dolphin", "Building/Maze"]

        result = strip_extensions(annotation, extended_tags)

        assert result == "((Animal-agent, Animal), (Move, Building))"
        assert "Dolphin" not in result
        assert "Maze" not in result

    def test_strip_nested_extension(self):
        """Strip extension with multiple levels."""
        annotation = "(Animal-agent, Animal/Mammal/Marmoset)"
        extended_tags = ["Animal/Mammal/Marmoset"]

        result = strip_extensions(annotation, extended_tags)

        # Should replace the whole path with just the base
        assert result == "(Animal-agent, Animal)"

    def test_no_extensions_to_strip(self):
        """Annotation without extensions remains unchanged."""
        annotation = "Sensory-event, Visual-presentation, (Red, Circle)"
        extended_tags = []

        result = strip_extensions(annotation, extended_tags)

        assert result == annotation

    def test_extension_not_in_annotation(self):
        """Extension not present in annotation - no change."""
        annotation = "Sensory-event, Visual-presentation"
        extended_tags = ["Animal/Marmoset"]

        result = strip_extensions(annotation, extended_tags)

        assert result == annotation

    def test_preserves_value_tags(self):
        """Value tags with units should NOT be stripped (they're not extensions)."""
        annotation = "Sensory-event, Duration/2 s, Frequency/440 Hz"
        extended_tags = []

        result = strip_extensions(annotation, extended_tags)

        assert result == annotation
        assert "Duration/2 s" in result
        assert "Frequency/440 Hz" in result

    def test_strip_preserves_structure(self):
        """Stripping preserves parentheses and comma structure."""
        annotation = "Agent-action, ((Animal-agent, Animal/Marmoset), (Press, Button))"
        extended_tags = ["Animal/Marmoset"]

        result = strip_extensions(annotation, extended_tags)

        assert result == "Agent-action, ((Animal-agent, Animal), (Press, Button))"
        # Verify structure preserved
        assert result.count("(") == result.count(")")
        assert ", " in result

    def test_case_sensitive_matching(self):
        """Extension matching is case-sensitive."""
        annotation = "(Animal/Marmoset, animal/marmoset)"
        extended_tags = ["Animal/Marmoset"]

        result = strip_extensions(annotation, extended_tags)

        # Only the matching case should be stripped
        assert result == "(Animal, animal/marmoset)"


class TestValidationAgentExtraction:
    """Tests for extracting extended tags from validation results."""

    @pytest.fixture
    def validator(self):
        """Create a Python validator for testing."""
        loader = HedSchemaLoader()
        schema = loader.load_schema("8.4.0")
        return HedPythonValidator(schema)

    @pytest.fixture
    def validation_agent(self):
        """Create a validation agent using Python validator."""
        loader = HedSchemaLoader()
        return ValidationAgent(loader, use_javascript=False)

    def test_extract_extended_tags_from_python_validator(self, validator, validation_agent):
        """Extract extended tags from Python validator output."""
        result = validator.validate("Animal/Marmoset")

        extended_tags = validation_agent._extract_extended_tags(result)

        assert len(extended_tags) == 1
        assert "Animal/Marmoset" in extended_tags

    def test_extract_multiple_extended_tags(self, validator, validation_agent):
        """Extract multiple extended tags."""
        result = validator.validate("Animal/Marmoset, Building/Cottage")

        extended_tags = validation_agent._extract_extended_tags(result)

        assert len(extended_tags) == 2
        assert "Animal/Marmoset" in extended_tags
        assert "Building/Cottage" in extended_tags

    def test_extract_no_extensions(self, validator, validation_agent):
        """No extensions when using only base tags."""
        result = validator.validate("Sensory-event, Red, Circle")

        extended_tags = validation_agent._extract_extended_tags(result)

        assert len(extended_tags) == 0

    def test_extract_from_javascript_format(self, validation_agent):
        """Extract from JavaScript validator format (tag field populated)."""
        # Simulate JavaScript validator output
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[
                ValidationIssue(
                    code="TAG_EXTENDED",
                    level="warning",
                    message="Tag was extended",
                    tag="Animal/Marmoset",
                )
            ],
        )

        extended_tags = validation_agent._extract_extended_tags(result)

        assert len(extended_tags) == 1
        assert "Animal/Marmoset" in extended_tags


class TestValidationAgentNoExtend:
    """Integration tests for no_extend functionality."""

    @pytest.fixture
    def validation_agent(self):
        """Create a validation agent using Python validator."""
        loader = HedSchemaLoader()
        return ValidationAgent(loader, use_javascript=False)

    @pytest.mark.asyncio
    async def test_no_extend_strips_extensions(self, validation_agent):
        """With no_extend=True, extensions are stripped from annotation."""
        state = {
            "current_annotation": "(Animal-agent, Animal/Marmoset), Sensory-event",
            "schema_version": "8.4.0",
            "validation_attempts": 0,
            "max_validation_attempts": 3,
            "no_extend": True,
        }

        result = await validation_agent.validate(state)

        # The annotation should be stripped
        assert "current_annotation" in result
        assert "Marmoset" not in result["current_annotation"]
        assert "Animal" in result["current_annotation"]
        # Should be: "(Animal-agent, Animal), Sensory-event"
        assert result["current_annotation"] == "(Animal-agent, Animal), Sensory-event"

    @pytest.mark.asyncio
    async def test_no_extend_false_keeps_extensions(self, validation_agent):
        """With no_extend=False (default), extensions are kept."""
        state = {
            "current_annotation": "(Animal-agent, Animal/Marmoset), Sensory-event",
            "schema_version": "8.4.0",
            "validation_attempts": 0,
            "max_validation_attempts": 3,
            "no_extend": False,
        }

        result = await validation_agent.validate(state)

        # current_annotation should NOT be in result (no change)
        assert "current_annotation" not in result
        # The original state is unchanged

    @pytest.mark.asyncio
    async def test_no_extend_strips_multiple_extensions(self, validation_agent):
        """Strip multiple extensions when no_extend=True."""
        state = {
            "current_annotation": "Animal/Dolphin, Building/Maze, Sensory-event",
            "schema_version": "8.4.0",
            "validation_attempts": 0,
            "max_validation_attempts": 3,
            "no_extend": True,
        }

        result = await validation_agent.validate(state)

        assert "current_annotation" in result
        stripped = result["current_annotation"]
        assert "Dolphin" not in stripped
        assert "Maze" not in stripped
        assert "Animal" in stripped
        assert "Building" in stripped

    @pytest.mark.asyncio
    async def test_no_extend_with_no_extensions(self, validation_agent):
        """No change when annotation has no extensions."""
        state = {
            "current_annotation": "Sensory-event, Red, Circle",
            "schema_version": "8.4.0",
            "validation_attempts": 0,
            "max_validation_attempts": 3,
            "no_extend": True,
        }

        result = await validation_agent.validate(state)

        # No stripping needed, so current_annotation not in result
        assert "current_annotation" not in result

    @pytest.mark.asyncio
    async def test_stripped_annotation_is_valid(self, validation_agent):
        """Stripped annotation should be valid."""
        state = {
            "current_annotation": "(Animal-agent, Animal/Marmoset), Sensory-event",
            "schema_version": "8.4.0",
            "validation_attempts": 0,
            "max_validation_attempts": 3,
            "no_extend": True,
        }

        result = await validation_agent.validate(state)

        # Should be valid (no errors)
        assert result["is_valid"] is True
        # Should have no TAG_EXTENDED warnings after stripping
        assert not any("TAG_EXTENDED" in w for w in result["validation_warnings"])
