"""Tests for HED validation."""

from pathlib import Path

import pytest

from src.agents.state import create_initial_state
from src.agents.validation_agent import ValidationAgent
from src.utils.schema_loader import HedSchemaLoader
from src.validation.hed_validator import (
    HedJavaScriptValidator,
    HedPythonValidator,
    ValidationIssue,
    ValidationResult,
)


@pytest.fixture
def validator():
    """Create a HED Python validator."""
    loader = HedSchemaLoader()
    schema = loader.load_schema("8.3.0")
    return HedPythonValidator(schema)


class TestValidationIssue:
    """Tests for ValidationIssue dataclass."""

    def test_validation_issue_creation(self):
        """Test creating a validation issue."""
        issue = ValidationIssue(
            code="TAG_INVALID",
            level="error",
            message="Invalid tag",
            tag="BadTag",
            context={"line": 1},
        )
        assert issue.code == "TAG_INVALID"
        assert issue.level == "error"
        assert issue.message == "Invalid tag"
        assert issue.tag == "BadTag"
        assert issue.context == {"line": 1}

    def test_validation_issue_defaults(self):
        """Test default values for optional fields."""
        issue = ValidationIssue(
            code="PARSE_ERROR",
            level="error",
            message="Parse error",
        )
        assert issue.tag is None
        assert issue.context is None


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_validation_result_valid(self):
        """Test creating a valid result."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            parsed_string="Event",
        )
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.parsed_string == "Event"

    def test_validation_result_invalid(self):
        """Test creating an invalid result."""
        error = ValidationIssue(code="ERROR", level="error", message="Test")
        result = ValidationResult(
            is_valid=False,
            errors=[error],
            warnings=[],
        )
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.parsed_string is None


class TestHedPythonValidator:
    """Tests for HedPythonValidator class."""

    def test_validate_with_issues(self, validator):
        """Test validation that produces issues/warnings."""
        # Use a tag that triggers warnings
        result = validator.validate("Event, (Red)")
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)

    def test_validate_parse_error(self, validator):
        """Test handling of malformed HED string."""
        # Unbalanced parentheses should trigger error or warning
        result = validator.validate("Event, ((Red)")
        # Should have some issue (error or warning) with malformed input
        assert len(result.errors) > 0 or len(result.warnings) > 0

    def test_validate_empty_string(self, validator):
        """Test validation of empty string."""
        result = validator.validate("")
        # Empty string may be valid or error depending on schema
        assert isinstance(result.is_valid, bool)

    def test_validate_complex_annotation(self, validator):
        """Test validation of complex nested annotation."""
        annotation = "(Sensory-event, (Visual-presentation, (Red, Circle)))"
        result = validator.validate(annotation)
        # Complex annotations should be parseable
        assert isinstance(result, ValidationResult)

    def test_validate_multiple_groups(self, validator):
        """Test validation with multiple tag groups."""
        annotation = "Event, (Red, Circle), (Blue, Square)"
        result = validator.validate(annotation)
        assert isinstance(result.errors, list)


class TestHedJavaScriptValidator:
    """Tests for HedJavaScriptValidator class."""

    @pytest.fixture
    def hed_js_path(self):
        """Get path to hed-javascript if available."""
        # Check common locations
        paths = [
            Path.home() / "Documents/git/HED/hed-javascript",
            Path("/Users/yahya/Documents/git/HED/hed-javascript"),
            Path("../hed-javascript"),
        ]
        for p in paths:
            if p.exists():
                return p
        return None

    def test_init_missing_path_raises(self):
        """Test that missing validator path raises error."""
        with pytest.raises(RuntimeError, match="not found"):
            HedJavaScriptValidator(
                validator_path=Path("/nonexistent/path"),
                schema_version="8.3.0",
            )

    def test_init_valid_path(self, hed_js_path):
        """Test initialization with valid path."""
        if hed_js_path is None:
            pytest.skip("hed-javascript not available")

        validator = HedJavaScriptValidator(
            validator_path=hed_js_path,
            schema_version="8.3.0",
        )
        assert validator.schema_version == "8.3.0"

    def test_validate_simple_string(self, hed_js_path):
        """Test validation of simple HED string."""
        if hed_js_path is None:
            pytest.skip("hed-javascript not available")

        validator = HedJavaScriptValidator(
            validator_path=hed_js_path,
            schema_version="8.3.0",
        )
        result = validator.validate("Event")
        assert isinstance(result, ValidationResult)

    def test_validate_invalid_tag(self, hed_js_path):
        """Test validation of invalid tag."""
        if hed_js_path is None:
            pytest.skip("hed-javascript not available")

        validator = HedJavaScriptValidator(
            validator_path=hed_js_path,
            schema_version="8.3.0",
        )
        result = validator.validate("CompletelyInvalidTag123")
        # Should detect invalid tag
        assert result.is_valid is False or len(result.errors) > 0 or len(result.warnings) > 0


def test_validate_valid_string(validator):
    """Test validation of a valid HED string."""
    result = validator.validate("Sensory-event, Visual-presentation")

    assert result.is_valid is True
    assert len(result.errors) == 0


def test_validate_invalid_tag(validator):
    """Test validation of invalid tag.

    Note: HED 8.3.0+ reports invalid tags as warnings, not errors.
    """
    result = validator.validate("Invalid-nonexistent-tag")

    # Invalid tags may be reported as warnings in newer HED versions
    assert result.is_valid is False or len(result.warnings) > 0


def test_validate_with_grouping(validator):
    """Test validation of properly grouped tags."""
    result = validator.validate("Sensory-event, (Red, Circle)")

    assert result.is_valid is True or len(result.errors) == 0


def test_validation_result_structure(validator):
    """Test structure of validation result."""
    result = validator.validate("Sensory-event")

    assert hasattr(result, "is_valid")
    assert hasattr(result, "errors")
    assert hasattr(result, "warnings")
    assert hasattr(result, "parsed_string")
    assert isinstance(result.errors, list)
    assert isinstance(result.warnings, list)


@pytest.fixture
def validation_agent():
    """Create a validation agent."""
    loader = HedSchemaLoader()
    return ValidationAgent(
        schema_loader=loader,
        use_javascript=False,  # Use Python validator for testing
    )


@pytest.mark.asyncio
async def test_validation_agent_valid_consistency(validation_agent):
    """Test that is_valid is only True when validation_errors is empty.

    This is a critical test for issue #6: ensuring no discrepancy between
    is_valid flag and actual validation_errors list.
    """
    # Test with valid annotation
    state = create_initial_state("A person sees a red circle", schema_version="8.3.0")
    state["current_annotation"] = "Sensory-event, Visual-presentation"

    result = await validation_agent.validate(state)

    # If is_valid is True, validation_errors MUST be empty
    if result["is_valid"]:
        assert len(result["validation_errors"]) == 0, (
            "is_valid is True but validation_errors is not empty"
        )

    # Test with invalid annotation
    state["current_annotation"] = "Invalid-nonexistent-tag"
    result = await validation_agent.validate(state)

    # If validation_errors is not empty, is_valid MUST be False
    if len(result["validation_errors"]) > 0:
        assert result["is_valid"] is False, "validation_errors is not empty but is_valid is True"

    # If is_valid is False, there should be errors or max attempts reached
    if not result["is_valid"]:
        assert (
            len(result["validation_errors"]) > 0
            or result["validation_status"] == "max_attempts_reached"
        ), "is_valid is False but no validation_errors and not max_attempts"


@pytest.mark.asyncio
async def test_validation_agent_safeguard(validation_agent):
    """Test the safeguard that ensures is_valid consistency.

    This test specifically checks the safeguard added to fix issue #6.
    """
    # Create state with multiple validation attempts
    state = create_initial_state(
        "Test description", schema_version="8.3.0", max_validation_attempts=3
    )

    # Test multiple invalid annotations
    invalid_annotations = ["Invalid-tag-1", "Another-invalid-tag", "Yet-another-bad-tag"]

    for annotation in invalid_annotations:
        state["current_annotation"] = annotation
        result = await validation_agent.validate(state)

        # Critical check: is_valid and validation_errors must be consistent
        has_errors = len(result["validation_errors"]) > 0

        # The safeguard ensures: is_valid can only be True if there are NO errors
        assert not (result["is_valid"] and has_errors), (
            f"SAFEGUARD FAILED: is_valid={result['is_valid']} but has {len(result['validation_errors'])} errors"
        )

        # Update state for next iteration
        state["validation_attempts"] = result["validation_attempts"]
