"""Tests for HED validation."""

import pytest

from src.agents.state import create_initial_state
from src.agents.validation_agent import ValidationAgent
from src.utils.schema_loader import HedSchemaLoader
from src.validation.hed_validator import HedPythonValidator


@pytest.fixture
def validator():
    """Create a HED Python validator."""
    loader = HedSchemaLoader()
    schema = loader.load_schema("8.3.0")
    return HedPythonValidator(schema)


def test_validate_valid_string(validator):
    """Test validation of a valid HED string."""
    result = validator.validate("Sensory-event, Visual-presentation")

    assert result.is_valid is True
    assert len(result.errors) == 0


def test_validate_invalid_tag(validator):
    """Test validation of invalid tag."""
    result = validator.validate("Invalid-nonexistent-tag")

    assert result.is_valid is False
    assert len(result.errors) > 0


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
    state = create_initial_state(
        "A person sees a red circle",
        schema_version="8.3.0"
    )
    state["current_annotation"] = "Sensory-event, Visual-presentation"

    result = await validation_agent.validate(state)

    # If is_valid is True, validation_errors MUST be empty
    if result["is_valid"]:
        assert len(result["validation_errors"]) == 0, \
            "is_valid is True but validation_errors is not empty"

    # Test with invalid annotation
    state["current_annotation"] = "Invalid-nonexistent-tag"
    result = await validation_agent.validate(state)

    # If validation_errors is not empty, is_valid MUST be False
    if len(result["validation_errors"]) > 0:
        assert result["is_valid"] is False, \
            "validation_errors is not empty but is_valid is True"

    # If is_valid is False, there should be errors or max attempts reached
    if not result["is_valid"]:
        assert len(result["validation_errors"]) > 0 or \
               result["validation_status"] == "max_attempts_reached", \
            "is_valid is False but no validation_errors and not max_attempts"


@pytest.mark.asyncio
async def test_validation_agent_safeguard(validation_agent):
    """Test the safeguard that ensures is_valid consistency.

    This test specifically checks the safeguard added to fix issue #6.
    """
    # Create state with multiple validation attempts
    state = create_initial_state(
        "Test description",
        schema_version="8.3.0",
        max_validation_attempts=3
    )

    # Test multiple invalid annotations
    invalid_annotations = [
        "Invalid-tag-1",
        "Another-invalid-tag",
        "Yet-another-bad-tag"
    ]

    for annotation in invalid_annotations:
        state["current_annotation"] = annotation
        result = await validation_agent.validate(state)

        # Critical check: is_valid and validation_errors must be consistent
        has_errors = len(result["validation_errors"]) > 0

        # The safeguard ensures: is_valid can only be True if there are NO errors
        assert not (result["is_valid"] and has_errors), \
            f"SAFEGUARD FAILED: is_valid={result['is_valid']} but has {len(result['validation_errors'])} errors"

        # Update state for next iteration
        state["validation_attempts"] = result["validation_attempts"]
