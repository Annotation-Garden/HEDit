"""Tests for error remediation module."""

import pytest

from src.utils.error_remediation import ErrorRemediator, get_remediator


@pytest.fixture
def remediator():
    """Create an ErrorRemediator instance."""
    return ErrorRemediator()


def test_error_remediator_initialization():
    """Test that ErrorRemediator initializes correctly."""
    remediator = ErrorRemediator()
    assert remediator is not None
    assert remediator.tests_data == []


def test_get_remediator_singleton():
    """Test that get_remediator returns a singleton instance."""
    rem1 = get_remediator()
    rem2 = get_remediator()
    assert rem1 is rem2


def test_get_remediation_tag_extended(remediator):
    """Test remediation guidance for TAG_EXTENDED warning."""
    guidance = remediator.get_remediation("TAG_EXTENDED")

    assert "REMEDIATION for TAG_EXTENDED" in guidance
    assert "MOST SPECIFIC" in guidance
    assert "is-a" in guidance.lower()


def test_tag_extended_has_diverse_examples(remediator):
    """Test that TAG_EXTENDED guidance has diverse examples from different schema trees."""
    guidance = remediator.get_remediation("TAG_EXTENDED")

    # Check for examples from different schema areas
    diverse_examples = [
        ("Building/Cottage", "buildings (Item tree)"),
        ("Move-body/Cartwheel", "actions (Action tree)"),
        ("Furniture/Armoire", "furniture (Item tree)"),
        ("Vehicle/Rickshaw", "vehicles (Item tree)"),
        ("Mammal/Dolphin", "animals (Agent tree)"),
    ]

    for example, tree in diverse_examples:
        assert example in guidance, f"Missing {tree} example: {example}"


def test_get_remediation_tag_extension_invalid(remediator):
    """Test remediation guidance for TAG_EXTENSION_INVALID error."""
    guidance = remediator.get_remediation("TAG_EXTENSION_INVALID")

    assert "REMEDIATION for TAG_EXTENSION_INVALID" in guidance
    assert "vocabulary" in guidance.lower()


def test_get_remediation_tag_invalid(remediator):
    """Test remediation guidance for TAG_INVALID error."""
    guidance = remediator.get_remediation("TAG_INVALID")

    assert "REMEDIATION for TAG_INVALID" in guidance
    assert "schema" in guidance.lower() or "vocabulary" in guidance.lower()


def test_get_remediation_definition_invalid(remediator):
    """Test remediation guidance for DEFINITION_INVALID error."""
    guidance = remediator.get_remediation("DEFINITION_INVALID")

    assert "REMEDIATION for DEFINITION_INVALID" in guidance
    assert "Definition" in guidance


def test_get_remediation_def_invalid(remediator):
    """Test remediation guidance for DEF_INVALID error."""
    guidance = remediator.get_remediation("DEF_INVALID")

    assert "REMEDIATION for DEF_INVALID" in guidance
    assert "Def" in guidance


def test_get_remediation_parentheses_mismatch(remediator):
    """Test remediation guidance for PARENTHESES_MISMATCH error."""
    guidance = remediator.get_remediation("PARENTHESES_MISMATCH")

    assert "REMEDIATION for PARENTHESES_MISMATCH" in guidance
    assert "parenthes" in guidance.lower()


def test_get_remediation_unknown_code(remediator):
    """Test remediation for unknown error code."""
    guidance = remediator.get_remediation("UNKNOWN_ERROR_CODE_12345")

    assert "No specific remediation guidance" in guidance


def test_augment_validation_errors(remediator):
    """Test augmenting validation errors with remediation."""
    errors = [
        "[TAG_INVALID] The tag 'Fake-tag' is not valid",
        "[TAG_EXTENSION_INVALID] Extension term already exists",
    ]
    warnings = [
        "[TAG_EXTENDED] Tag was extended from schema",
    ]

    aug_errors, aug_warnings = remediator.augment_validation_errors(errors, warnings)

    # Check that errors are augmented
    assert len(aug_errors) == 2
    assert "REMEDIATION" in aug_errors[0]
    assert "REMEDIATION" in aug_errors[1]

    # Check that warnings are augmented
    assert len(aug_warnings) == 1
    assert "REMEDIATION" in aug_warnings[0]


def test_augment_validation_errors_no_warnings(remediator):
    """Test augmenting errors when no warnings provided."""
    errors = ["[TAG_INVALID] Invalid tag"]

    aug_errors, aug_warnings = remediator.augment_validation_errors(errors)

    assert len(aug_errors) == 1
    assert aug_warnings == []


def test_augment_validation_errors_empty(remediator):
    """Test augmenting empty error list."""
    aug_errors, aug_warnings = remediator.augment_validation_errors([], [])

    assert aug_errors == []
    assert aug_warnings == []


def test_extract_error_code(remediator):
    """Test error code extraction from message."""
    # Standard format
    code = remediator._extract_error_code("[TAG_INVALID] Some message")
    assert code == "TAG_INVALID"

    # With complex code
    code = remediator._extract_error_code("[TAG_EXTENSION_INVALID] Another message")
    assert code == "TAG_EXTENSION_INVALID"

    # No brackets
    code = remediator._extract_error_code("No brackets here")
    assert code is None


def test_remediation_includes_examples(remediator):
    """Test that remediation includes examples."""
    guidance = remediator.get_remediation("TAG_EXTENDED")

    # Should include wrong and correct examples
    assert "WRONG" in guidance or "wrong" in guidance.lower()
    assert "CORRECT" in guidance or "correct" in guidance.lower()


def test_remediation_for_common_errors(remediator):
    """Test that common error codes have remediation guidance."""
    common_codes = [
        "TAG_EXTENDED",
        "TAG_EXTENSION_INVALID",
        "TAG_INVALID",
        "TAG_REQUIRES_CHILD",
        "DEFINITION_INVALID",
        "DEF_INVALID",
        "PARENTHESES_MISMATCH",
        "COMMA_MISSING",
        "TAG_EMPTY",
        "VALUE_INVALID",
        "UNITS_INVALID",
        "PLACEHOLDER_INVALID",
        "SIDECAR_BRACES_INVALID",
        "TEMPORAL_TAG_ERROR",
        "CHARACTER_INVALID",
    ]

    for code in common_codes:
        guidance = remediator.get_remediation(code)
        assert "REMEDIATION" in guidance, f"No remediation for {code}"
        assert code in guidance, f"Code {code} not in its own remediation"
