"""Validation Agent for HED annotation validation.

This agent validates HED annotation strings using HED validation tools
and provides detailed feedback for corrections.
"""

import re
from pathlib import Path

from src.agents.state import HedAnnotationState
from src.utils.error_remediation import get_remediator
from src.utils.schema_loader import HedSchemaLoader
from src.validation.hed_validator import (
    HedJavaScriptValidator,
    HedPythonValidator,
    ValidationResult,
)


def strip_extensions(annotation: str, extended_tags: list[str]) -> str:
    """Strip extensions from an annotation, replacing with base tags.

    For each extended tag like 'Animal/Marmoset', replaces with just 'Animal'.
    Handles both simple replacements and preserves HED structure.

    Args:
        annotation: The HED annotation string
        extended_tags: List of extended tags to strip (e.g., ['Animal/Marmoset'])

    Returns:
        Annotation with extensions stripped
    """
    result = annotation

    for extended_tag in extended_tags:
        if "/" not in extended_tag:
            continue

        # Split into base and extension: "Animal/Marmoset" -> ("Animal", "Marmoset")
        parts = extended_tag.split("/", 1)
        base_tag = parts[0]

        # Replace the extended tag with just the base
        # Use word boundaries to avoid partial matches
        # Handle cases like (Animal/Marmoset, ...) or Animal/Marmoset,
        pattern = re.escape(extended_tag)
        result = re.sub(pattern, base_tag, result)

    return result


class ValidationAgent:
    """Agent that validates HED annotations using HED validation tools.

    Supports both JavaScript validator (detailed feedback) and Python validator (fallback).
    """

    def __init__(
        self,
        schema_loader: HedSchemaLoader,
        use_javascript: bool = True,
        validator_path: Path | None = None,
        tests_json_path: Path | str | None = None,
    ) -> None:
        """Initialize the validation agent.

        Args:
            schema_loader: HED schema loader
            use_javascript: Whether to use JavaScript validator (more detailed)
            validator_path: Path to hed-javascript repository (required if use_javascript=True)
            tests_json_path: Optional path to javascriptTests.json for error remediation
        """
        self.schema_loader = schema_loader
        self.use_javascript = use_javascript
        self.error_remediator = get_remediator(tests_json_path)

        # Declare validators with proper types
        self.js_validator: HedJavaScriptValidator | None = None
        self.py_validator: HedPythonValidator | None = None

        if use_javascript:
            if validator_path is None:
                raise ValueError("validator_path required when use_javascript=True")
            self.js_validator = HedJavaScriptValidator(validator_path)

    async def validate(self, state: HedAnnotationState) -> dict:
        """Validate the current HED annotation.

        Args:
            state: Current annotation workflow state

        Returns:
            State update with validation results
        """
        annotation = state["current_annotation"]
        schema_version = state["schema_version"]
        no_extend = state.get("no_extend", False)

        # Validate using appropriate validator
        result = self._run_validation(annotation, schema_version)

        # If no_extend is True, strip any extensions and re-validate
        stripped_annotation = None
        if no_extend:
            extended_tags = self._extract_extended_tags(result)
            if extended_tags:
                stripped_annotation = strip_extensions(annotation, extended_tags)
                # Re-validate the stripped annotation
                result = self._run_validation(stripped_annotation, schema_version)

        # Extract error and warning messages (raw - for user display)
        raw_errors = [f"[{e.code}] {e.message}" for e in result.errors]
        raw_warnings = [f"[{w.code}] {w.message}" for w in result.warnings]

        # Augment with remediation guidance (for LLM feedback loop only)
        augmented_errors, augmented_warnings = self.error_remediator.augment_validation_errors(
            raw_errors, raw_warnings
        )

        # Determine validation status
        validation_attempts = state["validation_attempts"] + 1
        max_attempts = state["max_validation_attempts"]

        # IMPORTANT: Safeguard to ensure is_valid is only True when there are NO errors
        # This prevents discrepancies between is_valid flag and actual validation_errors
        is_valid = result.is_valid and len(raw_errors) == 0

        if is_valid:
            validation_status = "valid"
        elif validation_attempts >= max_attempts:
            validation_status = "max_attempts_reached"
        else:
            validation_status = "invalid"

        # Build result dict
        result_dict = {
            "validation_status": validation_status,
            "validation_errors": raw_errors,  # Raw errors for user display
            "validation_warnings": raw_warnings,  # Raw warnings for user display
            "validation_errors_augmented": augmented_errors,  # For LLM feedback
            "validation_warnings_augmented": augmented_warnings,  # For LLM feedback
            "validation_attempts": validation_attempts,
            "is_valid": is_valid,
        }

        # If we stripped extensions, update the annotation in the result
        if stripped_annotation is not None:
            result_dict["current_annotation"] = stripped_annotation

        return result_dict

    def _run_validation(self, annotation: str, schema_version: str) -> ValidationResult:
        """Run validation on an annotation string.

        Args:
            annotation: HED annotation to validate
            schema_version: Schema version to validate against

        Returns:
            ValidationResult with errors and warnings
        """
        if self.use_javascript and self.js_validator:
            return self.js_validator.validate(annotation)
        else:
            # Use Python validator
            schema = self.schema_loader.load_schema(schema_version)
            if self.py_validator is None:
                self.py_validator = HedPythonValidator(schema)
            return self.py_validator.validate(annotation)

    def _extract_extended_tags(self, result: ValidationResult) -> list[str]:
        """Extract extended tags from TAG_EXTENDED warnings.

        Handles both formats:
        - JavaScript validator: tag field contains the extended tag
        - Python validator: message contains "... in Animal/Marmoset" pattern

        Args:
            result: Validation result containing warnings

        Returns:
            List of extended tag strings (e.g., ['Animal/Marmoset', 'Building/Cottage'])
        """
        extended_tags = []
        for warning in result.warnings:
            if warning.code == "TAG_EXTENDED":
                # Try tag field first (JavaScript validator)
                if warning.tag:
                    extended_tags.append(warning.tag)
                # Fall back to parsing message (Python validator)
                # Format: "TAG_EXTENDED: ... '/Extension' in Parent/Extension\n"
                elif warning.message and " in " in warning.message:
                    # Extract the full tag after " in "
                    parts = warning.message.split(" in ")
                    if len(parts) >= 2:
                        full_tag = parts[-1].strip()
                        # Remove any trailing punctuation, whitespace, or newlines
                        full_tag = full_tag.rstrip(".\n\r\t ")
                        if "/" in full_tag:
                            extended_tags.append(full_tag)
        return extended_tags
