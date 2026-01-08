"""Validation Agent for HED annotation validation.

This agent validates HED annotation strings using HED validation tools
and provides detailed feedback for corrections. When invalid tags are found,
it uses hed-lsp CLI to suggest valid alternatives.
"""

import logging
import re
from pathlib import Path

from src.agents.state import HedAnnotationState
from src.utils.error_remediation import get_remediator
from src.utils.schema_loader import HedSchemaLoader
from src.validation.hed_lsp import is_hed_lsp_available, suggest_tags_for_keywords
from src.validation.hed_validator import (
    HedJavaScriptValidator,
    HedPythonValidator,
    ValidationResult,
)

logger = logging.getLogger(__name__)


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
    Uses hed-lsp CLI for suggesting valid tag alternatives when available.
    """

    # Error codes that indicate invalid or extended tags
    TAG_ERROR_CODES = [
        "TAG_INVALID",
        "TAG_EXTENSION_INVALID",
        "TAG_NOT_UNIQUE",
        "TAG_REQUIRES_CHILD",
        "TAG_NAMESPACE_PREFIX_INVALID",
    ]

    def __init__(
        self,
        schema_loader: HedSchemaLoader,
        use_javascript: bool = True,
        validator_path: Path | None = None,
        tests_json_path: Path | str | None = None,
        use_hed_lsp: bool = True,
    ) -> None:
        """Initialize the validation agent.

        Args:
            schema_loader: HED schema loader
            use_javascript: Whether to use JavaScript validator (more detailed)
            validator_path: Path to hed-javascript repository (required if use_javascript=True)
            tests_json_path: Optional path to javascriptTests.json for error remediation
            use_hed_lsp: Whether to use hed-lsp for tag suggestions (auto-detected)
        """
        self.schema_loader = schema_loader
        self.use_javascript = use_javascript
        self.error_remediator = get_remediator(tests_json_path)
        self.use_hed_lsp = use_hed_lsp and is_hed_lsp_available()

        # Declare validators with proper types
        self.js_validator: HedJavaScriptValidator | None = None
        self.py_validator: HedPythonValidator | None = None

        if use_javascript:
            if validator_path is None:
                raise ValueError("validator_path required when use_javascript=True")
            self.js_validator = HedJavaScriptValidator(validator_path)

    def _extract_problematic_tags(self, errors: list, warnings: list) -> list[str]:
        """Extract problematic tag names from validation errors and warnings.

        Args:
            errors: List of ValidationIssue errors
            warnings: List of ValidationIssue warnings

        Returns:
            List of problematic tag names
        """
        problematic_tags = []

        for issue in errors + warnings:
            # Check if this is a tag-related error
            if issue.code in self.TAG_ERROR_CODES:
                if issue.tag:
                    # Extract just the tag name (last part of path)
                    tag_name = issue.tag.split("/")[-1] if "/" in issue.tag else issue.tag
                    # Clean up the tag name (remove value placeholders like #)
                    tag_name = tag_name.split("#")[0].strip()
                    if tag_name and tag_name not in problematic_tags:
                        problematic_tags.append(tag_name)

        return problematic_tags

    def _get_tag_suggestions(
        self, problematic_tags: list[str], schema_version: str
    ) -> dict[str, list[str]]:
        """Get suggested valid tags for problematic tags using hed-lsp.

        Args:
            problematic_tags: List of problematic tag names
            schema_version: HED schema version

        Returns:
            Dictionary mapping problematic tags to suggested alternatives
        """
        if not self.use_hed_lsp or not problematic_tags:
            return {}

        try:
            return suggest_tags_for_keywords(
                problematic_tags,
                schema_version=schema_version,
                max_results=5,  # Limit suggestions for clarity
            )
        except Exception as e:
            logger.warning(
                "Failed to get tag suggestions from hed-lsp for tags %s: %s",
                problematic_tags,
                e,
            )
            return {}

    def _format_suggestions_for_feedback(self, suggestions: dict[str, list[str]]) -> str:
        """Format tag suggestions as feedback for the LLM.

        Args:
            suggestions: Dictionary mapping problematic tags to suggested alternatives

        Returns:
            Formatted suggestion string for LLM feedback
        """
        if not suggestions:
            return ""

        lines = ["\n--- Suggested valid HED tags for problematic terms ---"]
        for tag, suggested in suggestions.items():
            if suggested:
                lines.append(f"  '{tag}' → consider: {', '.join(suggested[:3])}")
            else:
                lines.append(f"  '{tag}' → no direct match found, check HED schema vocabulary")
        lines.append("---")

        return "\n".join(lines)

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
            # Detect extensions directly from the HedString parsing
            # This is more reliable than validator warnings which may be suppressed
            extended_tags = self._detect_extensions_from_hedstring(annotation, schema_version)

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

        # Extract problematic tags and get suggestions from hed-lsp
        if not result.is_valid and self.use_hed_lsp:
            problematic_tags = self._extract_problematic_tags(result.errors, result.warnings)
            if problematic_tags:
                suggestions = self._get_tag_suggestions(problematic_tags, schema_version)
                suggestion_feedback = self._format_suggestions_for_feedback(suggestions)
                if suggestion_feedback:
                    # Append suggestions to augmented errors for LLM feedback
                    if augmented_errors:
                        augmented_errors[-1] += suggestion_feedback
                    else:
                        augmented_errors = [suggestion_feedback]

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

    def _detect_extensions_from_hedstring(self, annotation: str, schema_version: str) -> list[str]:
        """Detect extended tags directly from HedString parsing.

        This is more reliable than validator warnings which may be suppressed
        when there are other errors (like TAG_INVALID).

        Falls back to regex-based detection if HedString parsing fails
        (e.g., due to unbalanced parentheses).

        Args:
            annotation: HED annotation string
            schema_version: Schema version to check against

        Returns:
            List of extended tag strings (e.g., ['Animal/Marmoset'])
        """
        from hed import HedString
        from hed.schema import load_schema_version

        extended_tags = []

        try:
            schema = load_schema_version(schema_version)
            hed_string = HedString(annotation, schema)

            for tag in hed_string.get_all_tags():
                # Check if this tag has an extension
                if tag.extension:
                    extended_tags.append(str(tag))

            # If HedString returned no tags but annotation has slashes,
            # parsing may have silently failed (e.g., unbalanced parens)
            if not extended_tags and "/" in annotation:
                extended_tags = self._detect_extensions_via_regex(annotation, schema)

        except Exception as e:
            # If parsing fails, try regex-based detection as fallback
            logger.debug("HedString parsing failed, falling back to regex: %s", e)
            try:
                schema = load_schema_version(schema_version)
                extended_tags = self._detect_extensions_via_regex(annotation, schema)
            except Exception as e2:
                logger.debug("Regex fallback also failed: %s", e2)

        return extended_tags

    def _detect_extensions_via_regex(self, annotation: str, schema: object) -> list[str]:
        """Detect extended tags using regex when HedString parsing fails.

        Finds patterns like "Parent/Extension" and checks if Parent is a valid
        base tag that can be extended.

        Args:
            annotation: HED annotation string
            schema: Loaded HED schema object

        Returns:
            List of extended tag strings
        """
        extended_tags = []

        # Find all potential extended tags (word/word patterns)
        # Match: word characters, then /, then word characters
        pattern = r"\b([A-Z][a-zA-Z-]*(?:/[A-Za-z][a-zA-Z-]*)+)"
        matches = re.findall(pattern, annotation)

        for match in matches:
            # Split into base and extension parts
            parts = match.split("/")
            base_tag = parts[0]

            # Check if base_tag is a valid HED tag that allows extensions
            # Get all tags from schema and check if base_tag is extendable
            try:
                tag_entry = schema.get_tag_entry(base_tag)  # type: ignore[attr-defined]
                if tag_entry and tag_entry.has_attribute("extensionAllowed"):
                    extended_tags.append(match)
            except Exception as e:
                # If we can't check the schema, log it but continue
                logger.debug("Could not check if '%s' allows extensions: %s", base_tag, e)

        return extended_tags
