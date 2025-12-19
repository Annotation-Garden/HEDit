"""HED validation using both Python and JavaScript validators.

This module provides integration with HED validation tools, primarily using
the JavaScript validator for comprehensive feedback, with Python fallback.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from hed import HedString
from hed.errors import get_printable_issue_string
from hed.schema import HedSchema, load_schema_version
from hed.validator import HedValidator

if TYPE_CHECKING:
    pass


def is_js_validator_available(validator_path: Path | str | None = None) -> bool:
    """Check if JavaScript validator is available.

    Args:
        validator_path: Path to hed-javascript. If None, uses HED_VALIDATOR_PATH env var.

    Returns:
        True if Node.js is installed and hed-javascript is available.
    """
    # Check Node.js
    if not shutil.which("node"):
        return False

    # Check validator path
    if validator_path is None:
        validator_path = os.environ.get("HED_VALIDATOR_PATH")
    if validator_path is None:
        return False

    path = Path(validator_path)
    return path.exists() and (path / "dist" / "commonjs" / "index.js").exists()


def get_validator(
    schema_version: str = "8.3.0",
    prefer_js: bool = True,
    require_js: bool = False,
    validator_path: Path | str | None = None,
) -> HedPythonValidator | HedJavaScriptValidator:
    """Get the appropriate HED validator based on availability and preferences.

    Args:
        schema_version: HED schema version (e.g., "8.3.0", "8.4.0")
        prefer_js: If True, prefer JavaScript validator when available
        require_js: If True, raise error if JavaScript validator unavailable (no fallback)
        validator_path: Path to hed-javascript. If None, uses HED_VALIDATOR_PATH env var.

    Returns:
        Configured validator instance (JavaScript or Python)

    Raises:
        RuntimeError: If require_js=True and JavaScript validator is unavailable
    """
    # Resolve validator path
    if validator_path is None:
        validator_path = os.environ.get("HED_VALIDATOR_PATH")

    js_available = is_js_validator_available(validator_path)

    if require_js and not js_available:
        raise RuntimeError(
            "JavaScript validator required but unavailable. "
            "Ensure Node.js is installed and HED_VALIDATOR_PATH is set."
        )

    if prefer_js and js_available and validator_path:
        return HedJavaScriptValidator(
            validator_path=Path(validator_path),
            schema_version=schema_version,
        )

    # Fall back to Python validator
    schema = load_schema_version(schema_version)
    return HedPythonValidator(schema=schema)


@dataclass
class ValidationIssue:
    """Represents a single validation issue (error or warning).

    Attributes:
        code: Issue code (e.g., 'TAG_INVALID')
        level: Severity level ('error' or 'warning')
        message: Human-readable error message
        tag: The problematic tag (if applicable)
        context: Additional context information
    """

    code: str
    level: Literal["error", "warning"]
    message: str
    tag: str | None = None
    context: dict | None = None


@dataclass
class ValidationResult:
    """Result of HED string validation.

    Attributes:
        is_valid: Whether the HED string is valid
        errors: List of error issues
        warnings: List of warning issues
        parsed_string: Successfully parsed HED string (if valid)
    """

    is_valid: bool
    errors: list[ValidationIssue]
    warnings: list[ValidationIssue]
    parsed_string: str | None = None


class HedPythonValidator:
    """Validates HED strings using the Python HED tools."""

    def __init__(self, schema: HedSchema) -> None:
        """Initialize validator with a HED schema.

        Args:
            schema: HedSchema object to validate against
        """
        self.schema = schema
        self.validator = HedValidator(schema)

    def validate(self, hed_string: str) -> ValidationResult:
        """Validate a HED string.

        Args:
            hed_string: HED annotation string to validate

        Returns:
            ValidationResult with errors and warnings
        """
        errors = []
        warnings = []

        try:
            # Parse and validate HED string
            hed_string_obj = HedString(hed_string, self.schema)
            issues = hed_string_obj.validate(self.validator)

            # Process issues
            for issue in issues:
                issue_str = get_printable_issue_string([issue])
                severity = "error" if issue["severity"] == "error" else "warning"

                validation_issue = ValidationIssue(
                    code=issue.get("code", "UNKNOWN"),
                    level=severity,
                    message=issue_str,
                    tag=issue.get("tag", None),
                )

                if severity == "error":
                    errors.append(validation_issue)
                else:
                    warnings.append(validation_issue)

            is_valid = len(errors) == 0
            parsed = str(hed_string_obj) if is_valid else None

            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                parsed_string=parsed,
            )

        except Exception as e:
            # Catch parsing errors
            errors.append(
                ValidationIssue(
                    code="PARSE_ERROR",
                    level="error",
                    message=str(e),
                )
            )
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)


class HedJavaScriptValidator:
    """Validates HED strings using the JavaScript HED validator.

    This provides more detailed feedback than the Python validator.
    Requires Node.js and the hed-javascript package.
    """

    def __init__(
        self,
        validator_path: Path,
        schema_version: str = "8.3.0",
    ) -> None:
        """Initialize JavaScript validator.

        Args:
            validator_path: Path to hed-javascript repository
            schema_version: HED schema version to use
        """
        self.validator_path = Path(validator_path)
        self.schema_version = schema_version
        self._check_installation()

    def _check_installation(self) -> None:
        """Verify that Node.js and hed-validator are available."""
        # Check Node.js
        try:
            subprocess.run(
                ["node", "--version"],
                check=True,
                capture_output=True,
                timeout=5,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError("Node.js is not installed or not in PATH") from e

        # Check validator path
        if not self.validator_path.exists():
            raise RuntimeError(f"HED JavaScript validator not found at {self.validator_path}")

    def validate(self, hed_string: str) -> ValidationResult:
        """Validate a HED string using JavaScript validator.

        Args:
            hed_string: HED annotation string to validate

        Returns:
            ValidationResult with detailed errors and warnings
        """
        # Create validation script
        script = f"""
        const {{ parseHedString, buildSchemasFromVersion }} = require('{self.validator_path}/dist/commonjs/index.js');

        async function validate() {{
            try {{
                const schemas = await buildSchemasFromVersion('{self.schema_version}');
                const hedString = `{hed_string}`;
                const [parsed, errors, warnings] = parseHedString(
                    hedString,
                    schemas,
                    false,  // no definitions
                    false,  // no placeholders
                    true    // full validation
                );

                // Reclassify warnings that should actually be errors
                // Based on HED validator source: these indicate invalid/malformed HED
                const errorCodes = [
                    'TAG_INVALID',                    // Invalid tag - doesn't exist in schema
                    'TAG_NAMESPACE_PREFIX_INVALID',   // Invalid tag prefix
                    'TAG_NOT_UNIQUE',                 // Multiple unique tags
                    'TAG_REQUIRES_CHILD',             // Child/value required
                    'TAG_EXTENSION_INVALID',          // Invalid extension
                    'TAG_EMPTY',                      // Empty tag
                    'UNITS_INVALID',                  // Invalid units
                    'VALUE_INVALID',                  // Invalid value
                ];
                const actualErrors = [];
                const actualWarnings = [];

                // Process errors
                errors.forEach(e => {{
                    actualErrors.push({{
                        code: e.hedCode || e.internalCode,
                        message: e.message,
                        tag: e.parameters?.tag,
                        level: 'error'
                    }});
                }});

                // Process warnings - promote critical ones to errors
                warnings.forEach(w => {{
                    const code = w.hedCode || w.internalCode;
                    const issue = {{
                        code: code,
                        message: w.message,
                        tag: w.parameters?.tag,
                        level: errorCodes.includes(code) ? 'error' : 'warning'
                    }};

                    if (errorCodes.includes(code)) {{
                        actualErrors.push(issue);
                    }} else {{
                        actualWarnings.push(issue);
                    }}
                }});

                const result = {{
                    isValid: actualErrors.length === 0,
                    parsed: parsed ? parsed.toString() : null,
                    errors: actualErrors,
                    warnings: actualWarnings
                }};

                console.log(JSON.stringify(result));
            }} catch (error) {{
                console.log(JSON.stringify({{
                    isValid: false,
                    errors: [{{ code: 'VALIDATOR_ERROR', message: error.message, level: 'error' }}],
                    warnings: []
                }}));
            }}
        }}

        validate();
        """

        try:
            # Run Node.js validation
            result = subprocess.run(
                ["node", "-e", script],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )

            # Parse result
            output = json.loads(result.stdout)

            errors = [
                ValidationIssue(
                    code=e["code"],
                    level="error",
                    message=e["message"],
                    tag=e.get("tag"),
                )
                for e in output["errors"]
            ]

            warnings = [
                ValidationIssue(
                    code=w["code"],
                    level="warning",
                    message=w["message"],
                    tag=w.get("tag"),
                )
                for w in output["warnings"]
            ]

            return ValidationResult(
                is_valid=output["isValid"],
                errors=errors,
                warnings=warnings,
                parsed_string=output.get("parsed"),
            )

        except subprocess.TimeoutExpired:
            return ValidationResult(
                is_valid=False,
                errors=[
                    ValidationIssue(
                        code="TIMEOUT",
                        level="error",
                        message="Validation timed out",
                    )
                ],
                warnings=[],
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[
                    ValidationIssue(
                        code="VALIDATION_ERROR",
                        level="error",
                        message=f"Validation failed: {e}",
                    )
                ],
                warnings=[],
            )
