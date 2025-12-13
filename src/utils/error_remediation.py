"""Error remediation module for HED validation feedback.

This module provides remediation guidance for HED validation errors and warnings
using examples from the HED specification test suite (javascriptTests.json).
"""

import json
from pathlib import Path


class ErrorRemediator:
    """Provides remediation guidance for HED validation errors and warnings.

    Uses the HED specification test suite (javascriptTests.json) to provide
    context-aware remediation suggestions with examples of correct usage.
    """

    # Mapping of error codes to human-readable remediation guidance
    # These are manually curated based on the error descriptions and common issues
    REMEDIATION_GUIDANCE = {
        "TAG_EXTENDED": {
            "description": "A tag was extended from the schema (this is a warning, not an error).",
            "guidance": """IMPORTANT: When extending tags, extend from the MOST SPECIFIC applicable parent tag.

The extension should preserve the taxonomic (is-a) relationship. Find the most specific
existing tag where your new concept IS-A type of that tag.

EXAMPLES FROM DIFFERENT SCHEMA AREAS:

1. BUILDINGS (Item/Object/Building tree):
   BAD:  Item/Cottage, Object/Cottage
   GOOD: Building/Cottage (a cottage IS-A building)

2. ACTIONS (Action/Move tree):
   BAD:  Action/Cartwheel, Move/Cartwheel
   GOOD: Move-body/Cartwheel (a cartwheel IS-A body movement)

3. FURNITURE (Item/Object/Furnishing/Furniture tree):
   BAD:  Item/Armoire, Furnishing/Armoire
   GOOD: Furniture/Armoire (an armoire IS-A furniture)

4. VEHICLES (Item/Object/Vehicle tree):
   BAD:  Item/Rickshaw, Object/Rickshaw
   GOOD: Vehicle/Rickshaw (a rickshaw IS-A vehicle)

5. ANIMALS (Agent/Animal tree):
   BAD:  Agent/Dolphin, Animal/Dolphin
   GOOD: Mammal/Dolphin (a dolphin IS-A mammal, which is more specific than Animal)

Before extending, check if the tag already exists in the schema vocabulary.
If it exists, use it directly without any parent path.""",
            "examples": {
                "wrong": [
                    "Item/Cottage",
                    "Action/Cartwheel",
                    "Item/Armoire",
                    "Object/Rickshaw",
                    "Agent/Dolphin",
                ],
                "correct": [
                    "Building/Cottage",
                    "Move-body/Cartwheel",
                    "Furniture/Armoire",
                    "Vehicle/Rickshaw",
                    "Mammal/Dolphin",
                ],
            },
        },
        "TAG_EXTENSION_INVALID": {
            "description": "The tag extension is invalid.",
            "guidance": """Extension errors occur when:
1. The extension term already exists in the schema (use the tag directly)
2. The extension term has invalid characters (use only alphanumeric and hyphens)
3. The parent tag doesn't allow extension

FIX: Check if the tag exists in the vocabulary. If it does, use it AS-IS.
Example: Use 'Red' instead of 'Property/Red' if 'Red' is in the vocabulary.""",
            "examples": {
                "wrong": ["Property/Red", "Item/Circle", "Sensory-presentation/Red"],
                "correct": ["Red", "Circle", "Red-color/Red/Redish"],
            },
        },
        "TAG_INVALID": {
            "description": "The tag is not valid in the schema.",
            "guidance": """The tag doesn't exist in the HED schema vocabulary.

FIX:
1. Check spelling (use exact capitalization and hyphenation)
2. Check if you're using the short-form (e.g., 'Red' not 'Property/Sensory-property/Red')
3. If the concept doesn't exist, extend from an appropriate parent tag""",
            "examples": {
                "wrong": ["Redd", "red", "RED", "Action/Press/Hard"],
                "correct": ["Red", "Press", "Press/Hard"],
            },
        },
        "TAG_REQUIRES_CHILD": {
            "description": "A tag requires a child value but none was provided.",
            "guidance": """This tag requires a child value (indicated by # in schema).

FIX: Add a value after the tag.
Example: Use 'Duration/2 s' instead of just 'Duration'
Example: Use 'Frequency/440 Hz' instead of just 'Frequency'""",
            "examples": {
                "wrong": ["Duration", "Frequency", "Age"],
                "correct": ["Duration/2 s", "Frequency/440 Hz", "Age/25 years"],
            },
        },
        "DEFINITION_INVALID": {
            "description": "The Definition tag usage is invalid.",
            "guidance": """Definition rules:
1. Must be at top level in a tag group: (Definition/Name, (content))
2. Inner group contains the definition content
3. Cannot contain Def, Def-expand, or nested Definition tags
4. If using placeholder (#), must have exactly two # characters

Pattern: (Definition/MyDef, (Tag1, Tag2, Tag3))
With placeholder: (Definition/MyDef/#, (Tag1/# units, Tag2))""",
            "examples": {
                "wrong": ["Definition/MyDef", "(Definition/MyDef)", "Definition/MyDef, Red"],
                "correct": [
                    "(Definition/MyDef, (Red, Circle))",
                    "(Definition/Acc/#, (Acceleration/# m-per-s^2, Red))",
                ],
            },
        },
        "DEF_INVALID": {
            "description": "The Def tag usage is invalid.",
            "guidance": """Def rules:
1. Must reference an existing Definition
2. If definition has placeholder, Def must provide value: Def/MyDef/value
3. Value must match the expected unit class

Pattern: Def/DefinitionName or Def/DefinitionName/value""",
            "examples": {
                "wrong": ["Def/NonExistentDef", "Def/Acc", "Def/Acc/invalid_units"],
                "correct": ["Def/MyColor", "Def/Acc/4.5"],
            },
        },
        "PARENTHESES_MISMATCH": {
            "description": "Parentheses are not properly balanced or nested.",
            "guidance": """Parentheses rules:
1. Every open parenthesis '(' must have a matching close ')'
2. Parentheses must be properly nested (no interleaving)

FIX: Count your parentheses and check nesting.""",
            "examples": {
                "wrong": ["(Red, Circle", "(Red, (Circle, Blue)"],
                "correct": ["(Red, Circle)", "(Red, (Circle, Blue))"],
            },
        },
        "COMMA_MISSING": {
            "description": "A comma is missing between tags or tag groups.",
            "guidance": """Commas separate tags and tag groups.

Pattern: Tag1, Tag2, (Group1), (Group2)
NOT: Tag1 Tag2(Group1)(Group2)""",
            "examples": {
                "wrong": ["Red Blue", "(A, B)(C, D)", "A(B, D)"],
                "correct": ["Red, Blue", "(A, B), (C, D)", "A, (B, D)"],
            },
        },
        "TAG_EMPTY": {
            "description": "Empty tag or extra commas/parentheses found.",
            "guidance": """Remove empty tags, extra commas, and empty parentheses.

BAD: Red,, Blue or (Red, ), or ()
GOOD: Red, Blue or (Red)""",
            "examples": {
                "wrong": ["Red,, Blue", "(Red, )", "()", ", Red"],
                "correct": ["Red, Blue", "(Red)", "Red"],
            },
        },
        "VALUE_INVALID": {
            "description": "The value provided is invalid.",
            "guidance": """Value format requirements:
1. Numeric values: proper number format
2. Units: space between value and unit (e.g., '2 s' not '2s')
3. Value class must match schema expectations

Pattern: Value/number unit (with space before unit)""",
            "examples": {
                "wrong": ["Duration/2s", "Frequency/440hz", "Distance/5meters"],
                "correct": ["Duration/2 s", "Frequency/440 Hz", "Distance/5 m"],
            },
        },
        "UNITS_INVALID": {
            "description": "The units are invalid for this tag.",
            "guidance": """Check the schema for valid units for this tag.

Common unit classes:
- Time: s, ms, minute, hour, day
- Frequency: Hz, kHz, MHz
- Distance: m, cm, mm, km
- Angle: degree, radian""",
            "examples": {
                "wrong": ["Duration/2 meters", "Frequency/440 s"],
                "correct": ["Duration/2 s", "Frequency/440 Hz"],
            },
        },
        "PLACEHOLDER_INVALID": {
            "description": "Invalid placeholder (#) usage.",
            "guidance": """Placeholder (#) rules:
1. In definitions: must have exactly two # characters
2. In sidecars: one # per HED string for value columns
3. Cannot appear in event file HED column directly
4. Parent tag must accept placeholder values""",
            "examples": {
                "wrong": ["Age/#", "Duration/# # s"],
                "correct": ["Age/# years", "(Definition/Acc/#, (Acceleration/# m-per-s^2))"],
            },
        },
        "SIDECAR_BRACES_INVALID": {
            "description": "Invalid curly brace usage in sidecar.",
            "guidance": """Curly brace rules (sidecars only):
1. {column_name} references another column's HED annotation
2. Column must exist and have HED annotations
3. No circular references (A references B, B references A)
4. Use for grouping across columns: ({color}, {shape})""",
            "examples": {
                "wrong": ["{nonexistent_column}", "{HED}", "Value/{column}"],
                "correct": ["Experimental-stimulus, ({color}, {shape})", "{response}"],
            },
        },
        "TEMPORAL_TAG_ERROR": {
            "description": "Temporal tag (Onset/Offset/Inset/Duration/Delay) usage error.",
            "guidance": """Temporal tag rules:
1. Onset/Offset must be in top-level tag groups
2. Must have exactly one Def anchor: (Def/Name, Onset)
3. Offset must follow a matching Onset
4. Duration/Delay for non-anchored temporal events

Patterns:
- (Def/Event, Onset) ... (Def/Event, Offset)
- (Duration/2 s, (Sensory-event, Visual-presentation, Red))""",
            "examples": {
                "wrong": ["Onset", "Def/Event, Onset", "(Def/Event, Onset, Offset)"],
                "correct": [
                    "(Def/Event, Onset)",
                    "(Def/Event, Offset)",
                    "(Duration/2 s, (Red, Circle))",
                ],
            },
        },
        "CHARACTER_INVALID": {
            "description": "Invalid character in tag or value.",
            "guidance": """Character rules:
1. Tags: alphanumeric, hyphens, underscores only
2. No special characters ($, %, @, etc.)
3. Curly braces {} only in sidecars for column references
4. No non-printing/control characters""",
            "examples": {
                "wrong": ["Red$color", "Tag@special", "{column} outside sidecar"],
                "correct": ["Red-color", "Special-tag", "Column-reference"],
            },
        },
    }

    def __init__(self, tests_json_path: Path | str | None = None) -> None:
        """Initialize the error remediator.

        Args:
            tests_json_path: Path to javascriptTests.json. If None, uses default location.
        """
        self.tests_data: list[dict] = []
        if tests_json_path:
            self._load_tests(Path(tests_json_path))

    def _load_tests(self, path: Path) -> None:
        """Load the test data from javascriptTests.json.

        Args:
            path: Path to the JSON file
        """
        if path.exists():
            with open(path) as f:
                self.tests_data = json.load(f)

    def get_remediation(self, error_code: str, error_message: str = "") -> str:
        """Get remediation guidance for an error code.

        Args:
            error_code: The HED error code (e.g., "TAG_EXTENDED", "TAG_INVALID")
            error_message: The original error message for context

        Returns:
            Formatted remediation guidance string
        """
        # Check if we have guidance for this error code
        if error_code in self.REMEDIATION_GUIDANCE:
            guidance_data = self.REMEDIATION_GUIDANCE[error_code]

            parts = [
                f"\n📋 REMEDIATION for {error_code}:",
                f"   {guidance_data['description']}",
                "",
                "   HOW TO FIX:",
                f"   {guidance_data['guidance']}",
            ]

            if "examples" in guidance_data:
                examples = guidance_data["examples"]
                if examples.get("wrong"):
                    parts.append("")
                    parts.append("   ❌ WRONG:")
                    for ex in examples["wrong"][:3]:
                        parts.append(f"      {ex}")
                if examples.get("correct"):
                    parts.append("   ✓ CORRECT:")
                    for ex in examples["correct"][:3]:
                        parts.append(f"      {ex}")

            return "\n".join(parts)

        # Fallback: try to find in loaded test data
        if self.tests_data:
            for test_entry in self.tests_data:
                if test_entry.get("error_code") == error_code:
                    return self._format_test_entry(test_entry)

        return f"\n📋 No specific remediation guidance available for {error_code}."

    def _format_test_entry(self, test_entry: dict) -> str:
        """Format a test entry as remediation guidance.

        Args:
            test_entry: Entry from javascriptTests.json

        Returns:
            Formatted guidance string
        """
        parts = [
            f"\n📋 REMEDIATION for {test_entry['error_code']}:",
            f"   {test_entry.get('description', 'No description available')}",
        ]

        # Get passing examples from string_tests
        string_tests = test_entry.get("tests", {}).get("string_tests", {})
        passes = string_tests.get("passes", [])
        fails = string_tests.get("fails", [])

        if fails:
            parts.append("")
            parts.append("   ❌ EXAMPLES THAT FAIL:")
            for ex in fails[:2]:
                if isinstance(ex, str):
                    parts.append(f"      {ex}")

        if passes:
            parts.append("   ✓ EXAMPLES THAT PASS:")
            for ex in passes[:2]:
                if isinstance(ex, str):
                    parts.append(f"      {ex}")

        return "\n".join(parts)

    def augment_validation_errors(
        self,
        errors: list[str],
        warnings: list[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Augment validation errors and warnings with remediation guidance.

        Args:
            errors: List of error messages
            warnings: List of warning messages

        Returns:
            Tuple of (augmented_errors, augmented_warnings)
        """
        augmented_errors = []
        augmented_warnings = []

        for error in errors:
            # Extract error code from message (assumes format "[CODE] message")
            code = self._extract_error_code(error)
            if code:
                remediation = self.get_remediation(code, error)
                augmented_errors.append(f"{error}{remediation}")
            else:
                augmented_errors.append(error)

        if warnings:
            for warning in warnings:
                code = self._extract_error_code(warning)
                if code:
                    remediation = self.get_remediation(code, warning)
                    augmented_warnings.append(f"{warning}{remediation}")
                else:
                    augmented_warnings.append(warning)

        return augmented_errors, augmented_warnings or []

    def _extract_error_code(self, message: str) -> str | None:
        """Extract error code from a validation message.

        Args:
            message: Validation error/warning message

        Returns:
            Extracted error code or None
        """
        # Format: "[CODE] message"
        if message.startswith("[") and "]" in message:
            return message[1 : message.index("]")]
        return None


# Default remediator instance
_default_remediator: ErrorRemediator | None = None


def get_remediator(tests_json_path: Path | str | None = None) -> ErrorRemediator:
    """Get the default error remediator instance.

    Args:
        tests_json_path: Optional path to javascriptTests.json

    Returns:
        ErrorRemediator instance
    """
    global _default_remediator
    if _default_remediator is None:
        _default_remediator = ErrorRemediator(tests_json_path)
    return _default_remediator
