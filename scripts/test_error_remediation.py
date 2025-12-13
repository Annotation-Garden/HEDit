#!/usr/bin/env python3
"""Quick test script for error remediation functionality.

Run this to verify the error remediation module works correctly
without needing the full backend setup.

Usage:
    python scripts/test_error_remediation.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.error_remediation import ErrorRemediator


def test_error_remediation():
    """Test the error remediation module."""
    print("=" * 60)
    print("Testing Error Remediation Module")
    print("=" * 60)

    remediator = ErrorRemediator()

    # Test cases: simulated validation errors with diverse examples
    test_errors = [
        "[TAG_EXTENDED] Tag 'Item/Cottage' is an extension",
        "[TAG_EXTENSION_INVALID] Extension term 'Red' already exists in schema",
        "[TAG_INVALID] Tag 'Fake-tag' not found in schema",
        "[DEFINITION_INVALID] Definition not in top-level tag group",
        "[PARENTHESES_MISMATCH] Unbalanced parentheses",
    ]

    test_warnings = [
        "[TAG_EXTENDED] Tag 'Move-body/Cartwheel' is an extension from schema",
    ]

    print("\n--- Original Errors ---")
    for err in test_errors:
        print(f"  {err}")

    print("\n--- Augmented Errors with Remediation ---")
    aug_errors, aug_warnings = remediator.augment_validation_errors(test_errors, test_warnings)

    for i, aug_err in enumerate(aug_errors):
        print(f"\n[Error {i + 1}]")
        print(aug_err)
        print("-" * 40)

    print("\n--- Augmented Warnings ---")
    for i, aug_warn in enumerate(aug_warnings):
        print(f"\n[Warning {i + 1}]")
        print(aug_warn)
        print("-" * 40)

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


def test_specific_guidance():
    """Test specific remediation guidance content."""
    print("\n" + "=" * 60)
    print("Testing Specific Guidance Content")
    print("=" * 60)

    remediator = ErrorRemediator()

    # Test TAG_EXTENDED guidance includes key information
    guidance = remediator.get_remediation("TAG_EXTENDED")
    print("\n--- TAG_EXTENDED Guidance ---")
    print(guidance)

    # Verify key content - check for diverse examples across schema trees
    checks = [
        ("MOST SPECIFIC", "Contains guidance about most specific parent"),
        ("is-a", "Mentions is-a relationship"),
        ("Building/Cottage", "Contains building example (Item tree)"),
        ("Move-body/Cartwheel", "Contains action example (Action tree)"),
        ("Furniture/Armoire", "Contains furniture example (Item tree)"),
        ("Vehicle/Rickshaw", "Contains vehicle example (Item tree)"),
        ("Mammal/Dolphin", "Contains animal example (Agent tree)"),
    ]

    print("\n--- Content Checks ---")
    for term, description in checks:
        if term.lower() in guidance.lower():
            print(f"  [PASS] {description}")
        else:
            print(f"  [FAIL] {description} - missing '{term}'")


if __name__ == "__main__":
    test_error_remediation()
    test_specific_guidance()
