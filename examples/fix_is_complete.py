#!/usr/bin/env python3
"""Fix is_complete flag in benchmark results.

The assessment agent output format is:
    COMPLETENESS: [complete/incomplete]
    ...
    STATUS: [COMPLETE/NEEDS-REVIEW]

But the parsing code was looking for "FINAL STATUS: COMPLETE".
This script fixes the is_complete flag based on the actual assessment_feedback text.
"""

import json
from pathlib import Path


def fix_is_complete(feedback: str) -> bool:
    """Parse is_complete from assessment feedback text."""
    if not feedback:
        return False
    feedback_upper = feedback.upper()
    return "COMPLETENESS: COMPLETE" in feedback_upper or "STATUS: COMPLETE" in feedback_upper


def fix_result_file(filepath: Path) -> tuple[int, int]:
    """Fix is_complete flags in a single result file.

    Returns:
        Tuple of (total_results, fixed_count)
    """
    with open(filepath) as f:
        data = json.load(f)

    total = 0
    fixed = 0

    for result in data.get("results", []):
        response = result.get("full_response", {})
        metadata = response.get("metadata", {})

        if "assessment_feedback" in metadata:
            total += 1
            feedback = metadata["assessment_feedback"]
            correct_value = fix_is_complete(feedback)

            if metadata.get("is_complete") != correct_value:
                metadata["is_complete"] = correct_value
                fixed += 1

    # Write back
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return total, fixed


def main():
    results_dir = Path(__file__).parent / "benchmark_results"

    total_files = 0
    total_results = 0
    total_fixed = 0

    # Process all JSON files
    for filepath in results_dir.glob("*.json"):
        if "summary" in filepath.name or "report" in filepath.name:
            continue

        try:
            results, fixed = fix_result_file(filepath)
            if results > 0:
                total_files += 1
                total_results += results
                total_fixed += fixed
                if fixed > 0:
                    print(f"Fixed {fixed}/{results} in {filepath.name}")
        except Exception as e:
            print(f"Error processing {filepath.name}: {e}")

    print("\nSummary:")
    print(f"  Files processed: {total_files}")
    print(f"  Total results: {total_results}")
    print(f"  Fixed: {total_fixed}")


if __name__ == "__main__":
    main()
