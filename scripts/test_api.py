#!/usr/bin/env python3
"""Quick API test script for local development.

Tests the annotation endpoint to verify the full workflow including
error remediation feedback.

Usage:
    # With running local server
    python scripts/test_api.py

    # With custom endpoint
    python scripts/test_api.py --url http://localhost:38427

    # With API key
    python scripts/test_api.py --api-key your-key
"""

import argparse
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def test_health(base_url: str) -> bool:
    """Test health endpoint."""
    print("\n1. Testing /health endpoint...")
    try:
        req = Request(f"{base_url}/health")
        with urlopen(req, timeout=10) as response:
            data = json.loads(response.read())
            print(f"   Status: {data.get('status')}")
            print(f"   Version: {data.get('version')}")
            print(f"   LLM Available: {data.get('llm_available')}")
            print(f"   Validator Available: {data.get('validator_available')}")
            return data.get("status") == "healthy"
    except (HTTPError, URLError) as e:
        print(f"   Error: {e}")
        return False


def test_annotation(base_url: str, api_key: str | None, description: str) -> dict | None:
    """Test annotation endpoint."""
    print("\n2. Testing /annotate endpoint...")
    print(f"   Description: {description}")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    payload = json.dumps(
        {
            "description": description,
            "schema_version": "8.3.0",
            "max_validation_attempts": 3,
            "run_assessment": True,
        }
    ).encode()

    try:
        req = Request(f"{base_url}/annotate", data=payload, headers=headers, method="POST")
        with urlopen(req, timeout=120) as response:
            return json.loads(response.read())
    except HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        print(f"   HTTP Error {e.code}: {e.reason}")
        if error_body:
            print(f"   Details: {error_body}")
        return None
    except URLError as e:
        print(f"   Connection Error: {e.reason}")
        return None


def display_result(result: dict):
    """Display annotation result with formatting."""
    print("\n" + "=" * 60)
    print("ANNOTATION RESULT")
    print("=" * 60)

    print(f"\nAnnotation: {result.get('annotation')}")
    print(f"Status: {result.get('status')}")
    print(f"Valid: {result.get('is_valid')}")
    print(f"Faithful: {result.get('is_faithful')}")
    print(f"Complete: {result.get('is_complete')}")
    print(f"Validation Attempts: {result.get('validation_attempts')}")

    errors = result.get("validation_errors", [])
    if errors:
        print(f"\n--- Validation Errors ({len(errors)}) ---")
        for err in errors:
            # Truncate long remediation messages for readability
            if len(err) > 200:
                print(f"\n{err[:200]}...")
                print("   [truncated - full message contains remediation guidance]")
            else:
                print(f"\n{err}")

    warnings = result.get("validation_warnings", [])
    if warnings:
        print(f"\n--- Validation Warnings ({len(warnings)}) ---")
        for warn in warnings:
            if len(warn) > 200:
                print(f"\n{warn[:200]}...")
            else:
                print(f"\n{warn}")

    if result.get("evaluation_feedback"):
        print("\n--- Evaluation Feedback ---")
        print(
            result["evaluation_feedback"][:500] + "..."
            if len(result["evaluation_feedback"]) > 500
            else result["evaluation_feedback"]
        )

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test HEDit API")
    parser.add_argument("--url", default="http://localhost:38427", help="API base URL")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument(
        "--description",
        default="A red house appears on the left side of the screen",
        help="Event description to annotate",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("HEDit API Test")
    print("=" * 60)
    print(f"Endpoint: {args.url}")

    # Test health
    if not test_health(args.url):
        print("\nHealth check failed. Is the server running?")
        print("Try: ./scripts/dev.sh")
        sys.exit(1)

    print("   Health check passed!")

    # Test annotation
    result = test_annotation(args.url, args.api_key, args.description)

    if result:
        display_result(result)

        # Check for error remediation (the feature we're testing)
        errors = result.get("validation_errors", [])
        warnings = result.get("validation_warnings", [])

        has_remediation = any("REMEDIATION" in str(e) for e in errors + warnings)

        print("\n--- Feature Check ---")
        if has_remediation:
            print("[PASS] Error remediation feedback is present!")
        else:
            print("[INFO] No remediation feedback (annotation may be fully valid)")

        sys.exit(0 if result.get("status") == "success" else 1)
    else:
        print("\nAnnotation request failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
