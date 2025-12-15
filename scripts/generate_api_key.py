#!/usr/bin/env python3
"""Generate secure API key for HEDit authentication.

This script generates a cryptographically secure 64-character hexadecimal
API key suitable for use with the HEDit API authentication system.

Usage:
    python scripts/generate_api_key.py

The generated key should be added to your .env file as:
    API_KEYS=<generated_key>

Or as individual keys:
    API_KEY_1=<generated_key>
"""

import secrets


def generate_api_key() -> str:
    """Generate a secure 64-character hexadecimal API key.

    Uses secrets.token_hex(32) to generate a cryptographically strong
    random token suitable for API authentication.

    Returns:
        str: A 64-character hexadecimal string
    """
    return secrets.token_hex(32)


if __name__ == "__main__":
    api_key = generate_api_key()

    print("=" * 70)
    print("HEDit API Key Generated")
    print("=" * 70)
    print(f"\nAPI Key: {api_key}")
    print(f"\nLength: {len(api_key)} characters")
    print("\n" + "-" * 70)
    print("Add to your .env file:")
    print("-" * 70)
    print("\n# Option 1: Comma-separated list (recommended)")
    print(f"API_KEYS={api_key}")
    print("\n# Option 2: Individual key")
    print(f"API_KEY_1={api_key}")
    print("\n" + "-" * 70)
    print("SECURITY REMINDERS:")
    print("-" * 70)
    print("1. Never commit this key to Git")
    print("2. Store in .env file (already in .gitignore)")
    print("3. Set file permissions: chmod 600 .env")
    print("4. Rotate keys quarterly for production use")
    print("5. Use different keys for dev/staging/production")
    print("=" * 70)
