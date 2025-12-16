"""Tests for API security middleware and utilities."""

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from src.api.security import (
    APIKeyAuth,
    AuditLogger,
    generate_api_key,
    verify_origin,
)


class TestAPIKeyAuth:
    """Tests for API key authentication."""

    def test_init_loads_keys_from_api_keys_env(self, monkeypatch):
        """Test loading API keys from API_KEYS environment variable."""
        monkeypatch.setenv("API_KEYS", "key1,key2,key3")
        monkeypatch.setenv("REQUIRE_API_AUTH", "true")

        auth = APIKeyAuth()
        assert "key1" in auth.api_keys
        assert "key2" in auth.api_keys
        assert "key3" in auth.api_keys
        assert len(auth.api_keys) == 3

    def test_init_loads_individual_keys(self, monkeypatch):
        """Test loading API keys from API_KEY_1, API_KEY_2, etc."""
        monkeypatch.setenv("API_KEY_1", "individual_key_1")
        monkeypatch.setenv("API_KEY_2", "individual_key_2")
        monkeypatch.delenv("API_KEYS", raising=False)
        monkeypatch.setenv("REQUIRE_API_AUTH", "true")

        auth = APIKeyAuth()
        assert "individual_key_1" in auth.api_keys
        assert "individual_key_2" in auth.api_keys

    def test_init_combines_all_key_sources(self, monkeypatch):
        """Test that API_KEYS and individual keys are combined."""
        monkeypatch.setenv("API_KEYS", "shared_key")
        monkeypatch.setenv("API_KEY_1", "individual_key")
        monkeypatch.setenv("REQUIRE_API_AUTH", "true")

        auth = APIKeyAuth()
        assert "shared_key" in auth.api_keys
        assert "individual_key" in auth.api_keys

    def test_init_auth_disabled(self, monkeypatch):
        """Test initialization with auth disabled."""
        monkeypatch.setenv("REQUIRE_API_AUTH", "false")
        monkeypatch.delenv("API_KEYS", raising=False)

        auth = APIKeyAuth()
        assert auth.require_auth is False

    def test_init_no_keys_configured_warning(self, monkeypatch, caplog):
        """Test warning when no keys configured but auth enabled and BYOK disabled."""
        monkeypatch.delenv("API_KEYS", raising=False)
        monkeypatch.delenv("API_KEY_1", raising=False)
        monkeypatch.setenv("REQUIRE_API_AUTH", "true")
        monkeypatch.setenv("ALLOW_BYOK", "false")  # Warning only shown when BYOK disabled

        auth = APIKeyAuth()
        assert len(auth.api_keys) == 0

    def test_verify_api_key_valid(self, monkeypatch):
        """Test verification of valid API key."""
        monkeypatch.setenv("API_KEYS", "valid_key")
        monkeypatch.setenv("REQUIRE_API_AUTH", "true")

        auth = APIKeyAuth()
        assert auth.verify_api_key("valid_key") is True

    def test_verify_api_key_invalid(self, monkeypatch):
        """Test verification of invalid API key."""
        monkeypatch.setenv("API_KEYS", "valid_key")
        monkeypatch.setenv("REQUIRE_API_AUTH", "true")

        auth = APIKeyAuth()
        assert auth.verify_api_key("invalid_key") is False

    def test_verify_api_key_none(self, monkeypatch):
        """Test verification with None key."""
        monkeypatch.setenv("API_KEYS", "valid_key")
        monkeypatch.setenv("REQUIRE_API_AUTH", "true")

        auth = APIKeyAuth()
        assert auth.verify_api_key(None) is False

    def test_verify_api_key_auth_disabled(self, monkeypatch):
        """Test verification when auth is disabled."""
        monkeypatch.setenv("REQUIRE_API_AUTH", "false")

        auth = APIKeyAuth()
        # Any key should be valid when auth is disabled
        assert auth.verify_api_key("any_key") is True
        assert auth.verify_api_key(None) is True

    @pytest.mark.asyncio
    async def test_call_auth_disabled(self, monkeypatch):
        """Test __call__ when auth is disabled."""
        monkeypatch.setenv("REQUIRE_API_AUTH", "false")

        auth = APIKeyAuth()
        result = await auth(api_key=None)
        assert result == "auth_disabled"

    @pytest.mark.asyncio
    async def test_call_no_key_raises(self, monkeypatch):
        """Test __call__ raises when no key provided."""
        monkeypatch.setenv("API_KEYS", "valid_key")
        monkeypatch.setenv("REQUIRE_API_AUTH", "true")

        auth = APIKeyAuth()
        with pytest.raises(HTTPException) as exc_info:
            await auth(api_key=None, openrouter_key=None)

        assert exc_info.value.status_code == 401
        assert "Authentication required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_call_invalid_key_raises(self, monkeypatch):
        """Test __call__ raises when invalid key provided."""
        monkeypatch.setenv("API_KEYS", "valid_key")
        monkeypatch.setenv("REQUIRE_API_AUTH", "true")

        auth = APIKeyAuth()
        with pytest.raises(HTTPException) as exc_info:
            await auth(api_key="wrong_key", openrouter_key=None)

        assert exc_info.value.status_code == 401
        assert "Invalid API key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_call_valid_key_returns_key(self, monkeypatch):
        """Test __call__ returns key when valid."""
        monkeypatch.setenv("API_KEYS", "valid_key")
        monkeypatch.setenv("REQUIRE_API_AUTH", "true")

        auth = APIKeyAuth()
        result = await auth(api_key="valid_key", openrouter_key=None)
        assert result == "valid_key"

    @pytest.mark.asyncio
    async def test_call_byok_valid_key(self, monkeypatch):
        """Test __call__ accepts valid OpenRouter key in BYOK mode."""
        monkeypatch.setenv("REQUIRE_API_AUTH", "true")
        monkeypatch.setenv("ALLOW_BYOK", "true")
        monkeypatch.delenv("API_KEYS", raising=False)

        auth = APIKeyAuth()
        result = await auth(api_key=None, openrouter_key="sk-or-v1-validkey12345678901234567890")
        assert result == "byok"

    @pytest.mark.asyncio
    async def test_call_byok_invalid_key_format(self, monkeypatch):
        """Test __call__ rejects invalid OpenRouter key format."""
        monkeypatch.setenv("REQUIRE_API_AUTH", "true")
        monkeypatch.setenv("ALLOW_BYOK", "true")

        auth = APIKeyAuth()
        with pytest.raises(HTTPException) as exc_info:
            await auth(api_key=None, openrouter_key="invalid-key")

        assert exc_info.value.status_code == 401
        assert "Invalid OpenRouter key format" in exc_info.value.detail


class TestAuditLogger:
    """Tests for audit logging."""

    def test_init_enabled(self, monkeypatch, tmp_path):
        """Test audit logger initialization when enabled."""
        log_file = tmp_path / "audit.log"
        monkeypatch.setenv("ENABLE_AUDIT_LOG", "true")
        monkeypatch.setenv("AUDIT_LOG_FILE", str(log_file))

        logger = AuditLogger()
        assert logger.enabled is True

    def test_init_disabled(self, monkeypatch):
        """Test audit logger initialization when disabled."""
        monkeypatch.setenv("ENABLE_AUDIT_LOG", "false")

        logger = AuditLogger()
        assert logger.enabled is False

    def test_init_fallback_on_permission_error(self, monkeypatch):
        """Test fallback to console when log file can't be created."""
        monkeypatch.setenv("ENABLE_AUDIT_LOG", "true")
        monkeypatch.setenv("AUDIT_LOG_FILE", "/nonexistent/path/audit.log")

        # Should not raise, falls back to console
        logger = AuditLogger()
        assert logger.enabled is True

    def test_log_request_disabled(self, monkeypatch):
        """Test log_request does nothing when disabled."""
        monkeypatch.setenv("ENABLE_AUDIT_LOG", "false")

        logger = AuditLogger()
        # Create a minimal request-like object
        request = create_test_request()
        # Should not raise
        logger.log_request(request, api_key_hash="abc123", user_id="test_user")

    def test_log_request_enabled(self, monkeypatch, tmp_path, caplog):
        """Test log_request logs when enabled."""
        import logging

        log_file = tmp_path / "audit.log"
        monkeypatch.setenv("ENABLE_AUDIT_LOG", "true")
        monkeypatch.setenv("AUDIT_LOG_FILE", str(log_file))

        logger = AuditLogger()
        logger.logger.setLevel(logging.INFO)

        request = create_test_request()
        logger.log_request(request, api_key_hash="abc123", user_id="test_user")

    def test_log_response_disabled(self, monkeypatch):
        """Test log_response does nothing when disabled."""
        monkeypatch.setenv("ENABLE_AUDIT_LOG", "false")

        logger = AuditLogger()
        request = create_test_request()
        logger.log_response(request, status_code=200, processing_time_ms=50.5)

    def test_log_response_enabled(self, monkeypatch, tmp_path):
        """Test log_response logs when enabled."""
        log_file = tmp_path / "audit.log"
        monkeypatch.setenv("ENABLE_AUDIT_LOG", "true")
        monkeypatch.setenv("AUDIT_LOG_FILE", str(log_file))

        logger = AuditLogger()
        request = create_test_request()
        logger.log_response(request, status_code=200, processing_time_ms=50.5)

    def test_log_error_disabled(self, monkeypatch):
        """Test log_error does nothing when disabled."""
        monkeypatch.setenv("ENABLE_AUDIT_LOG", "false")

        logger = AuditLogger()
        request = create_test_request()
        logger.log_error(request, error=ValueError("test error"), api_key_hash="abc")

    def test_log_error_enabled(self, monkeypatch, tmp_path):
        """Test log_error logs when enabled."""
        log_file = tmp_path / "audit.log"
        monkeypatch.setenv("ENABLE_AUDIT_LOG", "true")
        monkeypatch.setenv("AUDIT_LOG_FILE", str(log_file))

        logger = AuditLogger()
        request = create_test_request()
        logger.log_error(request, error=ValueError("test error"), api_key_hash="abc")


class TestGenerateAPIKey:
    """Tests for API key generation."""

    def test_generate_api_key_length(self):
        """Test generated key has correct length."""
        key = generate_api_key()
        assert len(key) == 64  # 32 bytes = 64 hex chars

    def test_generate_api_key_unique(self):
        """Test generated keys are unique."""
        keys = [generate_api_key() for _ in range(10)]
        assert len(set(keys)) == 10

    def test_generate_api_key_hex(self):
        """Test generated key is valid hex."""
        key = generate_api_key()
        # Should not raise
        int(key, 16)


class TestVerifyOrigin:
    """Tests for origin verification."""

    def test_verify_origin_allowed(self):
        """Test request from allowed origin passes."""
        request = create_test_request(headers={"origin": "https://example.com/page"})
        allowed = ["https://example.com"]

        assert verify_origin(request, allowed) is True

    def test_verify_origin_not_allowed(self):
        """Test request from disallowed origin fails."""
        request = create_test_request(headers={"origin": "https://evil.com"})
        allowed = ["https://example.com"]

        assert verify_origin(request, allowed) is False

    def test_verify_origin_no_header(self):
        """Test request without origin header passes (direct API access)."""
        request = create_test_request(headers={})
        allowed = ["https://example.com"]

        assert verify_origin(request, allowed) is True

    def test_verify_origin_referer_fallback(self):
        """Test using referer header when origin not present."""
        request = create_test_request(headers={"referer": "https://example.com/page"})
        allowed = ["https://example.com"]

        assert verify_origin(request, allowed) is True

    def test_verify_origin_multiple_allowed(self):
        """Test with multiple allowed origins."""
        request = create_test_request(headers={"origin": "https://second.com"})
        allowed = ["https://first.com", "https://second.com", "https://third.com"]

        assert verify_origin(request, allowed) is True


def create_test_request(
    method: str = "GET",
    path: str = "/test",
    headers: dict | None = None,
    client_host: str = "127.0.0.1",
) -> Request:
    """Create a test Request object without mocking.

    Args:
        method: HTTP method
        path: Request path
        headers: Request headers
        client_host: Client IP address

    Returns:
        A Request-like object for testing
    """

    class TestClient:
        """Minimal client for testing."""

        def __init__(self):
            self.host = client_host

    # Create a proper Request object
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": [(k.encode(), v.encode()) for k, v in (headers or {}).items()],
        "query_string": b"",
        "root_path": "",
    }

    request = Request(scope)
    # Manually set client since we're not in a real ASGI context
    request._client = TestClient()

    return request
