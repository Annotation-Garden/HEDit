"""Tests for API endpoints.

These tests use a test API key to authenticate requests.

IMPORTANT: These tests modify environment variables temporarily.
App is imported inside the fixture to avoid polluting global state.
"""

import importlib
import os

import pytest
from fastapi.testclient import TestClient

# Test API key header
TEST_AUTH_HEADERS = {"X-API-Key": "test-api-key-for-unit-tests"}


@pytest.fixture
def client():
    """Create a test client for the FastAPI app with auth enabled."""
    # Store original env state
    original_env = {}
    for key in ["REQUIRE_API_AUTH", "API_KEYS"]:
        if key in os.environ:
            original_env[key] = os.environ[key]

    # Set test environment
    os.environ["REQUIRE_API_AUTH"] = "true"
    os.environ["API_KEYS"] = "test-api-key-for-unit-tests"

    # Reload security module to pick up new env vars
    from src.api import security

    importlib.reload(security)

    # Import app after setting env vars
    from src.api.main import app

    yield TestClient(app, raise_server_exceptions=False)

    # Restore original values
    for key in ["REQUIRE_API_AUTH", "API_KEYS"]:
        if key in original_env:
            os.environ[key] = original_env[key]
        elif key in os.environ:
            del os.environ[key]

    # Reload security to restore original state
    importlib.reload(security)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_status(self, client):
        """Test health endpoint returns status (no auth required)."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data

    def test_health_response_model(self, client):
        """Test health response matches model."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        # Verify all expected fields
        assert "status" in data
        assert "version" in data
        assert "llm_available" in data
        assert "validator_available" in data


class TestVersionEndpoint:
    """Tests for version endpoint."""

    def test_version_returns_info(self, client):
        """Test version endpoint returns version info."""
        response = client.get("/version")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data


class TestValidationEndpoint:
    """Tests for validation endpoint."""

    def test_validate_valid_hed_string(self, client):
        """Test validation of valid HED string."""
        request_data = {
            "hed_string": "Sensory-event, Visual-presentation",
            "schema_version": "8.3.0",
        }
        response = client.post("/validate", json=request_data, headers=TEST_AUTH_HEADERS)
        # 200 if schema_loader initialized, 503 if not
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "is_valid" in data
            assert "errors" in data

    def test_validate_invalid_hed_string(self, client):
        """Test validation of invalid HED string."""
        request_data = {
            "hed_string": "CompletelyInvalidTag123",
            "schema_version": "8.3.0",
        }
        response = client.post("/validate", json=request_data, headers=TEST_AUTH_HEADERS)
        # 200 if schema_loader initialized, 503 if not
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            # Should have some issues
            assert "is_valid" in data

    def test_validate_empty_string(self, client):
        """Test validation of empty string."""
        request_data = {
            "hed_string": "",
            "schema_version": "8.3.0",
        }
        response = client.post("/validate", json=request_data, headers=TEST_AUTH_HEADERS)
        # 422 if empty string rejected by pydantic, 200/503 otherwise
        assert response.status_code in [200, 422, 503]

    def test_validate_without_auth(self, client):
        """Test validate endpoint without auth."""
        request_data = {
            "hed_string": "Event",
            "schema_version": "8.3.0",
        }
        response = client.post("/validate", json=request_data)
        assert response.status_code == 401


class TestAnnotateEndpoint:
    """Tests for annotation endpoint."""

    def test_annotate_with_auth(self, client):
        """Test annotate endpoint with auth."""
        request_data = {
            "description": "A red circle appears on the screen",
            "schema_version": "8.3.0",
        }
        response = client.post("/annotate", json=request_data, headers=TEST_AUTH_HEADERS)
        # May be 503 if workflow not initialized, or 200 if it is
        assert response.status_code in [200, 503]

    def test_annotate_with_invalid_auth(self, client):
        """Test annotate endpoint with invalid auth."""
        request_data = {
            "description": "A red circle appears on the screen",
            "schema_version": "8.3.0",
        }
        response = client.post(
            "/annotate",
            json=request_data,
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401

    def test_annotate_missing_auth(self, client):
        """Test annotate endpoint with missing auth."""
        request_data = {
            "description": "A red circle appears on the screen",
            "schema_version": "8.3.0",
        }
        response = client.post("/annotate", json=request_data)
        assert response.status_code == 401


class TestImageAnnotateEndpoint:
    """Tests for image annotation endpoint."""

    def test_image_annotate_with_auth(self, client):
        """Test image annotation with auth."""
        # Use a minimal valid base64 PNG (1x1 red pixel)
        request_data = {
            "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
        }
        response = client.post("/annotate-from-image", json=request_data, headers=TEST_AUTH_HEADERS)
        # May be 503 if vision agent not initialized, or 200 if it is
        assert response.status_code in [200, 503]


class TestCORSHeaders:
    """Tests for CORS configuration."""

    def test_cors_preflight(self, client):
        """Test CORS preflight request."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # OPTIONS should be handled by CORS middleware
        assert response.status_code in [200, 204, 405]


class TestSecurityHeaders:
    """Tests for security headers."""

    def test_security_headers_present(self, client):
        """Test that security headers are present in response."""
        response = client.get("/health")
        # Check security headers added by middleware
        headers = response.headers
        # X-Content-Type-Options should be present
        assert "x-content-type-options" in headers


class TestRequestValidation:
    """Tests for request validation."""

    def test_annotate_missing_description(self, client):
        """Test annotate endpoint with missing description."""
        request_data = {
            "schema_version": "8.3.0",
        }
        response = client.post("/annotate", json=request_data, headers=TEST_AUTH_HEADERS)
        assert response.status_code == 422  # Validation error

    def test_validate_missing_hed_string(self, client):
        """Test validate endpoint with missing HED string."""
        request_data = {
            "schema_version": "8.3.0",
        }
        response = client.post("/validate", json=request_data, headers=TEST_AUTH_HEADERS)
        assert response.status_code == 422  # Validation error
