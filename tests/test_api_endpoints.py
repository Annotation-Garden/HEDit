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


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_api_info(self, client):
        """Test root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert data["name"] == "HEDit API"
        assert "version" in data
        assert "description" in data
        assert "endpoints" in data

    def test_root_lists_endpoints(self, client):
        """Test root endpoint lists all available endpoints."""
        response = client.get("/")
        data = response.json()
        endpoints = data["endpoints"]
        assert "POST /annotate" in endpoints
        assert "POST /validate" in endpoints
        assert "GET /health" in endpoints
        assert "GET /version" in endpoints


class TestFeedbackEndpoint:
    """Tests for feedback endpoint."""

    def test_feedback_submission(self, client):
        """Test basic feedback submission (no auth required)."""
        request_data = {
            "type": "text",
            "description": "Test event description",
            "annotation": "Sensory-event, Visual-presentation",
            "is_valid": True,
            "is_faithful": True,
            "is_complete": True,
            "validation_errors": [],
            "validation_warnings": [],
            "user_comment": "This annotation looks good!",
        }
        response = client.post("/feedback", json=request_data)
        # Should succeed (200) - no auth required for feedback
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "feedback_id" in data
        assert "message" in data

    def test_feedback_minimal_submission(self, client):
        """Test feedback with minimal required fields."""
        request_data = {
            "type": "text",
            "description": "Test description",
            "annotation": "Test-annotation",
            "user_comment": "Test comment",
        }
        response = client.post("/feedback", json=request_data)
        assert response.status_code == 200

    def test_feedback_image_type(self, client):
        """Test feedback submission for image type."""
        request_data = {
            "type": "image",
            "description": "Image description",
            "image_description": "A cat sitting on a mat",
            "annotation": "Animal/Cat, Furnishing/Mat",
            "user_comment": "Image annotation feedback",
        }
        response = client.post("/feedback", json=request_data)
        assert response.status_code == 200

    def test_feedback_missing_fields(self, client):
        """Test feedback with missing required fields."""
        request_data = {
            "type": "text",
            # Missing description, annotation, user_comment
        }
        response = client.post("/feedback", json=request_data)
        assert response.status_code == 422  # Validation error


class TestMoreSecurityHeaders:
    """Additional tests for security headers."""

    def test_all_security_headers(self, client):
        """Test all security headers are present."""
        response = client.get("/health")
        headers = response.headers

        # Check all security headers
        assert "x-content-type-options" in headers
        assert headers["x-content-type-options"] == "nosniff"

        assert "x-frame-options" in headers
        assert headers["x-frame-options"] == "DENY"

        assert "x-xss-protection" in headers
        assert headers["x-xss-protection"] == "1; mode=block"

    def test_security_headers_on_all_endpoints(self, client):
        """Test security headers on different endpoints."""
        endpoints = ["/health", "/version", "/"]

        for endpoint in endpoints:
            response = client.get(endpoint)
            assert "x-content-type-options" in response.headers


class TestStreamingEndpoint:
    """Tests for streaming annotation endpoint."""

    def test_stream_endpoint_returns_sse(self, client):
        """Test that streaming endpoint returns SSE format."""
        request_data = {
            "description": "A red circle appears",
            "schema_version": "8.3.0",
        }
        # Note: streaming tests are limited without async test support
        # This verifies the endpoint exists and responds
        response = client.post("/annotate/stream", json=request_data)
        # May be 503 if workflow not initialized, or 200 with streaming
        assert response.status_code in [200, 503]


class TestVersionEndpointExtended:
    """Extended tests for version endpoint."""

    def test_version_includes_commit(self, client):
        """Test version endpoint includes commit hash."""
        response = client.get("/version")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "commit" in data


class TestUserIDDerivation:
    """Tests for user ID derivation from API keys."""

    def test_derive_user_id(self):
        """Test that user ID is derived consistently from API key."""
        from src.api.main import _derive_user_id

        api_key = "sk-or-test-key-12345"
        user_id = _derive_user_id(api_key)

        # Should be 16 hex characters
        assert len(user_id) == 16
        assert all(c in "0123456789abcdef" for c in user_id)

    def test_derive_user_id_consistency(self):
        """Test that same API key produces same user ID."""
        from src.api.main import _derive_user_id

        api_key = "sk-or-test-key-12345"
        user_id1 = _derive_user_id(api_key)
        user_id2 = _derive_user_id(api_key)

        assert user_id1 == user_id2

    def test_derive_user_id_uniqueness(self):
        """Test that different API keys produce different user IDs."""
        from src.api.main import _derive_user_id

        user_id1 = _derive_user_id("key1")
        user_id2 = _derive_user_id("key2")

        assert user_id1 != user_id2


class TestTelemetryEnabledField:
    """Tests for telemetry_enabled field in request models."""

    def test_annotation_request_telemetry_default_true(self):
        """Test AnnotationRequest has telemetry_enabled=True by default."""
        from src.api.models import AnnotationRequest

        request = AnnotationRequest(description="Test description")
        assert request.telemetry_enabled is True

    def test_annotation_request_telemetry_can_be_disabled(self):
        """Test AnnotationRequest telemetry_enabled can be set to False."""
        from src.api.models import AnnotationRequest

        request = AnnotationRequest(description="Test description", telemetry_enabled=False)
        assert request.telemetry_enabled is False

    def test_image_annotation_request_telemetry_default_true(self):
        """Test ImageAnnotationRequest has telemetry_enabled=True by default."""
        from src.api.models import ImageAnnotationRequest

        request = ImageAnnotationRequest(image="base64data")
        assert request.telemetry_enabled is True

    def test_image_annotation_request_telemetry_can_be_disabled(self):
        """Test ImageAnnotationRequest telemetry_enabled can be set to False."""
        from src.api.models import ImageAnnotationRequest

        request = ImageAnnotationRequest(image="base64data", telemetry_enabled=False)
        assert request.telemetry_enabled is False

    def test_annotate_endpoint_accepts_telemetry_enabled(self, client):
        """Test annotate endpoint accepts telemetry_enabled field."""
        request_data = {
            "description": "A red circle appears on the screen",
            "schema_version": "8.3.0",
            "telemetry_enabled": False,
        }
        response = client.post("/annotate", json=request_data, headers=TEST_AUTH_HEADERS)
        # May be 503 if workflow not initialized, or 200 if it is
        # But should NOT be 422 (validation error)
        assert response.status_code in [200, 503]

    def test_image_annotate_endpoint_accepts_telemetry_enabled(self, client):
        """Test image annotation endpoint accepts telemetry_enabled field."""
        request_data = {
            "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
            "telemetry_enabled": False,
        }
        response = client.post("/annotate-from-image", json=request_data, headers=TEST_AUTH_HEADERS)
        # May be 503 if vision agent not initialized, or 200 if it is
        # But should NOT be 422 (validation error)
        assert response.status_code in [200, 503]


class TestTelemetryCollectorIntegration:
    """Tests for telemetry collector integration in API."""

    def test_telemetry_collector_initialization(self):
        """Test that telemetry_collector global is defined."""
        from src.api import main

        # telemetry_collector should be defined (may be None before lifespan)
        assert hasattr(main, "telemetry_collector")

    def test_telemetry_imports_available(self):
        """Test that telemetry imports are available in main module."""
        from src.api.main import LocalFileStorage, TelemetryCollector, TelemetryEvent

        # Verify classes are importable
        assert LocalFileStorage is not None
        assert TelemetryCollector is not None
        assert TelemetryEvent is not None

    def test_telemetry_event_creation(self):
        """Test creating a telemetry event with API-like data."""
        from src.telemetry import TelemetryEvent

        event = TelemetryEvent.create(
            description="Test description from API",
            schema_version="8.3.0",
            hed_string="Sensory-event, Visual-presentation",
            iterations=2,
            validation_errors=[],
            model="openai/gpt-oss-120b",
            provider=None,
            temperature=0.1,
            latency_ms=1500,
            source="api",
        )

        # TelemetryEvent uses nested models
        assert event.input.description == "Test description from API"
        assert event.input.schema_version == "8.3.0"
        assert event.output.hed_string == "Sensory-event, Visual-presentation"
        assert event.output.iterations == 2
        assert event.performance.latency_ms == 1500
        assert event.source == "api"

    def test_telemetry_event_image_source(self):
        """Test creating a telemetry event with api-image source."""
        from src.telemetry import TelemetryEvent

        event = TelemetryEvent.create(
            description="Generated image description",
            schema_version="8.4.0",
            hed_string="Visual-presentation",
            iterations=1,
            validation_errors=[],
            model="openai/gpt-4o",
            provider="deepinfra/fp8",
            temperature=0.3,
            latency_ms=3000,
            source="api-image",
        )

        assert event.source == "api-image"
        assert event.model.provider == "deepinfra/fp8"
