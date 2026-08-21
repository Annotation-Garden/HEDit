"""Tests for API endpoints.

These tests use a test API key to authenticate requests.

IMPORTANT: These tests modify environment variables temporarily.
App is imported inside the fixture to avoid polluting global state.
"""

import importlib
import os
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from src.utils.llm_usage import process_ledger

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
        # This verifies the endpoint exists and responds with authentication
        response = client.post("/annotate/stream", json=request_data, headers=TEST_AUTH_HEADERS)
        # May be 503 if workflow not initialized, or 200 with streaming
        assert response.status_code in [200, 503]

    def test_stream_endpoint_requires_auth(self, client):
        """Test that streaming endpoint requires authentication."""
        request_data = {
            "description": "A red circle appears",
            "schema_version": "8.3.0",
        }
        response = client.post("/annotate/stream", json=request_data)
        assert response.status_code == 401

    def test_stream_endpoint_accepts_model_override_headers(self, client):
        """Test that streaming endpoint accepts model/provider override headers."""
        request_data = {
            "description": "A red circle appears",
            "schema_version": "8.3.0",
        }
        headers = {
            **TEST_AUTH_HEADERS,
            "X-OpenRouter-Model": "claude-haiku-4-5",
            "X-OpenRouter-Provider": "anthropic",
        }
        response = client.post("/annotate/stream", json=request_data, headers=headers)
        # 503 expected without workflow initialized, but should not be 400 for bad headers
        assert response.status_code in [200, 503]

    def test_stream_endpoint_accepts_temperature_header(self, client):
        """Test that streaming endpoint accepts temperature header."""
        request_data = {
            "description": "A red circle appears",
            "schema_version": "8.3.0",
        }
        headers = {
            **TEST_AUTH_HEADERS,
            "X-OpenRouter-Temperature": "0.5",
        }
        response = client.post("/annotate/stream", json=request_data, headers=headers)
        assert response.status_code in [200, 503]

    def test_stream_endpoint_handles_invalid_temperature_gracefully(self, client):
        """Test that streaming endpoint handles invalid temperature header."""
        request_data = {
            "description": "A red circle appears",
            "schema_version": "8.3.0",
        }
        headers = {
            **TEST_AUTH_HEADERS,
            "X-OpenRouter-Temperature": "invalid",
        }
        # Should not fail, just ignore invalid temperature
        response = client.post("/annotate/stream", json=request_data, headers=headers)
        assert response.status_code in [200, 503]

    def test_stream_endpoint_byok_rejects_non_anthropic_key(self, client):
        """A BYOK key that is not an Anthropic key is rejected at auth (401)."""
        request_data = {
            "description": "A red circle appears",
            "schema_version": "8.3.0",
        }
        headers = {"X-OpenRouter-Key": "sk-or-v1-validformatbutwrongprovider123"}
        response = client.post("/annotate/stream", json=request_data, headers=headers)
        assert response.status_code == 401
        assert "Invalid BYOK key format" in response.json()["detail"]

    def test_stream_endpoint_returns_sse_content_type(self, client):
        """Test that streaming endpoint returns SSE content type when successful."""
        request_data = {
            "description": "A red circle appears",
            "schema_version": "8.3.0",
        }
        response = client.post("/annotate/stream", json=request_data, headers=TEST_AUTH_HEADERS)
        if response.status_code == 200:
            assert response.headers.get("content-type") == "text/event-stream; charset=utf-8"

    def test_stream_endpoint_with_user_id_header(self, client):
        """Test that streaming endpoint accepts user ID header."""
        request_data = {
            "description": "A red circle appears",
            "schema_version": "8.3.0",
        }
        headers = {
            **TEST_AUTH_HEADERS,
            "X-User-Id": "frontend-test-0.6.6",
        }
        response = client.post("/annotate/stream", json=request_data, headers=headers)
        assert response.status_code in [200, 503]

    def test_stream_endpoint_with_eval_model_headers(self, client):
        """Test that streaming endpoint accepts eval model headers."""
        request_data = {
            "description": "A red circle appears",
            "schema_version": "8.3.0",
        }
        headers = {
            **TEST_AUTH_HEADERS,
            "X-OpenRouter-Eval-Model": "claude-haiku-4-5",
        }
        response = client.post("/annotate/stream", json=request_data, headers=headers)
        assert response.status_code in [200, 503]

    def test_stream_endpoint_with_assessment_flag(self, client):
        """Test that streaming endpoint accepts run_assessment flag."""
        request_data = {
            "description": "A red circle appears",
            "schema_version": "8.3.0",
            "run_assessment": True,
        }
        response = client.post("/annotate/stream", json=request_data, headers=TEST_AUTH_HEADERS)
        assert response.status_code in [200, 503]

    def test_stream_endpoint_with_max_validation_attempts(self, client):
        """Test that streaming endpoint accepts max_validation_attempts."""
        request_data = {
            "description": "A red circle appears",
            "schema_version": "8.3.0",
            "max_validation_attempts": 5,
        }
        response = client.post("/annotate/stream", json=request_data, headers=TEST_AUTH_HEADERS)
        assert response.status_code in [200, 503]


class TestStreamingWithMockedWorkflow:
    """Tests for streaming endpoint with mocked workflow."""

    @pytest.fixture
    def client_with_workflow(self):
        """Create a test client with a mocked workflow."""
        original_env = {}
        for key in ["REQUIRE_API_AUTH", "API_KEYS", "ANTHROPIC_API_KEY"]:
            if key in os.environ:
                original_env[key] = os.environ[key]

        os.environ["REQUIRE_API_AUTH"] = "true"
        os.environ["API_KEYS"] = "test-api-key-for-unit-tests"
        os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"

        from src.api import security

        importlib.reload(security)

        # Create mock workflow
        mock_workflow = MagicMock()
        mock_graph = MagicMock()

        # Create async generator for streaming events
        async def mock_stream_events(*args, **kwargs):
            # Simulate node events
            yield {"event": "on_chain_start", "name": "annotate", "data": {}}
            yield {
                "event": "on_chain_end",
                "name": "annotate",
                "data": {"output": {"current_annotation": "Sensory-event, Visual"}},
            }
            yield {"event": "on_chain_start", "name": "validate", "data": {}}
            yield {
                "event": "on_chain_end",
                "name": "validate",
                "data": {"output": {"is_valid": True, "validation_errors": []}},
            }
            yield {"event": "on_chain_start", "name": "evaluate", "data": {}}
            yield {
                "event": "on_chain_end",
                "name": "evaluate",
                "data": {"output": {"is_faithful": True, "is_complete": True}},
            }

        mock_graph.astream_events = mock_stream_events
        mock_workflow.graph = mock_graph

        # Patch the workflow in main module
        with patch("src.api.main.workflow", mock_workflow):
            from src.api.main import app

            yield TestClient(app, raise_server_exceptions=False)

        # Restore original values
        for key in ["REQUIRE_API_AUTH", "API_KEYS", "ANTHROPIC_API_KEY"]:
            if key in original_env:
                os.environ[key] = original_env[key]
            elif key in os.environ:
                del os.environ[key]

        importlib.reload(security)

    def test_stream_returns_progress_events(self, client_with_workflow):
        """Test that streaming returns progress events."""
        request_data = {
            "description": "A red circle appears",
            "schema_version": "8.3.0",
        }
        response = client_with_workflow.post(
            "/annotate/stream", json=request_data, headers=TEST_AUTH_HEADERS
        )
        # Should get 200 with mock workflow
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

    def test_stream_content_has_events(self, client_with_workflow):
        """Test that streaming response contains SSE events."""
        request_data = {
            "description": "A red circle appears",
            "schema_version": "8.3.0",
        }
        response = client_with_workflow.post(
            "/annotate/stream", json=request_data, headers=TEST_AUTH_HEADERS
        )
        content = response.text
        # Should contain event markers
        assert "event:" in content or response.status_code == 503

    def test_stream_has_safari_padding_comment(self, client_with_workflow):
        """SSE stream should start with padding comment for Safari compatibility."""
        request_data = {
            "description": "A red circle appears",
            "schema_version": "8.3.0",
        }
        response = client_with_workflow.post(
            "/annotate/stream", json=request_data, headers=TEST_AUTH_HEADERS
        )
        if response.status_code == 200:
            # Stream should start with the SSE comment
            assert response.text.startswith(": stream opened")

    def test_stream_has_nosniff_header(self, client_with_workflow):
        """SSE streaming response should include X-Content-Type-Options: nosniff."""
        request_data = {
            "description": "A red circle appears",
            "schema_version": "8.3.0",
        }
        response = client_with_workflow.post(
            "/annotate/stream", json=request_data, headers=TEST_AUTH_HEADERS
        )
        if response.status_code == 200:
            assert response.headers.get("x-content-type-options") == "nosniff"


class TestModelOverrideWithEnv:
    """Tests for model override with environment variables set."""

    @pytest.fixture
    def client_with_env(self):
        """Create a test client with ANTHROPIC_API_KEY set."""
        original_env = {}
        for key in ["REQUIRE_API_AUTH", "API_KEYS", "ANTHROPIC_API_KEY"]:
            if key in os.environ:
                original_env[key] = os.environ[key]

        os.environ["REQUIRE_API_AUTH"] = "true"
        os.environ["API_KEYS"] = "test-api-key-for-unit-tests"
        os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"

        from src.api import security

        importlib.reload(security)
        from src.api.main import app

        yield TestClient(app, raise_server_exceptions=False)

        # Restore original values
        for key in ["REQUIRE_API_AUTH", "API_KEYS", "ANTHROPIC_API_KEY"]:
            if key in original_env:
                os.environ[key] = original_env[key]
            elif key in os.environ:
                del os.environ[key]

        importlib.reload(security)

    def test_annotate_with_model_override(self, client_with_env):
        """Test annotate endpoint with model override headers."""
        request_data = {
            "description": "A red circle appears",
            "schema_version": "8.3.0",
        }
        headers = {
            **TEST_AUTH_HEADERS,
            "X-OpenRouter-Model": "claude-haiku-4-5",
            "X-OpenRouter-Provider": "anthropic",
        }
        response = client_with_env.post("/annotate", json=request_data, headers=headers)
        # The fake key passes workflow creation but fails at the LLM call,
        # which now surfaces as 401 (auth) rather than a generic 500
        assert response.status_code in [200, 401, 503]

    def test_stream_with_model_override(self, client_with_env):
        """Test streaming endpoint with model override headers."""
        request_data = {
            "description": "A red circle appears",
            "schema_version": "8.3.0",
        }
        headers = {
            **TEST_AUTH_HEADERS,
            "X-OpenRouter-Model": "claude-haiku-4-5",
            "X-OpenRouter-Provider": "anthropic",
        }
        response = client_with_env.post("/annotate/stream", json=request_data, headers=headers)
        # Will fail to create workflow with fake key, but tests the code path
        assert response.status_code in [200, 500, 503]

    def test_stream_with_all_headers(self, client_with_env):
        """Test streaming endpoint with all override headers."""
        request_data = {
            "description": "A red circle appears",
            "schema_version": "8.3.0",
            "run_assessment": True,
            "max_validation_attempts": 5,
        }
        headers = {
            **TEST_AUTH_HEADERS,
            "X-OpenRouter-Model": "claude-sonnet-5",
            "X-OpenRouter-Temperature": "0.5",
            "X-OpenRouter-Eval-Model": "claude-haiku-4-5",
            "X-User-Id": "test-user-123",
        }
        response = client_with_env.post("/annotate/stream", json=request_data, headers=headers)
        assert response.status_code in [200, 500, 503]


class TestVersionEndpointExtended:
    """Extended tests for version endpoint."""

    def test_version_includes_commit(self, client):
        """Test version endpoint includes commit hash."""
        response = client.get("/version")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "commit" in data


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
            model="claude-haiku-4-5",
            provider="anthropic",
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
            model="claude-sonnet-5",
            provider="anthropic",
            temperature=0.3,
            latency_ms=3000,
            source="api-image",
        )

        assert event.source == "api-image"
        assert event.model.provider == "anthropic"

    def test_telemetry_event_stream_source(self):
        """Test creating a telemetry event with api-stream source."""
        from src.telemetry import TelemetryEvent

        event = TelemetryEvent.create(
            description="Streaming annotation request",
            schema_version="8.4.0",
            hed_string="Sensory-event, Visual-presentation",
            iterations=2,
            validation_errors=[],
            model="claude-haiku-4-5",
            provider="anthropic",
            temperature=0.1,
            latency_ms=2000,
            source="api-stream",
        )

        assert event.source == "api-stream"
        assert event.input.description == "Streaming annotation request"
        assert event.performance.latency_ms == 2000

    def test_telemetry_event_image_stream_source(self):
        """Test creating a telemetry event with api-image-stream source."""
        from src.telemetry import TelemetryEvent

        event = TelemetryEvent.create(
            description="Image description from vision model",
            schema_version="8.4.0",
            hed_string="Visual-presentation",
            iterations=1,
            validation_errors=[],
            model="claude-sonnet-5",
            provider="anthropic",
            temperature=0.3,
            latency_ms=4000,
            source="api-image-stream",
        )

        assert event.source == "api-image-stream"
        assert event.input.description == "Image description from vision model"


class TestStreamingTelemetry:
    """Tests for telemetry collection in streaming endpoints."""

    @pytest.fixture
    def client_with_telemetry(self):
        """Create a test client with mocked workflow and telemetry collector."""
        original_env = {}
        for key in ["REQUIRE_API_AUTH", "API_KEYS", "ANTHROPIC_API_KEY"]:
            if key in os.environ:
                original_env[key] = os.environ[key]

        os.environ["REQUIRE_API_AUTH"] = "true"
        os.environ["API_KEYS"] = "test-api-key-for-unit-tests"
        os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"

        from src.api import security

        importlib.reload(security)

        # Create mock workflow
        mock_workflow = MagicMock()
        mock_graph = MagicMock()

        async def mock_stream_events(*args, **kwargs):
            yield {"event": "on_chain_start", "name": "annotate", "data": {}}
            yield {
                "event": "on_chain_end",
                "name": "annotate",
                "data": {
                    "output": {
                        "current_annotation": "Sensory-event, Visual-presentation",
                        "validation_attempts": 1,
                    }
                },
            }
            yield {"event": "on_chain_start", "name": "validate", "data": {}}
            yield {
                "event": "on_chain_end",
                "name": "validate",
                "data": {"output": {"is_valid": True, "validation_errors": []}},
            }
            yield {"event": "on_chain_start", "name": "evaluate", "data": {}}
            yield {
                "event": "on_chain_end",
                "name": "evaluate",
                "data": {"output": {"is_faithful": True, "is_complete": True}},
            }

        mock_graph.astream_events = mock_stream_events
        mock_workflow.graph = mock_graph

        # Create mock telemetry collector with a list to track collected events
        mock_collector = MagicMock()
        collected_events = []

        async def track_collect(event):
            collected_events.append(event)
            return True

        mock_collector.collect = track_collect

        with (
            patch("src.api.main.workflow", mock_workflow),
            patch("src.api.main.telemetry_collector", mock_collector),
        ):
            from src.api.main import app

            client = TestClient(app, raise_server_exceptions=False)
            yield client, collected_events

        for key in ["REQUIRE_API_AUTH", "API_KEYS", "ANTHROPIC_API_KEY"]:
            if key in original_env:
                os.environ[key] = original_env[key]
            elif key in os.environ:
                del os.environ[key]

        importlib.reload(security)

    @pytest.fixture
    def client_with_telemetry_disabled(self):
        """Create a test client with mocked workflow but no telemetry collector."""
        original_env = {}
        for key in ["REQUIRE_API_AUTH", "API_KEYS", "ANTHROPIC_API_KEY"]:
            if key in os.environ:
                original_env[key] = os.environ[key]

        os.environ["REQUIRE_API_AUTH"] = "true"
        os.environ["API_KEYS"] = "test-api-key-for-unit-tests"
        os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"

        from src.api import security

        importlib.reload(security)

        mock_workflow = MagicMock()
        mock_graph = MagicMock()

        async def mock_stream_events(*args, **kwargs):
            yield {"event": "on_chain_start", "name": "annotate", "data": {}}
            yield {
                "event": "on_chain_end",
                "name": "annotate",
                "data": {
                    "output": {
                        "current_annotation": "Sensory-event",
                        "validation_attempts": 1,
                    }
                },
            }
            yield {"event": "on_chain_start", "name": "validate", "data": {}}
            yield {
                "event": "on_chain_end",
                "name": "validate",
                "data": {"output": {"is_valid": True, "validation_errors": []}},
            }

        mock_graph.astream_events = mock_stream_events
        mock_workflow.graph = mock_graph

        # Track if collect was called
        collected_events = []

        async def track_collect(event):
            collected_events.append(event)
            return True

        mock_collector = MagicMock()
        mock_collector.collect = track_collect

        with (
            patch("src.api.main.workflow", mock_workflow),
            patch("src.api.main.telemetry_collector", mock_collector),
        ):
            from src.api.main import app

            client = TestClient(app, raise_server_exceptions=False)
            yield client, collected_events

        for key in ["REQUIRE_API_AUTH", "API_KEYS", "ANTHROPIC_API_KEY"]:
            if key in original_env:
                os.environ[key] = original_env[key]
            elif key in os.environ:
                del os.environ[key]

        importlib.reload(security)

    @pytest.fixture
    def client_with_failing_workflow(self):
        """Create a test client with a workflow that raises an error."""
        original_env = {}
        for key in ["REQUIRE_API_AUTH", "API_KEYS", "ANTHROPIC_API_KEY"]:
            if key in os.environ:
                original_env[key] = os.environ[key]

        os.environ["REQUIRE_API_AUTH"] = "true"
        os.environ["API_KEYS"] = "test-api-key-for-unit-tests"
        os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"

        from src.api import security

        importlib.reload(security)

        mock_workflow = MagicMock()
        mock_graph = MagicMock()

        async def mock_stream_events_error(*args, **kwargs):
            yield {"event": "on_chain_start", "name": "annotate", "data": {}}
            yield {
                "event": "on_chain_end",
                "name": "annotate",
                "data": {
                    "output": {
                        "current_annotation": "Partial-annotation",
                        "validation_attempts": 0,
                    }
                },
            }
            raise RuntimeError("Simulated workflow failure")

        mock_graph.astream_events = mock_stream_events_error
        mock_workflow.graph = mock_graph

        collected_events = []

        async def track_collect(event):
            collected_events.append(event)
            return True

        mock_collector = MagicMock()
        mock_collector.collect = track_collect

        with (
            patch("src.api.main.workflow", mock_workflow),
            patch("src.api.main.telemetry_collector", mock_collector),
        ):
            from src.api.main import app

            client = TestClient(app, raise_server_exceptions=False)
            yield client, collected_events

        for key in ["REQUIRE_API_AUTH", "API_KEYS", "ANTHROPIC_API_KEY"]:
            if key in original_env:
                os.environ[key] = original_env[key]
            elif key in os.environ:
                del os.environ[key]

        importlib.reload(security)

    def test_stream_telemetry_collected_on_success(self, client_with_telemetry):
        """Test that telemetry is collected for successful streaming requests."""
        client, collected_events = client_with_telemetry
        request_data = {
            "description": "A red circle appears on screen",
            "schema_version": "8.3.0",
            "telemetry_enabled": True,
        }
        response = client.post("/annotate/stream", json=request_data, headers=TEST_AUTH_HEADERS)
        assert response.status_code == 200
        assert len(collected_events) == 1

        event = collected_events[0]
        assert event.source == "api-stream"
        assert event.input.description == "A red circle appears on screen"
        assert event.input.schema_version == "8.3.0"
        assert event.performance.latency_ms >= 0

    def test_stream_telemetry_not_collected_when_disabled(self, client_with_telemetry_disabled):
        """Test that telemetry is not collected when telemetry_enabled=False."""
        client, collected_events = client_with_telemetry_disabled
        request_data = {
            "description": "A red circle appears on screen",
            "schema_version": "8.3.0",
            "telemetry_enabled": False,
        }
        response = client.post("/annotate/stream", json=request_data, headers=TEST_AUTH_HEADERS)
        assert response.status_code == 200
        assert len(collected_events) == 0

    def test_stream_telemetry_collected_on_workflow_error(self, client_with_failing_workflow):
        """Test that telemetry is collected even when workflow fails."""
        client, collected_events = client_with_failing_workflow
        request_data = {
            "description": "A red circle appears on screen",
            "schema_version": "8.3.0",
            "telemetry_enabled": True,
        }
        response = client.post("/annotate/stream", json=request_data, headers=TEST_AUTH_HEADERS)
        assert response.status_code == 200
        # Telemetry should still be collected on error
        assert len(collected_events) == 1

        event = collected_events[0]
        assert event.source == "api-stream"
        # Partial state should be captured
        assert event.output.hed_string == "Partial-annotation"

    def test_stream_telemetry_has_correct_model_info(self, client_with_telemetry):
        """Test that telemetry captures model info from headers."""
        client, collected_events = client_with_telemetry
        request_data = {
            "description": "A blue square flashes",
            "schema_version": "8.4.0",
            "telemetry_enabled": True,
            "model": "claude-haiku-4-5",
            "provider": "anthropic",
            "temperature": 0.3,
        }
        response = client.post("/annotate/stream", json=request_data, headers=TEST_AUTH_HEADERS)
        assert response.status_code == 200
        assert len(collected_events) == 1

        event = collected_events[0]
        assert event.model.model == "claude-haiku-4-5"
        assert event.model.provider == "anthropic"
        assert event.model.temperature == 0.3

    def test_stream_telemetry_result_still_sent(self, client_with_telemetry):
        """Test that result and done events are still sent with telemetry."""
        client, collected_events = client_with_telemetry
        request_data = {
            "description": "Test event",
            "schema_version": "8.3.0",
            "telemetry_enabled": True,
        }
        response = client.post("/annotate/stream", json=request_data, headers=TEST_AUTH_HEADERS)
        assert response.status_code == 200
        content = response.text
        # Result and done events should still be present
        assert "event: result" in content
        assert "event: done" in content


class TestCollectStreamTelemetryHelper:
    """Tests for the _collect_stream_telemetry helper function."""

    def test_helper_function_exists(self):
        """Test that _collect_stream_telemetry is importable."""
        from src.api.main import _collect_stream_telemetry

        assert callable(_collect_stream_telemetry)

    @pytest.mark.asyncio
    async def test_helper_skips_when_telemetry_disabled(self):
        """Test helper returns without collecting when telemetry is disabled."""
        from src.api.main import _collect_stream_telemetry
        from src.api.models import AnnotationRequest

        request = AnnotationRequest(description="Test", telemetry_enabled=False)
        mock_req = MagicMock()

        # Should not raise even with no collector
        with patch("src.api.main.telemetry_collector", None):
            await _collect_stream_telemetry(
                request=request,
                req=mock_req,
                current_state={},
                start_time=time.time(),
                source="api-stream",
                description="Test",
            )

    @pytest.mark.asyncio
    async def test_helper_skips_when_collector_is_none(self):
        """Test helper returns without collecting when collector is None."""
        from src.api.main import _collect_stream_telemetry
        from src.api.models import AnnotationRequest

        request = AnnotationRequest(description="Test", telemetry_enabled=True)
        mock_req = MagicMock()

        with patch("src.api.main.telemetry_collector", None):
            await _collect_stream_telemetry(
                request=request,
                req=mock_req,
                current_state={},
                start_time=time.time(),
                source="api-stream",
                description="Test",
            )

    @pytest.mark.asyncio
    async def test_helper_collects_when_enabled(self):
        """Test helper collects telemetry when enabled with collector."""
        from src.api.main import _collect_stream_telemetry
        from src.api.models import AnnotationRequest

        request = AnnotationRequest(
            description="Test event description",
            schema_version="8.4.0",
            telemetry_enabled=True,
            model="test/model",
            temperature=0.2,
        )

        mock_req = MagicMock()
        mock_req.headers = {}

        collected = []

        async def mock_collect(event):
            collected.append(event)
            return True

        mock_collector = MagicMock()
        mock_collector.collect = mock_collect

        with patch("src.api.main.telemetry_collector", mock_collector):
            await _collect_stream_telemetry(
                request=request,
                req=mock_req,
                current_state={
                    "current_annotation": "Sensory-event",
                    "validation_attempts": 2,
                    "validation_errors": [],
                },
                start_time=time.time() - 1.5,
                source="api-stream",
                description="Test event description",
            )

        assert len(collected) == 1
        event = collected[0]
        assert event.source == "api-stream"
        assert event.input.description == "Test event description"
        assert event.output.hed_string == "Sensory-event"
        assert event.output.iterations == 2
        assert event.performance.latency_ms >= 1400  # ~1.5 seconds


class TestModelValidationHTTP:
    """Model validation and credential errors observed through the HTTP layer.

    These exercise the endpoint dispatch wiring (ValueError -> 400,
    RuntimeError -> 503, BYOK header extraction) without any network calls:
    model rejection fires inside normalize_model before credentials are read,
    and the 503 case fires on the missing-env check before any request.
    """

    REQUEST = {"description": "A red circle appears", "schema_version": "8.4.0"}

    @pytest.mark.parametrize(
        "path,payload",
        [
            ("/annotate", REQUEST),
            ("/annotate/stream", REQUEST),
            ("/annotate-from-image", {**REQUEST, "image": "data:image/png;base64,aGk="}),
            ("/annotate-from-image/stream", {**REQUEST, "image": "data:image/png;base64,aGk="}),
        ],
    )
    def test_rejected_model_returns_400(self, client, path, payload):
        """Non-Anthropic model ids are rejected with 400 on every endpoint."""
        headers = {
            **TEST_AUTH_HEADERS,
            "X-OpenRouter-Model": "mistralai/mistral-small-3.2-24b-instruct",
        }
        response = client.post(path, json=payload, headers=headers)
        assert response.status_code == 400
        assert "not available" in response.json()["detail"]

    def test_rejected_eval_model_returns_400(self, client):
        """A non-Anthropic eval model override is rejected with 400."""
        headers = {**TEST_AUTH_HEADERS, "X-OpenRouter-Eval-Model": "qwen/qwen3.5-122b-a10b"}
        response = client.post("/annotate", json=self.REQUEST, headers=headers)
        assert response.status_code == 400
        assert "not available" in response.json()["detail"]

    def test_missing_server_credentials_return_503(self, client, monkeypatch):
        """A valid model override without ANTHROPIC_API_KEY maps to exactly 503."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        headers = {**TEST_AUTH_HEADERS, "X-OpenRouter-Model": "claude-sonnet-5"}
        response = client.post("/annotate", json=self.REQUEST, headers=headers)
        assert response.status_code == 503
        assert "ANTHROPIC_API_KEY" in response.json()["detail"]

    def test_legacy_alias_accepted_through_http(self, client, monkeypatch):
        """Legacy OpenRouter-style ids pass model validation over HTTP.

        With credentials removed, an accepted alias proceeds to the
        credential check (503); a rejected model would return 400 instead.
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        headers = {**TEST_AUTH_HEADERS, "X-OpenRouter-Model": "anthropic/claude-haiku-4.5"}
        response = client.post("/annotate", json=self.REQUEST, headers=headers)
        assert response.status_code == 503

    def test_byok_key_extracted_from_legacy_header(self, client):
        """A BYOK key sent via legacy X-OpenRouter-Key reaches the endpoint logic.

        The invalid model triggers 400 from inside create_byok_workflow; if the
        endpoint's legacy-header fallback were dropped, this would be a 401
        (Missing X-Anthropic-Key header) instead.
        """
        headers = {
            "X-OpenRouter-Key": "sk-ant-api03-validformatkey1234567890",
            "X-OpenRouter-Model": "openai/gpt-oss-120b",
        }
        response = client.post("/annotate", json=self.REQUEST, headers=headers)
        assert response.status_code == 400
        assert "not available" in response.json()["detail"]

    def test_byok_key_via_new_header(self, client):
        """Same as above through the canonical X-Anthropic-Key header."""
        headers = {
            "X-Anthropic-Key": "sk-ant-api03-validformatkey1234567890",
            "X-OpenRouter-Model": "openai/gpt-oss-120b",
        }
        response = client.post("/annotate", json=self.REQUEST, headers=headers)
        assert response.status_code == 400
        assert "not available" in response.json()["detail"]


def override_header(headers: dict[str, str], name: str) -> str | None:
    """Read an override header from a real Starlette request.

    The app is imported here rather than at module scope so the client
    fixture's auth environment is in place first.
    """
    from src.api.main import _override_header

    return _override_header(make_request(headers), name)


def make_request(headers: dict[str, str]) -> Request:
    """Build a real Starlette request carrying the given headers."""
    return Request(
        {
            "type": "http",
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        }
    )


class TestOverrideHeaders:
    """Tests for the X-Anthropic-* override headers and their legacy aliases."""

    def test_anthropic_spelling_is_read(self, client):
        assert override_header({"X-Anthropic-Model": "claude-sonnet-5"}, "model") == (
            "claude-sonnet-5"
        )

    def test_legacy_openrouter_spelling_still_works(self, client):
        assert override_header({"X-OpenRouter-Model": "claude-sonnet-5"}, "model") == (
            "claude-sonnet-5"
        )

    def test_anthropic_spelling_wins_when_both_are_sent(self, client):
        headers = {
            "X-Anthropic-Model": "claude-sonnet-5",
            "X-OpenRouter-Model": "claude-haiku-4-5",
        }
        assert override_header(headers, "model") == "claude-sonnet-5"

    def test_missing_header_is_none(self, client):
        assert override_header({}, "model") is None

    def test_all_override_names_are_supported(self, client):
        headers = {
            "X-Anthropic-Key": "sk-ant-key",
            "X-Anthropic-Eval-Model": "claude-haiku-4-5",
            "X-Anthropic-Vision-Model": "claude-haiku-4-5",
            "X-Anthropic-Temperature": "0.4",
        }
        assert override_header(headers, "key") == "sk-ant-key"
        assert override_header(headers, "eval-model") == "claude-haiku-4-5"
        assert override_header(headers, "vision-model") == "claude-haiku-4-5"
        assert override_header(headers, "temperature") == "0.4"

    def test_new_header_reaches_model_validation(self, client):
        """An unavailable model in the new header is rejected with 400.

        Model validation runs before any credential check, so this holds
        whether or not the test environment has server credentials.
        """
        response = client.post(
            "/annotate",
            json={"description": "A red circle appears", "schema_version": "8.3.0"},
            headers={**TEST_AUTH_HEADERS, "X-Anthropic-Model": "qwen/qwen3.5-122b-a10b"},
        )
        assert response.status_code == 400
        assert "not available" in response.json()["detail"]

    def test_legacy_header_reaches_model_validation(self, client):
        response = client.post(
            "/annotate",
            json={"description": "A red circle appears", "schema_version": "8.3.0"},
            headers={**TEST_AUTH_HEADERS, "X-OpenRouter-Model": "qwen/qwen3.5-122b-a10b"},
        )
        assert response.status_code == 400

    def test_both_spellings_are_advertised_for_cors(self, client):
        response = client.options(
            "/annotate",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "x-anthropic-model",
            },
        )
        allowed = response.headers.get("access-control-allow-headers", "").lower()
        assert "x-anthropic-model" in allowed
        assert "x-openrouter-model" in allowed


class TestMetricsEndpoint:
    """Tests for the server usage/savings metrics endpoint."""

    @pytest.fixture(autouse=True)
    def clean_ledger(self):
        process_ledger().reset()
        yield
        process_ledger().reset()

    def test_requires_auth(self, client):
        assert client.get("/metrics").status_code == 401

    def test_reports_totals_by_role_and_model(self, client):
        process_ledger().record(
            "annotation",
            "claude-haiku-4-5",
            {
                "input_tokens": 9000,
                "output_tokens": 200,
                "input_token_details": {"cache_read": 8000},
            },
        )
        process_ledger().record(
            "evaluation",
            "claude-haiku-4-5",
            {"input_tokens": 900, "output_tokens": 150},
        )

        response = client.get("/metrics", headers=TEST_AUTH_HEADERS)
        assert response.status_code == 200
        data = response.json()

        assert data["total"]["calls"] == 2
        assert data["total"]["cache_read_tokens"] == 8000
        assert data["total"]["savings_usd"] > 0
        assert data["by_role"]["annotation"]["cache_read_tokens"] == 8000
        assert data["by_role"]["evaluation"]["cache_read_tokens"] == 0
        assert data["by_model"]["claude-haiku-4-5"]["calls"] == 2
        assert data["since"]

    def test_empty_ledger_reports_zeros(self, client):
        response = client.get("/metrics", headers=TEST_AUTH_HEADERS)
        assert response.status_code == 200
        data = response.json()

        assert data["total"]["calls"] == 0
        assert data["total"]["cost_usd"] == 0
        assert data["by_role"] == {}

    def test_byok_callers_are_refused(self, client):
        response = client.get(
            "/metrics",
            headers={"X-Anthropic-Key": "sk-ant-api03-validlookingkey1234567890"},
        )
        assert response.status_code == 403
        assert "server API key" in response.json()["detail"]
