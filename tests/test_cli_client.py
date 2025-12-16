"""Tests for CLI HTTP client."""

from unittest.mock import MagicMock, patch

import pytest

from src.cli.client import APIError, HEDitClient, create_client
from src.cli.config import CLIConfig


class TestAPIError:
    """Tests for APIError exception."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = APIError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.status_code is None
        assert error.detail is None

    def test_error_with_details(self):
        """Test error with status code and detail."""
        error = APIError("Auth failed", status_code=401, detail="Invalid API key")
        assert str(error) == "Auth failed"
        assert error.status_code == 401
        assert error.detail == "Invalid API key"


class TestHEDitClient:
    """Tests for HEDit API client."""

    def test_client_initialization(self):
        """Test client initializes with correct settings."""
        client = HEDitClient(
            api_url="https://api.example.com/hedit",
            api_key="test-key",
        )
        assert client.api_url == "https://api.example.com/hedit"
        assert client.api_key == "test-key"

    def test_client_initialization_with_model_settings(self):
        """Test client initializes with model settings."""
        client = HEDitClient(
            api_url="https://api.example.com/hedit",
            api_key="test-key",
            model="gpt-4o",
            provider="OpenAI",
            temperature=0.5,
        )
        assert client.model == "gpt-4o"
        assert client.provider == "OpenAI"
        assert client.temperature == 0.5

    def test_client_strips_trailing_slash(self):
        """Test client strips trailing slash from URL."""
        client = HEDitClient(api_url="https://api.example.com/hedit/")
        assert client.api_url == "https://api.example.com/hedit"

    def test_headers_with_api_key(self):
        """Test headers include API key when provided."""
        client = HEDitClient(
            api_url="https://api.example.com",
            api_key="sk-or-test-key",
        )
        headers = client._get_headers()
        assert headers["X-OpenRouter-Key"] == "sk-or-test-key"
        assert headers["Content-Type"] == "application/json"
        assert headers["User-Agent"] == "hedit-cli"

    def test_headers_with_model_settings(self):
        """Test headers include model configuration."""
        client = HEDitClient(
            api_url="https://api.example.com",
            api_key="sk-or-test-key",
            model="gpt-4o",
            provider="Cerebras",
            temperature=0.3,
        )
        headers = client._get_headers()
        assert headers["X-OpenRouter-Model"] == "gpt-4o"
        assert headers["X-OpenRouter-Provider"] == "Cerebras"
        assert headers["X-OpenRouter-Temperature"] == "0.3"

    def test_headers_without_api_key(self):
        """Test headers without API key."""
        client = HEDitClient(api_url="https://api.example.com")
        headers = client._get_headers()
        assert "X-OpenRouter-Key" not in headers
        assert headers["Content-Type"] == "application/json"


class TestClientResponseHandling:
    """Tests for response handling."""

    def test_handle_200_response(self):
        """Test successful response handling."""
        client = HEDitClient(api_url="https://api.example.com")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}

        result = client._handle_response(mock_response)
        assert result == {"result": "success"}

    def test_handle_401_response(self):
        """Test authentication error handling."""
        client = HEDitClient(api_url="https://api.example.com")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"detail": "Invalid key"}

        with pytest.raises(APIError) as exc:
            client._handle_response(mock_response)

        assert exc.value.status_code == 401
        assert "Authentication" in str(exc.value)

    def test_handle_422_response(self):
        """Test validation error handling."""
        client = HEDitClient(api_url="https://api.example.com")

        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.json.return_value = {"detail": "Invalid input"}

        with pytest.raises(APIError) as exc:
            client._handle_response(mock_response)

        assert exc.value.status_code == 422

    def test_handle_500_response(self):
        """Test server error handling."""
        client = HEDitClient(api_url="https://api.example.com")

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"detail": "Internal error"}

        with pytest.raises(APIError) as exc:
            client._handle_response(mock_response)

        assert exc.value.status_code == 500

    def test_handle_503_response(self):
        """Test service unavailable handling."""
        client = HEDitClient(api_url="https://api.example.com")

        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.json.return_value = {}

        with pytest.raises(APIError) as exc:
            client._handle_response(mock_response)

        assert exc.value.status_code == 503
        assert "unavailable" in str(exc.value).lower()


class TestClientMethods:
    """Tests for client API methods."""

    @patch("src.cli.client.httpx.Client")
    def test_annotate_request(self, mock_client_class):
        """Test annotate makes correct request."""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "annotation": "Test",
            "is_valid": True,
            "status": "success",
        }
        mock_client.post.return_value = mock_response

        # Make request
        client = HEDitClient(api_url="https://api.example.com", api_key="test-key")
        client.annotate(
            description="Test description",
            schema_version="8.4.0",
            max_validation_attempts=3,
            run_assessment=True,
        )

        # Verify request
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://api.example.com/annotate"
        assert call_args[1]["json"]["description"] == "Test description"
        assert call_args[1]["json"]["schema_version"] == "8.4.0"
        assert call_args[1]["json"]["max_validation_attempts"] == 3
        assert call_args[1]["json"]["run_assessment"] is True

    @patch("src.cli.client.httpx.Client")
    def test_validate_request(self, mock_client_class):
        """Test validate makes correct request."""
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"is_valid": True}
        mock_client.post.return_value = mock_response

        client = HEDitClient(api_url="https://api.example.com", api_key="test-key")
        client.validate(hed_string="Event", schema_version="8.3.0")

        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://api.example.com/validate"
        assert call_args[1]["json"]["hed_string"] == "Event"
        assert call_args[1]["json"]["schema_version"] == "8.3.0"

    @patch("src.cli.client.httpx.Client")
    def test_health_request(self, mock_client_class):
        """Test health check request."""
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        mock_client.get.return_value = mock_response

        client = HEDitClient(api_url="https://api.example.com")
        client.health()

        mock_client.get.assert_called_once()
        assert "health" in mock_client.get.call_args[0][0]


class TestImageAnnotation:
    """Tests for image annotation."""

    def test_annotate_image_missing_file(self):
        """Test error when image file doesn't exist."""
        client = HEDitClient(api_url="https://api.example.com", api_key="test-key")

        with pytest.raises(APIError) as exc:
            client.annotate_image(image_path="/nonexistent/image.png")

        assert "not found" in str(exc.value).lower()

    @patch("src.cli.client.httpx.Client")
    def test_annotate_image_request(self, mock_client_class, tmp_path):
        """Test image annotation request."""
        # Create a test image file
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_client = MagicMock()
        mock_client_class.return_value.__enter__.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "image_description": "A test image",
            "annotation": "Visual-presentation",
            "is_valid": True,
            "status": "success",
        }
        mock_client.post.return_value = mock_response

        client = HEDitClient(api_url="https://api.example.com", api_key="test-key")
        client.annotate_image(image_path=image_path)

        call_args = mock_client.post.call_args
        assert "annotate-from-image" in call_args[0][0]
        assert "image" in call_args[1]["json"]
        assert call_args[1]["json"]["image"].startswith("data:image/png;base64,")


class TestCreateClient:
    """Tests for create_client helper."""

    def test_create_client_from_config(self):
        """Test creating client from config."""
        config = CLIConfig()
        config.api.url = "https://custom.api.com"

        client = create_client(config, api_key="test-key")

        assert client.api_url == "https://custom.api.com"
        assert client.api_key == "test-key"

    def test_create_client_without_key(self):
        """Test creating client without API key."""
        config = CLIConfig()
        client = create_client(config)

        assert client.api_key is None
