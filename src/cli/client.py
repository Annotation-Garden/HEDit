"""HTTP client for HEDit API.

Handles all API communication with proper error handling and timeout management.
"""

import base64
import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import httpx

from src.cli.config import CLIConfig

# Timeout settings (in seconds)
# Annotation can take 30-60+ seconds for complex descriptions
DEFAULT_TIMEOUT = httpx.Timeout(
    connect=10.0,  # Connection timeout
    read=120.0,  # Read timeout (annotations can be slow)
    write=10.0,  # Write timeout
    pool=10.0,  # Pool timeout
)


class APIError(Exception):
    """API request error."""

    def __init__(self, message: str, status_code: int | None = None, detail: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class HEDitClient:
    """Client for HEDit API."""

    def __init__(
        self,
        api_url: str,
        api_key: str | None = None,
        model: str | None = None,
        eval_model: str | None = None,
        eval_provider: str | None = None,
        provider: str | None = None,
        temperature: float | None = None,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        user_id: str | None = None,
    ):
        """Initialize client.

        Args:
            api_url: Base API URL
            api_key: OpenRouter API key for BYOK mode
            model: Model to use for annotation
            eval_model: Model for evaluation/assessment agents (for fair benchmarking)
            eval_provider: Provider for evaluation model (e.g., Cerebras for qwen models)
            provider: Provider preference (e.g., "Cerebras")
            temperature: LLM temperature (0.0-1.0)
            timeout: Request timeout settings
            user_id: Custom user ID for cache optimization (default: derived from API key)
        """
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.eval_model = eval_model
        self.eval_provider = eval_provider
        self.provider = provider
        self.temperature = temperature
        self.timeout = timeout
        self.user_id = user_id

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with BYOK configuration."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "hedit-cli",
        }
        if self.api_key:
            # Use X-OpenRouter-Key header for BYOK mode
            headers["X-OpenRouter-Key"] = self.api_key
        # Include model configuration in headers for BYOK
        if self.model:
            headers["X-OpenRouter-Model"] = self.model
        if self.eval_model:
            headers["X-OpenRouter-Eval-Model"] = self.eval_model
        if self.eval_provider:
            headers["X-OpenRouter-Eval-Provider"] = self.eval_provider
        if self.provider:
            headers["X-OpenRouter-Provider"] = self.provider
        if self.temperature is not None:
            headers["X-OpenRouter-Temperature"] = str(self.temperature)
        # Custom user ID for cache optimization
        if self.user_id:
            headers["X-User-Id"] = self.user_id
        return headers

    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        """Handle API response and errors.

        Args:
            response: HTTP response

        Returns:
            Response JSON data

        Raises:
            APIError: If request failed
        """
        if response.status_code == 200:
            return response.json()

        # Parse error detail
        try:
            error_data = response.json()
            detail = error_data.get("detail", str(error_data))
        except Exception:
            detail = response.text

        if response.status_code == 401:
            raise APIError(
                "Authentication required",
                status_code=401,
                detail="Please provide an OpenRouter API key with --api-key or run 'hedit init'",
            )
        elif response.status_code == 422:
            raise APIError(
                "Invalid request",
                status_code=422,
                detail=detail,
            )
        elif response.status_code == 500:
            raise APIError(
                "Server error",
                status_code=500,
                detail=detail,
            )
        elif response.status_code == 503:
            raise APIError(
                "Service unavailable",
                status_code=503,
                detail="The API is temporarily unavailable. Please try again later.",
            )
        else:
            raise APIError(
                f"Request failed with status {response.status_code}",
                status_code=response.status_code,
                detail=detail,
            )

    def annotate(
        self,
        description: str,
        schema_version: str = "8.3.0",
        max_validation_attempts: int = 5,
        run_assessment: bool = False,
    ) -> dict[str, Any]:
        """Generate HED annotation from text description.

        Args:
            description: Natural language event description
            schema_version: HED schema version
            max_validation_attempts: Maximum validation retries
            run_assessment: Whether to run assessment

        Returns:
            Annotation response dictionary
        """
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.api_url}/annotate",
                headers=self._get_headers(),
                json={
                    "description": description,
                    "schema_version": schema_version,
                    "max_validation_attempts": max_validation_attempts,
                    "run_assessment": run_assessment,
                },
            )
            return self._handle_response(response)

    def annotate_stream(
        self,
        description: str,
        schema_version: str = "8.3.0",
        max_validation_attempts: int = 5,
        run_assessment: bool = False,
    ) -> Generator[tuple[str, dict[str, Any]], None, None]:
        """Generate HED annotation with streaming progress.

        Yields SSE events as (event_type, data) tuples.

        Args:
            description: Natural language event description
            schema_version: HED schema version
            max_validation_attempts: Maximum validation retries
            run_assessment: Whether to run assessment

        Yields:
            Tuple of (event_type, event_data) for each SSE event.
            Event types: "progress", "validation", "result", "error", "done"
        """
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST",
                f"{self.api_url}/annotate/stream",
                headers=self._get_headers(),
                json={
                    "description": description,
                    "schema_version": schema_version,
                    "max_validation_attempts": max_validation_attempts,
                    "run_assessment": run_assessment,
                },
            ) as response:
                if response.status_code != 200:
                    # Read full response for error
                    response.read()
                    self._handle_response(response)
                    return

                # Parse SSE stream
                current_event = None
                for line in response.iter_lines():
                    if line.startswith("event: "):
                        current_event = line[7:]
                    elif line.startswith("data: ") and current_event:
                        try:
                            data = json.loads(line[6:])
                            yield (current_event, data)
                        except json.JSONDecodeError:
                            pass  # Skip malformed data
                        current_event = None

    def annotate_image(
        self,
        image_path: Path | str,
        prompt: str | None = None,
        schema_version: str = "8.4.0",
        max_validation_attempts: int = 5,
        run_assessment: bool = False,
    ) -> dict[str, Any]:
        """Generate HED annotation from image.

        Args:
            image_path: Path to image file
            prompt: Optional custom prompt for vision model
            schema_version: HED schema version
            max_validation_attempts: Maximum validation retries
            run_assessment: Whether to run assessment

        Returns:
            Annotation response dictionary
        """
        image_uri = self._encode_image(image_path)

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.api_url}/annotate-from-image",
                headers=self._get_headers(),
                json={
                    "image": image_uri,
                    "prompt": prompt,
                    "schema_version": schema_version,
                    "max_validation_attempts": max_validation_attempts,
                    "run_assessment": run_assessment,
                },
            )
            return self._handle_response(response)

    def annotate_image_stream(
        self,
        image_path: Path | str,
        prompt: str | None = None,
        schema_version: str = "8.4.0",
        max_validation_attempts: int = 5,
        run_assessment: bool = False,
    ) -> Generator[tuple[str, dict[str, Any]], None, None]:
        """Generate HED annotation from image with streaming progress.

        Yields SSE events as (event_type, data) tuples.

        Args:
            image_path: Path to image file
            prompt: Optional custom prompt for vision model
            schema_version: HED schema version
            max_validation_attempts: Maximum validation retries
            run_assessment: Whether to run assessment

        Yields:
            Tuple of (event_type, event_data) for each SSE event.
            Event types: "progress", "image_description", "validation", "result", "error", "done"
        """
        image_uri = self._encode_image(image_path)

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST",
                f"{self.api_url}/annotate-from-image/stream",
                headers=self._get_headers(),
                json={
                    "image": image_uri,
                    "prompt": prompt,
                    "schema_version": schema_version,
                    "max_validation_attempts": max_validation_attempts,
                    "run_assessment": run_assessment,
                },
            ) as response:
                if response.status_code != 200:
                    # Read full response for error
                    response.read()
                    self._handle_response(response)
                    return

                # Parse SSE stream
                current_event = None
                for line in response.iter_lines():
                    if line.startswith("event: "):
                        current_event = line[7:]
                    elif line.startswith("data: ") and current_event:
                        try:
                            data = json.loads(line[6:])
                            yield (current_event, data)
                        except json.JSONDecodeError:
                            pass  # Skip malformed data
                        current_event = None

    def _encode_image(self, image_path: Path | str) -> str:
        """Encode an image file to base64 data URI.

        Args:
            image_path: Path to image file

        Returns:
            Base64-encoded data URI string

        Raises:
            APIError: If image file not found
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise APIError(f"Image file not found: {image_path}")

        # Detect MIME type
        suffix = image_path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_types.get(suffix, "image/png")

        # Read and encode
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        return f"data:{mime_type};base64,{image_data}"

    def validate(
        self,
        hed_string: str,
        schema_version: str = "8.3.0",
    ) -> dict[str, Any]:
        """Validate HED string.

        Args:
            hed_string: HED annotation to validate
            schema_version: HED schema version

        Returns:
            Validation response dictionary
        """
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.api_url}/validate",
                headers=self._get_headers(),
                json={
                    "hed_string": hed_string,
                    "schema_version": schema_version,
                },
            )
            return self._handle_response(response)

    def health(self) -> dict[str, Any]:
        """Check API health.

        Returns:
            Health status dictionary
        """
        with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
            response = client.get(f"{self.api_url}/health")
            return self._handle_response(response)

    def version(self) -> dict[str, Any]:
        """Get API version info.

        Returns:
            Version information dictionary
        """
        with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
            response = client.get(f"{self.api_url}/version")
            return self._handle_response(response)


def create_client(config: CLIConfig, api_key: str | None = None) -> HEDitClient:
    """Create API client from config.

    Args:
        config: CLI configuration
        api_key: API key (overrides config)

    Returns:
        Configured HEDitClient
    """
    return HEDitClient(
        api_url=config.api.url,
        api_key=api_key,
        model=config.models.default,
        provider=config.models.provider,
        temperature=config.models.temperature,
    )
