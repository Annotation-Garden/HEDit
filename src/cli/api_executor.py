"""API execution backend for HEDit CLI.

Uses api.annotation.garden (or custom endpoint) to execute annotation workflows.
This is the default, lightweight execution mode.

Dependencies: httpx (always available)
"""

from pathlib import Path
from typing import Any

from src.cli.client import APIError, HEDitClient
from src.cli.executor import ExecutionBackend, ExecutionError


class APIExecutionBackend(ExecutionBackend):
    """Execution backend that uses the HEDit API.

    This backend sends requests to api.annotation.garden (or a custom endpoint)
    which runs the LangGraph workflow server-side. This is the default mode
    for `pip install hedit` as it requires minimal dependencies.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str | None = None,
        model: str | None = None,
        vision_model: str | None = None,
        provider: str | None = None,
        temperature: float | None = None,
    ):
        """Initialize API execution backend.

        Args:
            api_url: HEDit API endpoint URL
            api_key: OpenRouter API key for BYOK mode
            model: Model for text annotation
            vision_model: Model for image annotation
            provider: Provider preference (e.g., "Cerebras")
            temperature: LLM temperature (0.0-1.0)
        """
        self._api_url = api_url
        self._api_key = api_key
        self._model = model
        self._vision_model = vision_model
        self._provider = provider
        self._temperature = temperature

        # Create underlying HTTP client
        self._client = HEDitClient(
            api_url=api_url,
            api_key=api_key,
            model=model,
            provider=provider,
            temperature=temperature,
        )

    @property
    def mode(self) -> str:
        """Get execution mode name."""
        return "api"

    def is_available(self) -> bool:
        """Check if API backend is available.

        Always returns True since httpx is a core dependency.
        """
        return True

    def annotate(
        self,
        description: str,
        schema_version: str = "8.4.0",
        max_validation_attempts: int = 5,
        run_assessment: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate HED annotation via API."""
        try:
            return self._client.annotate(
                description=description,
                schema_version=schema_version,
                max_validation_attempts=max_validation_attempts,
                run_assessment=run_assessment,
            )
        except APIError as e:
            raise ExecutionError(
                str(e),
                code=str(e.status_code) if e.status_code else None,
                detail=e.detail,
            ) from e

    def annotate_image(
        self,
        image_path: Path | str,
        prompt: str | None = None,
        schema_version: str = "8.4.0",
        max_validation_attempts: int = 5,
        run_assessment: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate HED annotation from image via API."""
        try:
            return self._client.annotate_image(
                image_path=image_path,
                prompt=prompt,
                schema_version=schema_version,
                max_validation_attempts=max_validation_attempts,
                run_assessment=run_assessment,
            )
        except APIError as e:
            raise ExecutionError(
                str(e),
                code=str(e.status_code) if e.status_code else None,
                detail=e.detail,
            ) from e

    def validate(
        self,
        hed_string: str,
        schema_version: str = "8.4.0",
    ) -> dict[str, Any]:
        """Validate HED string via API."""
        try:
            return self._client.validate(
                hed_string=hed_string,
                schema_version=schema_version,
            )
        except APIError as e:
            raise ExecutionError(
                str(e),
                code=str(e.status_code) if e.status_code else None,
                detail=e.detail,
            ) from e

    def health(self) -> dict[str, Any]:
        """Check API health."""
        try:
            result = self._client.health()
            result["mode"] = "api"
            return result
        except APIError as e:
            raise ExecutionError(
                str(e),
                code=str(e.status_code) if e.status_code else None,
                detail=e.detail,
            ) from e
