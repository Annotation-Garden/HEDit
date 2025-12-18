"""Abstract execution backend for HEDit CLI.

Defines the interface for different execution modes:
- API mode: Uses api.annotation.garden (default, lightweight)
- Standalone mode: Runs LangGraph workflow locally (requires hedit[standalone])
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ExecutionBackend(ABC):
    """Abstract base class for CLI execution backends.

    All execution backends must implement these methods to provide
    a consistent interface regardless of whether execution happens
    via API or locally.
    """

    @abstractmethod
    def annotate(
        self,
        description: str,
        schema_version: str = "8.4.0",
        max_validation_attempts: int = 5,
        run_assessment: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate HED annotation from text description.

        Args:
            description: Natural language event description
            schema_version: HED schema version (e.g., "8.4.0")
            max_validation_attempts: Maximum validation retries
            run_assessment: Whether to run completeness assessment
            **kwargs: Additional backend-specific options

        Returns:
            Annotation result dictionary with keys:
                - status: "success" or "error"
                - hed_string: Generated HED annotation
                - is_valid: Whether annotation is valid
                - validation_messages: List of validation messages
                - metadata: Additional metadata
        """
        pass

    @abstractmethod
    def annotate_image(
        self,
        image_path: Path | str,
        prompt: str | None = None,
        schema_version: str = "8.4.0",
        max_validation_attempts: int = 5,
        run_assessment: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate HED annotation from image.

        Args:
            image_path: Path to image file (PNG, JPG, etc.)
            prompt: Optional custom prompt for vision model
            schema_version: HED schema version
            max_validation_attempts: Maximum validation retries
            run_assessment: Whether to run completeness assessment
            **kwargs: Additional backend-specific options

        Returns:
            Annotation result dictionary with keys:
                - status: "success" or "error"
                - description: Generated image description
                - hed_string: Generated HED annotation
                - is_valid: Whether annotation is valid
                - validation_messages: List of validation messages
                - metadata: Additional metadata
        """
        pass

    @abstractmethod
    def validate(
        self,
        hed_string: str,
        schema_version: str = "8.4.0",
    ) -> dict[str, Any]:
        """Validate HED string.

        Args:
            hed_string: HED annotation to validate
            schema_version: HED schema version

        Returns:
            Validation result dictionary with keys:
                - is_valid: Whether string is valid
                - messages: List of validation messages
        """
        pass

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """Check backend health/availability.

        Returns:
            Health status dictionary with keys:
                - status: "healthy" or "unhealthy"
                - version: Backend version
                - mode: "api" or "standalone"
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available (dependencies installed).

        Returns:
            True if backend can be used, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def mode(self) -> str:
        """Get the execution mode name.

        Returns:
            "api" or "standalone"
        """
        pass


class ExecutionError(Exception):
    """Execution backend error."""

    def __init__(
        self,
        message: str,
        code: str | None = None,
        detail: str | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.detail = detail
