"""Telemetry data schema for HEDit.

Defines the structure of telemetry events collected for service improvement
and model fine-tuning.
"""

import hashlib
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class TelemetryInput(BaseModel):
    """Input data for annotation request."""

    description: str = Field(..., description="Natural language event description")
    schema_version: str = Field(..., description="HED schema version used")


class TelemetryOutput(BaseModel):
    """Output data from annotation workflow."""

    hed_string: str = Field(..., description="Generated HED annotation")
    iterations: int = Field(..., description="Number of validation iterations")
    validation_errors: list[str] = Field(
        default_factory=list, description="List of validation errors (if any)"
    )


class TelemetryModel(BaseModel):
    """Model configuration used for annotation."""

    model: str = Field(..., description="Model identifier (e.g., claude-haiku-4-5)")
    provider: str | None = Field(None, description="Provider preference (if specified)")
    temperature: float = Field(..., description="Model temperature")


class TelemetryPerformance(BaseModel):
    """Performance metrics for the annotation request."""

    latency_ms: int = Field(..., description="Total request latency in milliseconds")
    input_tokens: int | None = Field(None, description="Number of input tokens")
    output_tokens: int | None = Field(None, description="Number of output tokens")
    cost_usd: float | None = Field(None, description="Estimated cost in USD")


class TelemetryEvent(BaseModel):
    """Complete telemetry event.

    This represents a single annotation request with all relevant metadata.
    """

    event_id: str = Field(default_factory=lambda: uuid4().hex, description="Unique event ID")
    input_hash: str = Field(..., description="SHA-256 hash of input description (first 16 chars)")
    session_id: str | None = Field(None, description="Ephemeral session identifier")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="Event timestamp (ISO 8601)",
    )
    input: TelemetryInput = Field(..., description="Input data")
    output: TelemetryOutput = Field(..., description="Output data")
    model: TelemetryModel = Field(..., description="Model configuration")
    performance: TelemetryPerformance = Field(..., description="Performance metrics")
    source: str = Field(..., description="Request source (cli|api|web)")

    @staticmethod
    def hash_input(description: str) -> str:
        """Generate hash of input description for deduplication.

        Args:
            description: Natural language input

        Returns:
            First 16 characters of SHA-256 hash
        """
        return hashlib.sha256(description.encode()).hexdigest()[:16]

    @classmethod
    def create(
        cls,
        description: str,
        schema_version: str,
        hed_string: str,
        iterations: int,
        validation_errors: list[str],
        model: str,
        provider: str | None,
        temperature: float,
        latency_ms: int,
        source: str,
        session_id: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: float | None = None,
    ) -> "TelemetryEvent":
        """Create a telemetry event from annotation data.

        Args:
            description: Natural language input
            schema_version: HED schema version
            hed_string: Generated HED annotation
            iterations: Number of validation iterations
            validation_errors: List of validation errors
            model: Model identifier
            provider: Provider preference
            temperature: Model temperature
            latency_ms: Request latency in milliseconds
            source: Request source (cli|api|web)
            session_id: Optional session identifier
            input_tokens: Optional token count
            output_tokens: Optional token count
            cost_usd: Optional cost estimate

        Returns:
            TelemetryEvent instance
        """
        return cls(
            input_hash=cls.hash_input(description),
            session_id=session_id,
            input=TelemetryInput(description=description, schema_version=schema_version),
            output=TelemetryOutput(
                hed_string=hed_string,
                iterations=iterations,
                validation_errors=validation_errors,
            ),
            model=TelemetryModel(model=model, provider=provider, temperature=temperature),
            performance=TelemetryPerformance(
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
            ),
            source=source,
        )

    def to_kv_key(self) -> str:
        """Generate Cloudflare KV key for this event.

        Format: telemetry:{input_hash}:{event_id}

        Returns:
            KV key string
        """
        return f"telemetry:{self.input_hash}:{self.event_id}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage.

        Returns:
            Dictionary representation
        """
        return self.model_dump()
