"""Telemetry data schema for HEDit.

Defines the structure of telemetry events collected for service improvement
and model fine-tuning.
"""

import hashlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.utils.llm_usage import UsageTotals


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
    """Performance metrics for the annotation request.

    Token and cache counters come from the LLM usage ledger, so they reflect
    every call the workflow made (annotation, evaluation, assessment,
    feedback, keyword extraction, and vision), not just the annotation call.
    ``uncached_cost_usd`` is what the same calls would have cost with no
    prompt caching, which is what makes the cache figures interpretable
    after the fact.
    """

    latency_ms: int = Field(..., description="Total request latency in milliseconds")
    input_tokens: int | None = Field(None, description="Input tokens (cached included)")
    output_tokens: int | None = Field(None, description="Number of output tokens")
    cost_usd: float | None = Field(None, description="Estimated cost in USD")
    llm_calls: int | None = Field(None, description="Number of LLM calls made")
    cache_read_tokens: int | None = Field(None, description="Input tokens served from cache")
    cache_write_tokens: int | None = Field(None, description="Input tokens written to cache")
    cache_hit_rate: float | None = Field(None, description="Cached share of input tokens")
    uncached_cost_usd: float | None = Field(
        None, description="Estimated cost the same calls would have had without caching"
    )

    @classmethod
    def from_usage(cls, latency_ms: int, usage: "UsageTotals") -> "TelemetryPerformance":
        """Build performance metrics from a usage ledger's totals.

        Args:
            latency_ms: Total request latency in milliseconds
            usage: Totals collected for this request

        Returns:
            TelemetryPerformance with token, cache, and cost figures filled in
        """
        return cls(
            latency_ms=latency_ms,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=round(usage.cost_usd, 6),
            llm_calls=usage.calls,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            cache_hit_rate=round(usage.cache_hit_rate, 4),
            uncached_cost_usd=round(usage.uncached_cost_usd, 6),
        )


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
        usage: "UsageTotals | None" = None,
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
            input_tokens: Optional token count (ignored when usage is given)
            output_tokens: Optional token count (ignored when usage is given)
            cost_usd: Optional cost estimate (ignored when usage is given)
            usage: Usage totals collected for this request; when present they
                supply the token, cache, and cost figures

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
            performance=(
                TelemetryPerformance.from_usage(latency_ms, usage)
                if usage is not None
                else TelemetryPerformance(
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                )
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
