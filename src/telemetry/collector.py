"""Telemetry collector for HEDit.

Handles filtering (model blacklist, deduplication) and event storage.
"""

from typing import Any

from src.telemetry.schema import TelemetryEvent
from src.telemetry.storage import TelemetryStorage

# Default model blacklist (models we don't want to collect telemetry for)
DEFAULT_MODEL_BLACKLIST = [
    "openai/gpt-oss-120b",  # Legacy default model (poor quality)
]


class TelemetryCollector:
    """Collects and stores telemetry events with smart filtering.

    Filters out:
    - Blacklisted models (e.g., GPT-OSS)
    - Duplicate inputs (already seen descriptions)
    """

    def __init__(
        self,
        storage: TelemetryStorage,
        enabled: bool = True,
        model_blacklist: list[str] | None = None,
    ):
        """Initialize telemetry collector.

        Args:
            storage: Storage backend for telemetry events
            enabled: Whether telemetry is enabled
            model_blacklist: List of models to exclude (uses default if None)
        """
        self.storage = storage
        self.enabled = enabled
        self.model_blacklist = set(model_blacklist or DEFAULT_MODEL_BLACKLIST)

    def should_collect(self, event: TelemetryEvent) -> bool:
        """Determine if an event should be collected.

        Applies filtering rules:
        1. Telemetry must be enabled
        2. Model must not be blacklisted
        3. Input must not be a duplicate (already in storage)

        Args:
            event: Telemetry event to evaluate

        Returns:
            True if event should be collected, False otherwise
        """
        if not self.enabled:
            return False

        # Filter: Model blacklist
        if event.model.model in self.model_blacklist:
            return False

        # Filter: Deduplication (check if input hash exists)
        if self.storage.has_input(event.input_hash):
            return False

        return True

    async def collect(self, event: TelemetryEvent) -> bool:
        """Collect a telemetry event (async).

        Applies filters and stores the event if it passes.

        Args:
            event: Telemetry event to collect

        Returns:
            True if event was collected, False if filtered out
        """
        if not self.should_collect(event):
            return False

        await self.storage.store(event)
        return True

    def collect_sync(self, event: TelemetryEvent) -> bool:
        """Collect a telemetry event (synchronous).

        Applies filters and stores the event if it passes.

        Args:
            event: Telemetry event to collect

        Returns:
            True if event was collected, False if filtered out
        """
        if not self.should_collect(event):
            return False

        self.storage.store_sync(event)
        return True

    @classmethod
    def create_event(
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
        **kwargs: Any,
    ) -> TelemetryEvent:
        """Convenience method to create a telemetry event.

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
            **kwargs: Additional optional fields

        Returns:
            TelemetryEvent instance
        """
        return TelemetryEvent.create(
            description=description,
            schema_version=schema_version,
            hed_string=hed_string,
            iterations=iterations,
            validation_errors=validation_errors,
            model=model,
            provider=provider,
            temperature=temperature,
            latency_ms=latency_ms,
            source=source,
            **kwargs,
        )
