"""Tests for telemetry module.

Tests cover:
- Telemetry event creation and schema
- Model blacklist filtering
- Input deduplication
- Storage backends (local file, Cloudflare KV)
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.telemetry import (
    CloudflareKVStorage,
    LocalFileStorage,
    TelemetryCollector,
    TelemetryEvent,
)


class TestTelemetryEvent:
    """Tests for telemetry event creation and schema."""

    def test_create_event(self):
        """Test creating a telemetry event."""
        event = TelemetryEvent.create(
            description="participant pressed button",
            schema_version="8.4.0",
            hed_string="Agent-action, Press",
            iterations=2,
            validation_errors=[],
            model="anthropic/claude-haiku-4.5",
            provider=None,
            temperature=0.1,
            latency_ms=450,
            source="cli",
        )

        assert event.input.description == "participant pressed button"
        assert event.input.schema_version == "8.4.0"
        assert event.output.hed_string == "Agent-action, Press"
        assert event.output.iterations == 2
        assert event.model.model == "anthropic/claude-haiku-4.5"
        assert event.model.temperature == 0.1
        assert event.performance.latency_ms == 450
        assert event.source == "cli"

    def test_input_hash_generation(self):
        """Test input hash generation is consistent."""
        desc = "test description"
        hash1 = TelemetryEvent.hash_input(desc)
        hash2 = TelemetryEvent.hash_input(desc)

        assert hash1 == hash2
        assert len(hash1) == 16
        assert all(c in "0123456789abcdef" for c in hash1)

    def test_input_hash_uniqueness(self):
        """Test different inputs produce different hashes."""
        hash1 = TelemetryEvent.hash_input("description 1")
        hash2 = TelemetryEvent.hash_input("description 2")

        assert hash1 != hash2

    def test_kv_key_generation(self):
        """Test Cloudflare KV key generation."""
        event = TelemetryEvent.create(
            description="test",
            schema_version="8.4.0",
            hed_string="Test",
            iterations=1,
            validation_errors=[],
            model="test/model",
            provider=None,
            temperature=0.1,
            latency_ms=100,
            source="cli",
        )

        kv_key = event.to_kv_key()
        assert kv_key.startswith(f"telemetry:{event.input_hash}:")
        assert event.event_id in kv_key


class TestLocalFileStorage:
    """Tests for local file storage backend."""

    def test_storage_initialization(self, tmp_path):
        """Test storage initialization creates directory."""
        storage_dir = tmp_path / "telemetry"
        storage = LocalFileStorage(storage_dir=storage_dir)

        assert storage.storage_dir.exists()
        # Index file is created on first write, not initialization
        assert storage._input_hashes == set()

    def test_store_event(self, tmp_path):
        """Test storing a telemetry event."""
        storage = LocalFileStorage(storage_dir=tmp_path / "telemetry")
        event = TelemetryEvent.create(
            description="test event",
            schema_version="8.4.0",
            hed_string="Test",
            iterations=1,
            validation_errors=[],
            model="test/model",
            provider=None,
            temperature=0.1,
            latency_ms=100,
            source="cli",
        )

        storage.store_sync(event)

        # Check event file exists
        event_file = storage.storage_dir / f"{event.event_id}.json"
        assert event_file.exists()

        # Check event content
        with open(event_file) as f:
            stored_data = json.load(f)
            assert stored_data["input"]["description"] == "test event"
            assert stored_data["model"]["model"] == "test/model"

    def test_input_hash_indexing(self, tmp_path):
        """Test input hash index is maintained."""
        storage = LocalFileStorage(storage_dir=tmp_path / "telemetry")
        event = TelemetryEvent.create(
            description="indexed event",
            schema_version="8.4.0",
            hed_string="Test",
            iterations=1,
            validation_errors=[],
            model="test/model",
            provider=None,
            temperature=0.1,
            latency_ms=100,
            source="cli",
        )

        storage.store_sync(event)

        # Check hash is in index
        assert storage.has_input(event.input_hash)
        assert not storage.has_input("nonexistent_hash")

    def test_index_persistence(self, tmp_path):
        """Test index persists across storage instances."""
        storage_dir = tmp_path / "telemetry"
        storage1 = LocalFileStorage(storage_dir=storage_dir)

        event = TelemetryEvent.create(
            description="persistent test",
            schema_version="8.4.0",
            hed_string="Test",
            iterations=1,
            validation_errors=[],
            model="test/model",
            provider=None,
            temperature=0.1,
            latency_ms=100,
            source="cli",
        )

        storage1.store_sync(event)
        input_hash = event.input_hash

        # Create new storage instance
        storage2 = LocalFileStorage(storage_dir=storage_dir)
        assert storage2.has_input(input_hash)


class TestTelemetryCollector:
    """Tests for telemetry collector with filtering."""

    def test_collector_initialization(self, tmp_path):
        """Test collector initialization."""
        storage = LocalFileStorage(storage_dir=tmp_path / "telemetry")
        collector = TelemetryCollector(storage=storage, enabled=True)

        assert collector.enabled
        assert "openai/gpt-oss-120b" in collector.model_blacklist

    def test_model_blacklist_filtering(self, tmp_path):
        """Test that blacklisted models are filtered out."""
        storage = LocalFileStorage(storage_dir=tmp_path / "telemetry")
        collector = TelemetryCollector(storage=storage, enabled=True)

        # Blacklisted model
        event1 = TelemetryEvent.create(
            description="test",
            schema_version="8.4.0",
            hed_string="Test",
            iterations=1,
            validation_errors=[],
            model="openai/gpt-oss-120b",  # Blacklisted
            provider=None,
            temperature=0.1,
            latency_ms=100,
            source="cli",
        )

        # Non-blacklisted model
        event2 = TelemetryEvent.create(
            description="test",
            schema_version="8.4.0",
            hed_string="Test",
            iterations=1,
            validation_errors=[],
            model="anthropic/claude-haiku-4.5",  # Not blacklisted
            provider=None,
            temperature=0.1,
            latency_ms=100,
            source="cli",
        )

        assert not collector.should_collect(event1)
        assert collector.should_collect(event2)

    def test_deduplication_filtering(self, tmp_path):
        """Test that duplicate inputs are filtered out."""
        storage = LocalFileStorage(storage_dir=tmp_path / "telemetry")
        collector = TelemetryCollector(storage=storage, enabled=True)

        # First event
        event1 = TelemetryEvent.create(
            description="unique description",
            schema_version="8.4.0",
            hed_string="Test",
            iterations=1,
            validation_errors=[],
            model="anthropic/claude-haiku-4.5",
            provider=None,
            temperature=0.1,
            latency_ms=100,
            source="cli",
        )

        # Second event with same description
        event2 = TelemetryEvent.create(
            description="unique description",  # Same as event1
            schema_version="8.4.0",
            hed_string="Test 2",
            iterations=2,
            validation_errors=[],
            model="anthropic/claude-haiku-4.5",
            provider=None,
            temperature=0.1,
            latency_ms=200,
            source="cli",
        )

        # First event should be collected
        collected1 = collector.collect_sync(event1)
        assert collected1

        # Second event should be filtered (duplicate)
        collected2 = collector.collect_sync(event2)
        assert not collected2

    def test_telemetry_disabled(self, tmp_path):
        """Test that no events are collected when telemetry is disabled."""
        storage = LocalFileStorage(storage_dir=tmp_path / "telemetry")
        collector = TelemetryCollector(storage=storage, enabled=False)

        event = TelemetryEvent.create(
            description="test",
            schema_version="8.4.0",
            hed_string="Test",
            iterations=1,
            validation_errors=[],
            model="anthropic/claude-haiku-4.5",
            provider=None,
            temperature=0.1,
            latency_ms=100,
            source="cli",
        )

        collected = collector.collect_sync(event)
        assert not collected

    def test_custom_model_blacklist(self, tmp_path):
        """Test custom model blacklist."""
        storage = LocalFileStorage(storage_dir=tmp_path / "telemetry")
        custom_blacklist = ["custom/model-1", "custom/model-2"]
        collector = TelemetryCollector(
            storage=storage,
            enabled=True,
            model_blacklist=custom_blacklist,
        )

        event = TelemetryEvent.create(
            description="test",
            schema_version="8.4.0",
            hed_string="Test",
            iterations=1,
            validation_errors=[],
            model="custom/model-1",  # In custom blacklist
            provider=None,
            temperature=0.1,
            latency_ms=100,
            source="cli",
        )

        assert not collector.should_collect(event)


class TestTelemetryIntegration:
    """Integration tests for telemetry system."""

    def test_end_to_end_collection(self, tmp_path):
        """Test complete telemetry flow from creation to storage."""
        storage = LocalFileStorage(storage_dir=tmp_path / "telemetry")
        collector = TelemetryCollector(storage=storage, enabled=True)

        # Collect several events
        events = []
        for i in range(5):
            event = TelemetryEvent.create(
                description=f"event {i}",
                schema_version="8.4.0",
                hed_string=f"Test {i}",
                iterations=1,
                validation_errors=[],
                model="anthropic/claude-haiku-4.5",
                provider=None,
                temperature=0.1,
                latency_ms=100 + i * 10,
                source="cli",
            )
            events.append(event)
            collected = collector.collect_sync(event)
            assert collected

        # Verify all events were stored (excluding index file)
        event_files = [
            f for f in storage.storage_dir.glob("*.json") if f.name != "input_hashes.json"
        ]
        assert len(event_files) == 5

        # Verify all hashes are in index
        for event in events:
            assert storage.has_input(event.input_hash)


class TestLocalFileStorageAsync:
    """Tests for async methods of LocalFileStorage."""

    @pytest.mark.asyncio
    async def test_async_store(self, tmp_path):
        """Test async store method."""
        storage = LocalFileStorage(storage_dir=tmp_path / "telemetry")
        event = TelemetryEvent.create(
            description="async test",
            schema_version="8.4.0",
            hed_string="Test",
            iterations=1,
            validation_errors=[],
            model="test/model",
            provider=None,
            temperature=0.1,
            latency_ms=100,
            source="cli",
        )

        await storage.store(event)

        # Check event was stored
        event_file = storage.storage_dir / f"{event.event_id}.json"
        assert event_file.exists()
        assert storage.has_input(event.input_hash)


class TestLocalFileStorageErrorHandling:
    """Tests for error handling in LocalFileStorage."""

    def test_load_index_with_corrupted_json(self, tmp_path):
        """Test loading index with corrupted JSON file."""
        storage_dir = tmp_path / "telemetry"
        storage_dir.mkdir()

        # Create corrupted index file
        index_file = storage_dir / "input_hashes.json"
        index_file.write_text("not valid json{{{")

        # Should return empty set on corrupted file
        storage = LocalFileStorage(storage_dir=storage_dir)
        assert storage._input_hashes == set()

    def test_store_event_with_write_error(self, tmp_path):
        """Test storing event when write fails."""
        storage = LocalFileStorage(storage_dir=tmp_path / "telemetry")
        event = TelemetryEvent.create(
            description="error test",
            schema_version="8.4.0",
            hed_string="Test",
            iterations=1,
            validation_errors=[],
            model="test/model",
            provider=None,
            temperature=0.1,
            latency_ms=100,
            source="cli",
        )

        # Mock open to raise OSError
        with patch("builtins.open", side_effect=OSError("Write failed")):
            # Should not raise, just silently fail
            storage.store_sync(event)

        # Hash should still be added to memory (index update happens after file write)
        # But since we mocked open globally, index save also failed
        # The in-memory set should still have the hash
        assert event.input_hash in storage._input_hashes


class TestCloudflareKVStorage:
    """Tests for Cloudflare KV storage backend."""

    @pytest.fixture
    def kv_storage(self):
        """Create a CloudflareKVStorage instance."""
        return CloudflareKVStorage(
            account_id="test-account",
            namespace_id="test-namespace",
            api_token="test-token",
        )

    @pytest.fixture
    def sample_event(self):
        """Create a sample telemetry event."""
        return TelemetryEvent.create(
            description="kv test",
            schema_version="8.4.0",
            hed_string="Test",
            iterations=1,
            validation_errors=[],
            model="test/model",
            provider=None,
            temperature=0.1,
            latency_ms=100,
            source="cli",
        )

    def test_initialization(self, kv_storage):
        """Test CloudflareKVStorage initialization."""
        assert kv_storage.account_id == "test-account"
        assert kv_storage.namespace_id == "test-namespace"
        assert kv_storage.api_token == "test-token"
        assert "test-account" in kv_storage.base_url
        assert "test-namespace" in kv_storage.base_url

    def test_store_sync(self, kv_storage, sample_event):
        """Test synchronous store method."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_class.return_value.__exit__ = MagicMock(return_value=False)

            kv_storage.store_sync(sample_event)

            mock_client.put.assert_called_once()
            call_args = mock_client.put.call_args
            assert sample_event.to_kv_key() in call_args[0][0]

    @pytest.mark.asyncio
    async def test_async_store(self, kv_storage, sample_event):
        """Test async store method."""
        from unittest.mock import AsyncMock

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.put = AsyncMock(return_value=MagicMock())
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=False)

            await kv_storage.store(sample_event)

            mock_client.put.assert_called_once()

    def test_has_input_returns_true(self, kv_storage):
        """Test has_input returns True when key exists."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"result": [{"name": "key1"}]}
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_class.return_value.__exit__ = MagicMock(return_value=False)

            result = kv_storage.has_input("test_hash")

            assert result is True

    def test_has_input_returns_false(self, kv_storage):
        """Test has_input returns False when key doesn't exist."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"result": []}
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_class.return_value.__exit__ = MagicMock(return_value=False)

            result = kv_storage.has_input("nonexistent_hash")

            assert result is False

    def test_has_input_returns_false_on_error(self, kv_storage):
        """Test has_input returns False on HTTP error."""
        with patch("httpx.Client") as mock_client_class:
            import httpx

            mock_client = MagicMock()
            mock_client.get.side_effect = httpx.HTTPError("Connection failed")
            mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_class.return_value.__exit__ = MagicMock(return_value=False)

            result = kv_storage.has_input("test_hash")

            assert result is False


class TestTelemetryCollectorAsync:
    """Tests for async methods of TelemetryCollector."""

    @pytest.mark.asyncio
    async def test_async_collect(self, tmp_path):
        """Test async collect method."""
        storage = LocalFileStorage(storage_dir=tmp_path / "telemetry")
        collector = TelemetryCollector(storage=storage, enabled=True)

        event = TelemetryEvent.create(
            description="async collect test",
            schema_version="8.4.0",
            hed_string="Test",
            iterations=1,
            validation_errors=[],
            model="anthropic/claude-haiku-4.5",
            provider=None,
            temperature=0.1,
            latency_ms=100,
            source="cli",
        )

        collected = await collector.collect(event)

        assert collected is True
        assert storage.has_input(event.input_hash)

    @pytest.mark.asyncio
    async def test_async_collect_filtered(self, tmp_path):
        """Test async collect with filtered event."""
        storage = LocalFileStorage(storage_dir=tmp_path / "telemetry")
        collector = TelemetryCollector(storage=storage, enabled=False)

        event = TelemetryEvent.create(
            description="filtered test",
            schema_version="8.4.0",
            hed_string="Test",
            iterations=1,
            validation_errors=[],
            model="anthropic/claude-haiku-4.5",
            provider=None,
            temperature=0.1,
            latency_ms=100,
            source="cli",
        )

        collected = await collector.collect(event)

        assert collected is False


class TestTelemetryCollectorCreateEvent:
    """Tests for TelemetryCollector.create_event class method."""

    def test_create_event_basic(self):
        """Test creating event via collector class method."""
        event = TelemetryCollector.create_event(
            description="method test",
            schema_version="8.4.0",
            hed_string="Test",
            iterations=1,
            validation_errors=["error1"],
            model="test/model",
            provider="TestProvider",
            temperature=0.2,
            latency_ms=200,
            source="api",
        )

        assert event.input.description == "method test"
        assert event.output.hed_string == "Test"
        assert event.output.validation_errors == ["error1"]
        assert event.model.model == "test/model"
        assert event.model.provider == "TestProvider"
        assert event.source == "api"

    def test_create_event_with_kwargs(self):
        """Test creating event with additional kwargs."""
        event = TelemetryCollector.create_event(
            description="kwargs test",
            schema_version="8.4.0",
            hed_string="Test",
            iterations=1,
            validation_errors=[],
            model="test/model",
            provider=None,
            temperature=0.1,
            latency_ms=100,
            source="web",
            session_id="test-session-123",
        )

        assert event.session_id == "test-session-123"
        assert event.source == "web"


class TestTelemetryFullIntegration:
    """Full integration tests for telemetry with real file I/O."""

    def test_stored_event_can_be_read_back(self, tmp_path):
        """Test that stored events can be read back and verified."""
        storage = LocalFileStorage(storage_dir=tmp_path / "telemetry")
        collector = TelemetryCollector(storage=storage, enabled=True)

        event = TelemetryEvent.create(
            description="A red circle appears on the left side of the screen",
            schema_version="8.3.0",
            hed_string="Sensory-event, Visual-presentation, (Red, Circle)",
            iterations=3,
            validation_errors=["Warning: Consider adding temporal info"],
            model="anthropic/claude-haiku-4.5",
            provider="Cerebras",
            temperature=0.1,
            latency_ms=1250,
            source="api",
        )

        collector.collect_sync(event)

        # Read back the stored event
        event_file = storage.storage_dir / f"{event.event_id}.json"
        assert event_file.exists()

        with open(event_file) as f:
            stored_data = json.load(f)

        # Verify all fields were stored correctly
        assert stored_data["input"]["description"] == event.input.description
        assert stored_data["input"]["schema_version"] == "8.3.0"
        assert stored_data["output"]["hed_string"] == event.output.hed_string
        assert stored_data["output"]["iterations"] == 3
        assert stored_data["output"]["validation_errors"] == [
            "Warning: Consider adding temporal info"
        ]
        assert stored_data["model"]["model"] == "anthropic/claude-haiku-4.5"
        assert stored_data["model"]["provider"] == "Cerebras"
        assert stored_data["performance"]["latency_ms"] == 1250
        assert stored_data["source"] == "api"

    def test_multiple_sources_stored_correctly(self, tmp_path):
        """Test events from different sources are stored correctly."""
        storage = LocalFileStorage(storage_dir=tmp_path / "telemetry")
        collector = TelemetryCollector(storage=storage, enabled=True)

        sources = ["cli", "api", "api-image", "web"]
        stored_events = []

        for source in sources:
            event = TelemetryEvent.create(
                description=f"Event from {source}",
                schema_version="8.4.0",
                hed_string=f"Test-{source}",
                iterations=1,
                validation_errors=[],
                model="test/model",
                provider=None,
                temperature=0.1,
                latency_ms=100,
                source=source,
            )
            collector.collect_sync(event)
            stored_events.append(event)

        # Verify each event was stored with correct source
        for event in stored_events:
            event_file = storage.storage_dir / f"{event.event_id}.json"
            with open(event_file) as f:
                data = json.load(f)
            assert data["source"] == event.source

    def test_deduplication_persists_across_restarts(self, tmp_path):
        """Test that deduplication works across storage restarts."""
        storage_dir = tmp_path / "telemetry"

        # First session: store an event
        storage1 = LocalFileStorage(storage_dir=storage_dir)
        collector1 = TelemetryCollector(storage=storage1, enabled=True)

        event1 = TelemetryEvent.create(
            description="persistent dedup test",
            schema_version="8.4.0",
            hed_string="Test",
            iterations=1,
            validation_errors=[],
            model="test/model",
            provider=None,
            temperature=0.1,
            latency_ms=100,
            source="cli",
        )
        assert collector1.collect_sync(event1) is True

        # Second session: create new storage instance (simulating restart)
        storage2 = LocalFileStorage(storage_dir=storage_dir)
        collector2 = TelemetryCollector(storage=storage2, enabled=True)

        # Try to store event with same description
        event2 = TelemetryEvent.create(
            description="persistent dedup test",  # Same description
            schema_version="8.4.0",
            hed_string="Different output",
            iterations=2,
            validation_errors=[],
            model="test/model",
            provider=None,
            temperature=0.1,
            latency_ms=200,
            source="api",
        )
        # Should be rejected due to deduplication
        assert collector2.collect_sync(event2) is False

    def test_validation_errors_stored_correctly(self, tmp_path):
        """Test that validation errors are stored and retrieved correctly."""
        storage = LocalFileStorage(storage_dir=tmp_path / "telemetry")
        collector = TelemetryCollector(storage=storage, enabled=True)

        errors = [
            "[TAG_INVALID] Unknown tag 'Foo'",
            "[PARENTHESES] Unmatched parentheses",
            "[REQUIRED_TAG] Missing required tag",
        ]

        event = TelemetryEvent.create(
            description="validation error test",
            schema_version="8.3.0",
            hed_string="Invalid-tag",
            iterations=5,
            validation_errors=errors,
            model="test/model",
            provider=None,
            temperature=0.1,
            latency_ms=500,
            source="cli",
        )

        collector.collect_sync(event)

        # Read back and verify errors
        event_file = storage.storage_dir / f"{event.event_id}.json"
        with open(event_file) as f:
            data = json.load(f)

        assert data["output"]["validation_errors"] == errors
        assert len(data["output"]["validation_errors"]) == 3

    def test_kv_key_format(self):
        """Test that KV key format is correct for Cloudflare storage."""
        event = TelemetryEvent.create(
            description="kv key test",
            schema_version="8.4.0",
            hed_string="Test",
            iterations=1,
            validation_errors=[],
            model="test/model",
            provider=None,
            temperature=0.1,
            latency_ms=100,
            source="api",
        )

        kv_key = event.to_kv_key()

        # Key format: telemetry:{input_hash}:{event_id}
        assert kv_key.startswith("telemetry:")
        parts = kv_key.split(":")
        assert len(parts) == 3
        assert parts[0] == "telemetry"
        assert parts[1] == event.input_hash
        assert parts[2] == event.event_id

    def test_high_latency_events_stored(self, tmp_path):
        """Test that high latency events are stored correctly."""
        storage = LocalFileStorage(storage_dir=tmp_path / "telemetry")
        collector = TelemetryCollector(storage=storage, enabled=True)

        # Simulate a slow annotation (30 seconds)
        event = TelemetryEvent.create(
            description="slow annotation test",
            schema_version="8.4.0",
            hed_string="Test",
            iterations=10,
            validation_errors=[],
            model="slow/model",
            provider=None,
            temperature=0.1,
            latency_ms=30000,  # 30 seconds
            source="api",
        )

        collector.collect_sync(event)

        event_file = storage.storage_dir / f"{event.event_id}.json"
        with open(event_file) as f:
            data = json.load(f)

        assert data["performance"]["latency_ms"] == 30000

    @pytest.mark.asyncio
    async def test_async_full_flow(self, tmp_path):
        """Test complete async flow from creation to storage."""
        storage = LocalFileStorage(storage_dir=tmp_path / "telemetry")
        collector = TelemetryCollector(storage=storage, enabled=True)

        event = TelemetryEvent.create(
            description="async full flow test",
            schema_version="8.4.0",
            hed_string="Sensory-event, Visual-presentation",
            iterations=2,
            validation_errors=[],
            model="anthropic/claude-haiku-4.5",
            provider="Cerebras",
            temperature=0.1,
            latency_ms=800,
            source="api",
        )

        # Use async collect
        result = await collector.collect(event)
        assert result is True

        # Verify stored
        event_file = storage.storage_dir / f"{event.event_id}.json"
        assert event_file.exists()

        with open(event_file) as f:
            data = json.load(f)

        assert data["source"] == "api"
        assert data["model"]["provider"] == "Cerebras"
