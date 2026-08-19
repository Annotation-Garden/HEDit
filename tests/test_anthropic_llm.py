"""Tests for the Anthropic LLM factory (Claude Platform on AWS)."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.utils.anthropic_llm import (
    ALLOWED_MODELS,
    DEFAULT_MODEL,
    CachingLLMWrapper,
    create_anthropic_llm,
    normalize_model,
)


class TestNormalizeModel:
    """Tests for model id normalization and validation."""

    def test_none_returns_default(self):
        assert normalize_model(None) == DEFAULT_MODEL

    def test_empty_returns_default(self):
        assert normalize_model("") == DEFAULT_MODEL

    def test_first_party_ids_pass_through(self):
        assert normalize_model("claude-haiku-4-5") == "claude-haiku-4-5"
        assert normalize_model("claude-sonnet-5") == "claude-sonnet-5"

    def test_legacy_openrouter_ids_are_aliased(self):
        assert normalize_model("anthropic/claude-haiku-4.5") == "claude-haiku-4-5"
        assert normalize_model("anthropic/claude-sonnet-5") == "claude-sonnet-5"
        assert normalize_model("claude-haiku-4.5") == "claude-haiku-4-5"

    def test_non_anthropic_models_rejected(self):
        with pytest.raises(ValueError, match="not available"):
            normalize_model("mistralai/mistral-small-3.2-24b-instruct")
        with pytest.raises(ValueError, match="not available"):
            normalize_model("openai/gpt-oss-120b")

    def test_opus_not_offered(self):
        with pytest.raises(ValueError, match="not available"):
            normalize_model("claude-opus-5")

    def test_default_is_haiku(self):
        assert DEFAULT_MODEL == "claude-haiku-4-5"
        assert DEFAULT_MODEL in ALLOWED_MODELS
        assert "claude-sonnet-5" in ALLOWED_MODELS


class TestCreateAnthropicLLM:
    """Tests for LLM construction (no network calls)."""

    def test_server_mode_uses_env_credentials(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "server-key")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://aws-external-anthropic.us-east-2.api.aws")
        monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_test")

        llm = create_anthropic_llm(enable_caching=False)
        assert llm.model == DEFAULT_MODEL
        assert llm.anthropic_api_key.get_secret_value() == "server-key"
        assert "us-east-2.api.aws" in llm.anthropic_api_url
        assert llm.default_headers == {"anthropic-workspace-id": "wrkspc_test"}

    def test_server_mode_without_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            create_anthropic_llm()

    def test_byok_mode_skips_workspace_header(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://aws-external-anthropic.us-east-2.api.aws")
        monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_test")

        llm = create_anthropic_llm(api_key="sk-ant-user-key", enable_caching=False)
        assert llm.anthropic_api_key.get_secret_value() == "sk-ant-user-key"
        # BYOK keys go to the first-party API, not the AWS workspace
        assert llm.default_headers is None

    def test_temperature_only_sent_to_sampling_models(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "server-key")

        haiku = create_anthropic_llm(
            model="claude-haiku-4-5", temperature=0.3, enable_caching=False
        )
        assert haiku.temperature == 0.3

        # Sonnet 5 rejects temperature with a 400; it must not be set
        sonnet = create_anthropic_llm(
            model="claude-sonnet-5", temperature=0.3, enable_caching=False
        )
        assert sonnet.temperature is None

    def test_disable_reasoning_on_sonnet(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "server-key")

        sonnet = create_anthropic_llm(
            model="claude-sonnet-5", disable_reasoning=True, enable_caching=False
        )
        assert sonnet.thinking == {"type": "disabled"}

        # Haiku has thinking off by default; no flag should be sent
        haiku = create_anthropic_llm(
            model="claude-haiku-4-5", disable_reasoning=True, enable_caching=False
        )
        assert haiku.thinking is None

    def test_caching_enabled_by_default(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "server-key")
        llm = create_anthropic_llm()
        assert isinstance(llm, CachingLLMWrapper)

    def test_max_tokens_default_and_override(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "server-key")
        llm = create_anthropic_llm(enable_caching=False)
        assert llm.max_tokens == 8000
        small = create_anthropic_llm(max_tokens=200, enable_caching=False)
        assert small.max_tokens == 200

    def test_invalid_model_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "server-key")
        with pytest.raises(ValueError, match="not available"):
            create_anthropic_llm(model="qwen/qwen3.5-122b-a10b")


class TestCachingLLMWrapper:
    """Tests for the cache_control message transformation."""

    @pytest.fixture
    def wrapper(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "server-key")
        llm = create_anthropic_llm()
        assert isinstance(llm, CachingLLMWrapper)
        return llm

    def test_adds_cache_control_to_system_message(self, wrapper):
        messages = [
            SystemMessage(content="Large HED vocabulary guide..."),
            HumanMessage(content="Annotate this event"),
        ]
        result = wrapper._add_cache_control(messages)

        assert result[0]["role"] == "system"
        assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert result[0]["content"][0]["text"] == "Large HED vocabulary guide..."
        assert result[1] == {"role": "user", "content": "Annotate this event"}

    def test_handles_ai_message(self, wrapper):
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
        ]
        result = wrapper._add_cache_control(messages)
        assert result[1] == {"role": "assistant", "content": "Hi there"}
