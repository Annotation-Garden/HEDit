"""Tests for the Anthropic LLM factory (Claude Platform on AWS)."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.utils.anthropic_llm import (
    ALLOWED_MODELS,
    DEFAULT_MODEL,
    CachingLLMWrapper,
    create_anthropic_llm,
    normalize_model,
)
from src.utils.llm_usage import process_ledger


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

    def test_byok_mode_uses_first_party_endpoint(self, monkeypatch):
        # Server env fully configured, as in production
        monkeypatch.setenv("ANTHROPIC_API_KEY", "server-key")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://aws-external-anthropic.us-east-2.api.aws")
        monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_test")

        llm = create_anthropic_llm(api_key="sk-ant-user-key", enable_caching=False)
        assert llm.anthropic_api_key.get_secret_value() == "sk-ant-user-key"
        # BYOK keys go to the first-party API, not the AWS workspace.
        # The URL must be pinned explicitly: ChatAnthropic otherwise inherits
        # the server's ANTHROPIC_BASE_URL from the process environment.
        assert llm.anthropic_api_url == "https://api.anthropic.com"
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

    def test_rejects_unsupported_message_types(self, wrapper):
        from langchain_core.messages import ToolMessage

        with pytest.raises(TypeError, match="ToolMessage"):
            wrapper._add_cache_control([ToolMessage(content="result", tool_call_id="t1")])

    def test_rejects_ai_message_with_tool_calls(self, wrapper):
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "f", "args": {}, "id": "t1", "type": "tool_call"}],
        )
        with pytest.raises(TypeError, match="tool calls"):
            wrapper._add_cache_control([HumanMessage(content="hi"), msg])


class TestUsageAccounting:
    """Tests for the usage the wrapper reports to the ledger.

    These exercise the extraction path with real LangChain response objects;
    the live cache-hit behavior is covered by the key-gated integration test
    in tests/test_integration_anthropic.py.
    """

    @pytest.fixture
    def wrapper(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "server-key")
        process_ledger().reset()
        return create_anthropic_llm(role="annotation")

    def test_factory_labels_the_wrapper(self, wrapper):
        assert isinstance(wrapper, CachingLLMWrapper)
        assert wrapper.role == "annotation"

    def test_default_role_is_unspecified(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "server-key")
        assert create_anthropic_llm().role == "unspecified"

    def test_records_usage_from_a_message(self, wrapper):
        message = AIMessage(
            content="Sensory-event, Visual-presentation",
            usage_metadata={
                "input_tokens": 9000,
                "output_tokens": 120,
                "total_tokens": 9120,
                "input_token_details": {"cache_read": 8500, "cache_creation": 0},
            },
        )

        wrapper._record_usage(message)

        totals = process_ledger().by_role()["annotation"]
        assert totals.calls == 1
        assert totals.cache_read_tokens == 8500
        assert totals.uncached_input_tokens == 500
        assert totals.output_tokens == 120
        # The model that served the call is attributed, not the wrapper.
        assert process_ledger().by_model()[DEFAULT_MODEL].calls == 1

    def test_records_usage_from_a_chat_result(self, wrapper):
        message = AIMessage(
            content="Agent-action",
            usage_metadata={
                "input_tokens": 4200,
                "output_tokens": 30,
                "total_tokens": 4230,
                "input_token_details": {"cache_creation": 4096},
            },
        )
        result = ChatResult(generations=[ChatGeneration(message=message)])

        wrapper._record_usage(result)

        totals = process_ledger().by_role()["annotation"]
        assert totals.cache_write_tokens == 4096
        assert totals.calls == 1

    def test_response_without_usage_is_not_counted(self, wrapper):
        wrapper._record_usage(AIMessage(content="no usage metadata"))
        assert process_ledger().is_empty()

    def test_empty_chat_result_is_not_counted(self, wrapper):
        wrapper._record_usage(ChatResult(generations=[]))
        assert process_ledger().is_empty()

    def test_uncached_llm_has_no_accounting(self, monkeypatch):
        """enable_caching=False returns the bare model, so nothing is recorded."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "server-key")
        llm = create_anthropic_llm(enable_caching=False, role="annotation")
        assert not isinstance(llm, CachingLLMWrapper)
        assert not hasattr(llm, "_record_usage")


class TestCacheTtl:
    """Tests for the prompt-cache lifetime knob."""

    @pytest.fixture(autouse=True)
    def server_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "server-key")
        monkeypatch.delenv("HEDIT_PROMPT_CACHE_TTL", raising=False)

    def test_default_ttl_sends_a_bare_marker(self):
        """The 5-minute default is implicit, so no ttl key is sent."""
        wrapper = create_anthropic_llm()
        transformed = wrapper._add_cache_control([SystemMessage(content="guide")])

        assert transformed[0]["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_one_hour_ttl_is_sent_explicitly(self):
        wrapper = create_anthropic_llm(cache_ttl="1h")
        transformed = wrapper._add_cache_control([SystemMessage(content="guide")])

        assert transformed[0]["content"][0]["cache_control"] == {
            "type": "ephemeral",
            "ttl": "1h",
        }

    def test_env_var_sets_the_default(self, monkeypatch):
        monkeypatch.setenv("HEDIT_PROMPT_CACHE_TTL", "1h")
        assert create_anthropic_llm().cache_ttl == "1h"

    def test_argument_wins_over_env_var(self, monkeypatch):
        monkeypatch.setenv("HEDIT_PROMPT_CACHE_TTL", "1h")
        assert create_anthropic_llm(cache_ttl="5m").cache_ttl == "5m"

    def test_unsupported_ttl_is_rejected(self):
        with pytest.raises(ValueError, match="Unsupported prompt cache TTL"):
            create_anthropic_llm(cache_ttl="7d")

    def test_unsupported_env_ttl_is_rejected(self, monkeypatch):
        monkeypatch.setenv("HEDIT_PROMPT_CACHE_TTL", "forever")
        with pytest.raises(ValueError, match="Unsupported prompt cache TTL"):
            create_anthropic_llm()
