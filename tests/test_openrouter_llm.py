"""Unit tests for OpenRouter LLM utility.

Tests the LiteLLM-based implementation with optional prompt caching support.
"""

import pytest

pytest.importorskip("litellm")


class TestCreateOpenRouterLLM:
    """Tests for create_openrouter_llm function."""

    def test_creates_llm_with_default_params(self):
        """Test creating LLM with default parameters."""
        from src.utils.openrouter_llm import create_openrouter_llm

        llm = create_openrouter_llm(api_key="test-key")

        assert llm is not None

    def test_creates_llm_with_custom_model(self):
        """Test creating LLM with custom model."""
        from src.utils.openrouter_llm import create_openrouter_llm

        llm = create_openrouter_llm(model="anthropic/claude-3-haiku", api_key="test-key")

        assert llm is not None

    def test_creates_llm_with_provider(self):
        """Test creating LLM with provider preference."""
        from langchain_litellm import ChatLiteLLM

        from src.utils.openrouter_llm import create_openrouter_llm

        # Use non-Anthropic model to avoid caching wrapper
        llm = create_openrouter_llm(
            model="openai/gpt-3.5-turbo",
            api_key="test-key",
            provider="Cerebras",
            enable_caching=False,
        )

        assert llm is not None
        assert isinstance(llm, ChatLiteLLM)
        # Provider is passed in model_kwargs
        assert llm.model_kwargs is not None
        assert llm.model_kwargs.get("provider") == {"only": ["Cerebras"]}

    def test_creates_llm_with_user_id(self):
        """Test creating LLM with user_id for cache optimization."""
        from langchain_litellm import ChatLiteLLM

        from src.utils.openrouter_llm import create_openrouter_llm

        llm = create_openrouter_llm(
            model="openai/gpt-3.5-turbo",
            api_key="test-key",
            user_id="test-user-123",
            enable_caching=False,
        )

        assert llm is not None
        assert isinstance(llm, ChatLiteLLM)
        # User ID is passed in model_kwargs for sticky cache routing
        assert llm.model_kwargs is not None
        assert llm.model_kwargs.get("user") == "test-user-123"

    def test_creates_llm_with_provider_and_user_id(self):
        """Test creating LLM with both provider and user_id."""
        from langchain_litellm import ChatLiteLLM

        from src.utils.openrouter_llm import create_openrouter_llm

        llm = create_openrouter_llm(
            model="openai/gpt-3.5-turbo",
            api_key="test-key",
            provider="Cerebras",
            user_id="test-user-456",
            enable_caching=False,
        )

        assert llm is not None
        assert isinstance(llm, ChatLiteLLM)
        assert llm.model_kwargs is not None
        assert llm.model_kwargs.get("provider") == {"only": ["Cerebras"]}
        assert llm.model_kwargs.get("user") == "test-user-456"

    def test_creates_llm_with_max_tokens(self):
        """Test creating LLM with max_tokens."""
        from src.utils.openrouter_llm import create_openrouter_llm

        llm = create_openrouter_llm(
            api_key="test-key",
            max_tokens=1000,
            enable_caching=False,
        )

        assert llm.max_tokens == 1000

    def test_auto_enables_caching_for_anthropic_models(self):
        """Test that caching is auto-enabled for Anthropic models."""
        from src.utils.openrouter_llm import CachingLLMWrapper, create_openrouter_llm

        llm = create_openrouter_llm(
            model="anthropic/claude-haiku-4.5",
            api_key="test-key",
        )

        assert isinstance(llm, CachingLLMWrapper)

    def test_no_caching_for_non_anthropic_models(self):
        """Test that caching is not enabled for non-Anthropic models."""
        from langchain_litellm import ChatLiteLLM

        from src.utils.openrouter_llm import create_openrouter_llm

        llm = create_openrouter_llm(
            model="openai/gpt-oss-120b",
            api_key="test-key",
        )

        assert isinstance(llm, ChatLiteLLM)

    def test_can_force_caching_off(self):
        """Test that caching can be explicitly disabled."""
        from langchain_litellm import ChatLiteLLM

        from src.utils.openrouter_llm import create_openrouter_llm

        llm = create_openrouter_llm(
            model="anthropic/claude-haiku-4.5",
            api_key="test-key",
            enable_caching=False,
        )

        assert isinstance(llm, ChatLiteLLM)

    def test_can_force_caching_on(self):
        """Test that caching can be explicitly enabled."""
        from src.utils.openrouter_llm import CachingLLMWrapper, create_openrouter_llm

        llm = create_openrouter_llm(
            model="openai/gpt-oss-120b",
            api_key="test-key",
            enable_caching=True,
        )

        assert isinstance(llm, CachingLLMWrapper)


class TestCachingLLMWrapper:
    """Tests for CachingLLMWrapper."""

    def test_adds_cache_control_to_system_message(self):
        """Test that cache_control is added to system messages."""
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_litellm import ChatLiteLLM

        from src.utils.openrouter_llm import CachingLLMWrapper

        base_llm = ChatLiteLLM(model="openrouter/openai/gpt-3.5-turbo", api_key="test")
        wrapper = CachingLLMWrapper(llm=base_llm)

        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello!"),
        ]

        cached = wrapper._add_cache_control(messages)

        # System message should be transformed
        assert cached[0]["role"] == "system"
        assert isinstance(cached[0]["content"], list)
        assert cached[0]["content"][0]["type"] == "text"
        assert cached[0]["content"][0]["text"] == "You are a helpful assistant."
        assert cached[0]["content"][0]["cache_control"] == {"type": "ephemeral"}

        # Human message should be simple
        assert cached[1]["role"] == "user"
        assert cached[1]["content"] == "Hello!"


class TestGetModelName:
    """Tests for get_model_name function."""

    def test_get_known_model_alias(self):
        """Test getting model name for known alias."""
        from src.utils.openrouter_llm import get_model_name

        result = get_model_name("gpt-oss-120b")

        assert result == "openai/gpt-oss-120b"

    def test_get_unknown_model_returns_input(self):
        """Test getting model name for unknown alias returns input."""
        from src.utils.openrouter_llm import get_model_name

        result = get_model_name("some-unknown-model")

        assert result == "some-unknown-model"


class TestIsCacheableModel:
    """Tests for is_cacheable_model function."""

    def test_anthropic_models_are_cacheable(self):
        """Test that Anthropic Claude models are cacheable."""
        from src.utils.openrouter_llm import is_cacheable_model

        assert is_cacheable_model("anthropic/claude-haiku-4.5") is True
        assert is_cacheable_model("anthropic/claude-sonnet-4") is True
        assert is_cacheable_model("anthropic/claude-opus-4-5") is True

    def test_aliases_are_cacheable(self):
        """Test that model aliases are recognized as cacheable."""
        from src.utils.openrouter_llm import is_cacheable_model

        assert is_cacheable_model("claude-haiku-4.5") is True
        assert is_cacheable_model("claude-sonnet-4.5") is True

    def test_non_anthropic_models_not_cacheable(self):
        """Test that non-Anthropic models are not cacheable."""
        from src.utils.openrouter_llm import is_cacheable_model

        assert is_cacheable_model("openai/gpt-4") is False
        assert is_cacheable_model("openai/gpt-oss-120b") is False
