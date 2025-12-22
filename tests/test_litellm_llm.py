"""Tests for LiteLLM integration with prompt caching.

These tests verify that:
1. LiteLLM wrapper creates valid LLM instances
2. Cache control is properly added to system messages
3. Real API calls work with OpenRouter (integration tests)

Run unit tests:
    uv run pytest tests/test_litellm_llm.py -v -m "not integration"

Run integration tests (requires OPENROUTER_API_KEY_FOR_TESTING):
    uv run pytest tests/test_litellm_llm.py -v -m integration
"""

import os

import pytest

# Skip all tests if litellm is not installed
pytest.importorskip("litellm")


class TestCreateLiteLLMOpenRouter:
    """Unit tests for create_litellm_openrouter function."""

    def test_creates_llm_with_default_params(self):
        """Test creating LLM with default parameters."""
        from src.utils.litellm_llm import create_litellm_openrouter

        llm = create_litellm_openrouter(api_key="test-key")

        assert llm is not None

    def test_creates_llm_with_custom_model(self):
        """Test creating LLM with custom model."""
        from src.utils.litellm_llm import create_litellm_openrouter

        llm = create_litellm_openrouter(
            model="anthropic/claude-haiku-4.5",
            api_key="test-key",
        )

        assert llm is not None

    def test_creates_llm_with_caching_enabled(self):
        """Test creating LLM with caching enabled returns wrapper."""
        from src.utils.litellm_llm import CachingLLMWrapper, create_litellm_openrouter

        llm = create_litellm_openrouter(
            model="anthropic/claude-haiku-4.5",
            api_key="test-key",
            enable_caching=True,
        )

        assert isinstance(llm, CachingLLMWrapper)

    def test_creates_llm_without_caching_returns_base(self):
        """Test creating LLM without caching returns base LLM."""
        from langchain_litellm import ChatLiteLLM

        from src.utils.litellm_llm import create_litellm_openrouter

        llm = create_litellm_openrouter(
            model="anthropic/claude-haiku-4.5",
            api_key="test-key",
            enable_caching=False,
        )

        assert isinstance(llm, ChatLiteLLM)


class TestCachingLLMWrapper:
    """Unit tests for CachingLLMWrapper."""

    def test_adds_cache_control_to_system_message(self):
        """Test that cache_control is added to system messages."""
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_litellm import ChatLiteLLM

        from src.utils.litellm_llm import CachingLLMWrapper

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

    def test_handles_ai_message(self):
        """Test that AI messages are handled correctly."""
        from langchain_core.messages import AIMessage, HumanMessage
        from langchain_litellm import ChatLiteLLM

        from src.utils.litellm_llm import CachingLLMWrapper

        base_llm = ChatLiteLLM(model="openrouter/openai/gpt-3.5-turbo", api_key="test")
        wrapper = CachingLLMWrapper(llm=base_llm)

        messages = [
            HumanMessage(content="Hello!"),
            AIMessage(content="Hi there!"),
        ]

        cached = wrapper._add_cache_control(messages)

        assert cached[0]["role"] == "user"
        assert cached[1]["role"] == "assistant"
        assert cached[1]["content"] == "Hi there!"


class TestIsCacheableModel:
    """Tests for is_cacheable_model function."""

    def test_anthropic_models_are_cacheable(self):
        """Test that Anthropic Claude models are cacheable."""
        from src.utils.litellm_llm import is_cacheable_model

        assert is_cacheable_model("anthropic/claude-haiku-4.5") is True
        assert is_cacheable_model("anthropic/claude-sonnet-4") is True
        assert is_cacheable_model("anthropic/claude-opus-4-5") is True

    def test_aliases_are_cacheable(self):
        """Test that model aliases are recognized as cacheable."""
        from src.utils.litellm_llm import is_cacheable_model

        assert is_cacheable_model("claude-haiku-4.5") is True
        assert is_cacheable_model("claude-sonnet-4.5") is True

    def test_non_anthropic_models_not_cacheable(self):
        """Test that non-Anthropic models are not cacheable."""
        from src.utils.litellm_llm import is_cacheable_model

        assert is_cacheable_model("openai/gpt-4") is False
        assert is_cacheable_model("openai/gpt-oss-120b") is False


@pytest.mark.integration
class TestLiteLLMIntegration:
    """Integration tests that make real API calls.

    These tests require OPENROUTER_API_KEY_FOR_TESTING environment variable.
    Run with: uv run pytest tests/test_litellm_llm.py -v -m integration
    """

    @pytest.fixture
    def api_key(self):
        """Get API key from environment."""
        key = os.getenv("OPENROUTER_API_KEY_FOR_TESTING")
        if not key:
            pytest.skip("OPENROUTER_API_KEY_FOR_TESTING not set")
        return key

    @pytest.mark.asyncio
    async def test_basic_completion_without_caching(self, api_key):
        """Test basic completion without caching works."""
        from langchain_core.messages import HumanMessage, SystemMessage

        from src.utils.litellm_llm import create_litellm_openrouter

        llm = create_litellm_openrouter(
            model="anthropic/claude-haiku-4.5",
            api_key=api_key,
            temperature=0.1,
            max_tokens=50,
            enable_caching=False,
        )

        messages = [
            SystemMessage(content="You are a helpful assistant. Respond briefly."),
            HumanMessage(content="Say 'Hello, World!' and nothing else."),
        ]

        response = await llm.ainvoke(messages)

        assert response is not None
        assert "Hello" in response.content or "hello" in response.content.lower()

    @pytest.mark.asyncio
    async def test_completion_with_caching_enabled(self, api_key):
        """Test completion with caching enabled works."""
        from langchain_core.messages import HumanMessage, SystemMessage

        from src.utils.litellm_llm import create_litellm_openrouter

        llm = create_litellm_openrouter(
            model="anthropic/claude-haiku-4.5",
            api_key=api_key,
            temperature=0.1,
            max_tokens=50,
            enable_caching=True,
        )

        # Use a large system prompt that qualifies for caching (>1024 tokens)
        large_system_prompt = """You are a helpful assistant specialized in HED annotations.

        """ + ("This is padding text to ensure the prompt is large enough for caching. " * 100)

        messages = [
            SystemMessage(content=large_system_prompt),
            HumanMessage(content="Say 'Cache test passed!' and nothing else."),
        ]

        response = await llm.ainvoke(messages)

        assert response is not None
        assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_cache_reduces_cost_on_second_call(self, api_key):
        """Test that making the same call twice shows caching works.

        Note: OpenRouter doesn't return cache metrics directly, but we can verify
        the call succeeds with caching enabled.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        from src.utils.litellm_llm import create_litellm_openrouter

        llm = create_litellm_openrouter(
            model="anthropic/claude-haiku-4.5",
            api_key=api_key,
            temperature=0.0,  # Deterministic for caching
            max_tokens=50,
            enable_caching=True,
        )

        # Large system prompt that qualifies for caching
        large_system_prompt = """You are a HED annotation expert.

        """ + ("Reference material for HED annotations. " * 150)

        messages = [
            SystemMessage(content=large_system_prompt),
            HumanMessage(content="Respond with: 'First call'"),
        ]

        # First call - should cache the system prompt
        response1 = await llm.ainvoke(messages)
        assert response1 is not None

        # Second call with same system prompt - should hit cache
        messages2 = [
            SystemMessage(content=large_system_prompt),
            HumanMessage(content="Respond with: 'Second call'"),
        ]

        response2 = await llm.ainvoke(messages2)
        assert response2 is not None

        # Both calls should succeed
        assert len(response1.content) > 0
        assert len(response2.content) > 0
