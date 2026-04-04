"""OpenRouter LLM integration for cloud model access.

This module uses LiteLLM under the hood for native Anthropic prompt caching support.
Cache control is automatically enabled for Anthropic models, reducing costs by up to
90% for repeated prompts with large static content (like the HED vocabulary guide).
"""

import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage


def create_openrouter_llm(
    model: str = "anthropic/claude-haiku-4.5",
    api_key: str | None = None,
    temperature: float = 0.1,
    max_tokens: int | None = None,
    provider: str | None = None,
    user_id: str | None = None,
    enable_caching: bool | None = None,
) -> BaseChatModel:
    """Create an OpenRouter LLM instance with optional prompt caching.

    Uses LiteLLM for native support of Anthropic's prompt caching feature.
    When caching is enabled, system messages are automatically transformed
    to include cache_control markers for 90% cost reduction on cache hits.

    Args:
        model: Model identifier (e.g., "anthropic/claude-haiku-4.5", "qwen/qwen3.5-122b-a10b")
        api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate
        provider: Specific provider to use (e.g., "Cerebras", "Anthropic")
        user_id: User identifier for cache optimization (sticky routing)
        enable_caching: Enable Anthropic prompt caching. If None (default),
            auto-enables for Anthropic Claude models.

    Returns:
        LLM instance configured for OpenRouter
    """
    from langchain_litellm import ChatLiteLLM

    # LiteLLM uses openrouter/ prefix for OpenRouter models
    litellm_model = f"openrouter/{model}"

    # Build model_kwargs for OpenRouter-specific options
    model_kwargs: dict[str, Any] = {
        # OpenRouter app identification headers
        "extra_headers": {
            "HTTP-Referer": "https://annotation.garden/hedit",
            "X-Title": "HEDit - HED Annotation Generator",
        },
    }

    # Provider routing (e.g., {"only": ["Cerebras"]})
    if provider:
        model_kwargs["provider"] = {"only": [provider]}

    # User ID for sticky cache routing
    if user_id:
        model_kwargs["user"] = user_id

    # Create base LLM with timeout to prevent hanging on slow providers
    llm = ChatLiteLLM(
        model=litellm_model,
        api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
        temperature=temperature,
        max_tokens=max_tokens,
        model_kwargs=model_kwargs,
        request_timeout=15,
    )

    # Determine if caching should be enabled
    if enable_caching is None:
        # Auto-enable for Anthropic models
        enable_caching = is_cacheable_model(model)

    if enable_caching:
        return CachingLLMWrapper(llm=llm)

    return llm


class CachingLLMWrapper(BaseChatModel):
    """Wrapper that adds cache_control to system messages for Anthropic caching.

    This wrapper intercepts messages before they're sent to the LLM and
    transforms system messages to use the multipart format with cache_control.

    The cache_control parameter tells Anthropic to cache the content, reducing
    costs by 90% on cache hits (after initial 25% cache write premium).

    Minimum cacheable prompt: 1024 tokens for Claude Sonnet/Opus, 4096 for Haiku 4.5
    Cache TTL: 5 minutes (refreshed on each hit)
    """

    llm: BaseChatModel
    """The underlying LLM to wrap."""

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, llm: BaseChatModel, **kwargs) -> None:
        super().__init__(llm=llm, **kwargs)  # type: ignore[call-arg]

    @property
    def _llm_type(self) -> str:
        return "caching_llm_wrapper"

    def _add_cache_control(self, messages: list[BaseMessage]) -> list[dict]:
        """Transform messages to add cache_control to system messages.

        Args:
            messages: List of LangChain messages

        Returns:
            List of message dicts with cache_control on system messages
        """
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                # Transform system message to multipart format with cache_control
                result.append(
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": msg.content,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                )
            elif isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
            else:
                # Fallback for other message types
                result.append({"role": "user", "content": str(msg.content)})

        return result

    def _generate(  # type: ignore[override]
        self, messages: list[BaseMessage], **kwargs: Any
    ) -> Any:
        """Generate response with cache_control on system messages."""
        cached_messages = self._add_cache_control(messages)
        return self.llm._generate(cached_messages, **kwargs)  # type: ignore[arg-type]

    async def _agenerate(  # type: ignore[override]
        self, messages: list[BaseMessage], **kwargs: Any
    ) -> Any:
        """Async generate response with cache_control on system messages."""
        cached_messages = self._add_cache_control(messages)
        return await self.llm._agenerate(cached_messages, **kwargs)  # type: ignore[arg-type]

    def invoke(  # type: ignore[override]
        self, messages: list[BaseMessage], **kwargs: Any
    ) -> Any:
        """Invoke LLM with cache_control on system messages."""
        cached_messages = self._add_cache_control(messages)
        return self.llm.invoke(cached_messages, **kwargs)

    async def ainvoke(  # type: ignore[override]
        self, messages: list[BaseMessage], **kwargs: Any
    ) -> Any:
        """Async invoke LLM with cache_control on system messages."""
        cached_messages = self._add_cache_control(messages)
        return await self.llm.ainvoke(cached_messages, **kwargs)


# Model configuration - using gpt-oss-120b via Cerebras
OPENROUTER_MODELS = {
    # Primary model for all agents (fast inference via Cerebras)
    "gpt-oss-120b": "openai/gpt-oss-120b",
}


def get_model_name(alias: str) -> str:
    """Get full model name from alias.

    Args:
        alias: Model alias (e.g., "gpt-oss-120b")

    Returns:
        Full model identifier for OpenRouter
    """
    return OPENROUTER_MODELS.get(alias, alias)


# Anthropic models that support prompt caching
CACHEABLE_MODELS = {
    "claude-opus-4.5": "anthropic/claude-opus-4-5",
    "claude-sonnet-4.5": "anthropic/claude-sonnet-4-5",
    "claude-sonnet-4": "anthropic/claude-sonnet-4",
    "claude-haiku-4.5": "anthropic/claude-haiku-4.5",
    "claude-haiku-3.5": "anthropic/claude-3.5-haiku",
}


def is_cacheable_model(model: str) -> bool:
    """Check if a model supports Anthropic prompt caching.

    Args:
        model: Model identifier

    Returns:
        True if the model supports cache_control
    """
    # Check exact match in aliases
    if model in CACHEABLE_MODELS:
        return True
    # Check if it's an Anthropic Claude model
    return model.startswith("anthropic/claude-")
