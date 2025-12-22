"""LiteLLM integration for OpenRouter with prompt caching support.

This module provides LLM access through LiteLLM, which natively supports
Anthropic's prompt caching via the cache_control parameter. This reduces
costs by up to 90% for repeated prompts with large static content.

Usage:
    from src.utils.litellm_llm import create_litellm_openrouter

    # Create LLM with caching enabled
    llm = create_litellm_openrouter(
        model="anthropic/claude-haiku-4.5",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        enable_caching=True,
    )

    # Use with LangChain messages (cache_control added automatically to system)
    response = await llm.ainvoke([
        SystemMessage(content=large_system_prompt),
        HumanMessage(content=user_query),
    ])
"""

import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage


def create_litellm_openrouter(
    model: str = "openai/gpt-oss-120b",
    api_key: str | None = None,
    temperature: float = 0.1,
    max_tokens: int | None = None,
    provider: str | None = None,
    user_id: str | None = None,
    enable_caching: bool = False,
) -> BaseChatModel:
    """Create a LiteLLM-backed LLM for OpenRouter with optional prompt caching.

    Args:
        model: Model identifier (e.g., "anthropic/claude-haiku-4.5")
        api_key: OpenRouter API key
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate
        provider: Specific provider to use (e.g., "Anthropic")
        user_id: User identifier for cache optimization (sticky routing)
        enable_caching: Enable Anthropic prompt caching (default: False)

    Returns:
        ChatLiteLLM instance configured for OpenRouter
    """
    from langchain_litellm import ChatLiteLLM

    # LiteLLM uses openrouter/ prefix for OpenRouter models
    litellm_model = f"openrouter/{model}"

    # Build extra parameters
    extra_params: dict[str, Any] = {
        # OpenRouter app identification
        "extra_headers": {
            "HTTP-Referer": "https://annotation.garden/hedit",
            "X-Title": "HEDit - HED Annotation Generator",
        },
    }

    # Provider routing
    if provider:
        extra_params["provider"] = {"only": [provider]}

    # User ID for sticky cache routing
    if user_id:
        extra_params["user"] = user_id

    # Return wrapped LLM with caching if enabled
    llm = ChatLiteLLM(
        model=litellm_model,
        api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
        temperature=temperature,
        max_tokens=max_tokens,
        model_kwargs=extra_params,
    )

    if enable_caching:
        return CachingLLMWrapper(llm)

    return llm


class CachingLLMWrapper(BaseChatModel):
    """Wrapper that adds cache_control to system messages for Anthropic caching.

    This wrapper intercepts messages before they're sent to the LLM and
    transforms system messages to use the multipart format with cache_control.

    The cache_control parameter tells Anthropic to cache the content, reducing
    costs by 90% on cache hits (after initial 25% cache write premium).
    """

    llm: BaseChatModel
    """The underlying LLM to wrap."""

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, llm: BaseChatModel, **kwargs):
        super().__init__(llm=llm, **kwargs)

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

    def _generate(self, messages: list[BaseMessage], **kwargs) -> Any:
        """Generate response with cache_control on system messages."""
        cached_messages = self._add_cache_control(messages)
        return self.llm._generate(cached_messages, **kwargs)

    async def _agenerate(self, messages: list[BaseMessage], **kwargs) -> Any:
        """Async generate response with cache_control on system messages."""
        cached_messages = self._add_cache_control(messages)
        return await self.llm._agenerate(cached_messages, **kwargs)

    def invoke(self, messages: list[BaseMessage], **kwargs) -> Any:
        """Invoke LLM with cache_control on system messages."""
        cached_messages = self._add_cache_control(messages)
        return self.llm.invoke(cached_messages, **kwargs)

    async def ainvoke(self, messages: list[BaseMessage], **kwargs) -> Any:
        """Async invoke LLM with cache_control on system messages."""
        cached_messages = self._add_cache_control(messages)
        return await self.llm.ainvoke(cached_messages, **kwargs)


# Supported models that benefit from prompt caching
# Anthropic models via OpenRouter that support cache_control
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
    # Check if it's an Anthropic model
    return model.startswith("anthropic/claude-")
