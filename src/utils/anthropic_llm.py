"""Anthropic Claude LLM integration via the Claude Platform on AWS.

This module replaces the OpenRouter/LiteLLM integration (2026-08-18). All
LLM calls now go to Anthropic models only, through one of two endpoints:

- Server mode (default): the Claude Platform on AWS, an Anthropic-operated
  Messages API billed through AWS Marketplace (NOT Amazon Bedrock).
  Requires three environment variables; the endpoint rejects any request
  that lacks the ``anthropic-workspace-id`` header:
    ANTHROPIC_API_KEY       long-lived key from AWS Console -> Claude Platform
    ANTHROPIC_BASE_URL      e.g. https://aws-external-anthropic.us-east-2.api.aws
    ANTHROPIC_WORKSPACE_ID  workspace the key is authorized on (wrkspc_...)

- BYOK mode: a caller-supplied Anthropic API key, sent to the first-party
  API (api.anthropic.com). BYOK keys are not authorized on the AWS
  workspace, so the base URL and workspace header are intentionally NOT
  applied to them.

Prompt caching is enabled by default: system messages are transformed to
the multipart format with ``cache_control`` markers, reducing cost by up
to 90% on cache hits for the large static HED vocabulary guide.

The same wrapper reports every response's token and cache usage to
:mod:`src.utils.llm_usage`, which is where the CLI efficiency report, the
``usage`` field on API responses, and ``GET /metrics`` get their numbers.
Passing ``enable_caching=False`` returns the bare model, so that path has
neither caching nor usage accounting; production always leaves it on.
"""

import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from src.utils.llm_usage import record_usage

# Default (and judge/vision) model: fast, cheap, strong enough for HED tasks.
DEFAULT_MODEL = "claude-haiku-4-5"

# Models offered to users (web UI dropdown / CLI). Opus is deliberately
# not offered.
ALLOWED_MODELS = {
    "claude-haiku-4-5": "Claude Haiku 4.5 (fast, default)",
    "claude-sonnet-5": "Claude Sonnet 5 (highest quality)",
}

# Legacy identifiers from older clients, saved CLI configs, and cached
# frontends (OpenRouter-style ids). Normalized to first-party model ids.
MODEL_ALIASES = {
    "anthropic/claude-haiku-4.5": "claude-haiku-4-5",
    "anthropic/claude-haiku-4-5": "claude-haiku-4-5",
    "claude-haiku-4.5": "claude-haiku-4-5",
    "anthropic/claude-sonnet-5": "claude-sonnet-5",
    "anthropic/claude-sonnet-4.5": "claude-sonnet-5",
    "anthropic/claude-sonnet-4-5": "claude-sonnet-5",
}

# Extended thinking is kept OFF wherever the model allows it. With thinking
# on, the agents emit far more text and tend to circle in reasoning loops
# instead of converging on an annotation, which costs latency and tokens
# without improving the result. Where a model cannot turn thinking off, the
# closest equivalent is the lowest reasoning effort.
#
# Models where thinking runs by default (adaptive) and can be disabled
# explicitly. Haiku 4.5 does not think unless given a budget, so it needs no
# flag in either direction.
_ADAPTIVE_THINKING_MODELS = {"claude-sonnet-5"}

# Models that reject thinking={"type": "disabled"} because reasoning is
# always on. Empty today: neither offered model is in this position. The rule
# lives here so that adding such a model to ALLOWED_MODELS applies the
# lowest-effort fallback automatically instead of silently running at full
# reasoning depth.
_ALWAYS_THINKING_MODELS: set[str] = set()
LOWEST_REASONING_EFFORT = "low"

# Models that still accept sampling parameters. Sonnet 5 rejects
# `temperature` with a 400 (sampling params are removed on Claude 5 models).
_SAMPLING_MODELS = {"claude-haiku-4-5"}

# Prompt-cache lifetimes. A 5-minute entry costs 1.25x the input price to
# write, a 1-hour entry 2x; both read back at 0.1x. Which is cheaper depends
# on the traffic: back-to-back requests break even on the 5-minute entry
# after two calls, while requests spaced further apart never read a
# 5-minute entry back and pay the write premium every time. Interactive CLI
# use is the case that benefits from "1h"; a busy server does not.
CACHE_TTLS = ("5m", "1h")
DEFAULT_CACHE_TTL = "5m"


def normalize_model(model: str | None) -> str:
    """Normalize a requested model id to an offered Anthropic model.

    Args:
        model: Requested model identifier (first-party id, legacy
            OpenRouter-style id, or None for the default)

    Returns:
        A first-party Anthropic model id from ALLOWED_MODELS

    Raises:
        ValueError: If the model is not an offered Anthropic model
    """
    if not model:
        return DEFAULT_MODEL
    resolved = MODEL_ALIASES.get(model, model)
    if resolved not in ALLOWED_MODELS:
        offered = ", ".join(sorted(ALLOWED_MODELS))
        raise ValueError(f"Model '{model}' is not available. Offered models: {offered}")
    return resolved


def create_anthropic_llm(
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.1,
    max_tokens: int | None = None,
    enable_caching: bool = True,
    disable_reasoning: bool = False,
    timeout: float = 60.0,
    role: str = "unspecified",
    cache_ttl: str | None = None,
) -> BaseChatModel:
    """Create a Claude LLM instance with prompt caching.

    Args:
        model: Model identifier (default: claude-haiku-4-5). Accepts legacy
            OpenRouter-style ids (e.g. "anthropic/claude-haiku-4.5").
        api_key: BYOK Anthropic API key. When provided, requests go to the
            first-party API. When None (server mode), the key, base URL,
            and workspace id are read from the environment.
        temperature: Sampling temperature (0.0-1.0). Only sent to models
            that still accept sampling parameters (Haiku 4.5); Claude 5
            models reject it with a 400.
        max_tokens: Maximum tokens to generate (default: 8000)
        enable_caching: Add cache_control to system messages (default True)
        disable_reasoning: Turn extended thinking off. With thinking on, the
            agents produce much more output and can circle in reasoning loops
            rather than converging, so it is disabled wherever the model
            permits it (Sonnet 5 accepts thinking={"type": "disabled"}); on a
            model where reasoning is always on, the lowest reasoning effort is
            requested instead. No-op on Haiku 4.5, which does not think unless
            given a token budget.
        timeout: Per-request timeout in seconds
        role: Agent role this LLM serves ("annotation", "evaluation",
            "assessment", "feedback", "keyword", "vision"). Used to label
            token and cache usage; has no effect on the request itself.
        cache_ttl: Prompt-cache lifetime, "5m" (default) or "1h". Falls back
            to the HEDIT_PROMPT_CACHE_TTL environment variable when None.

    Returns:
        LLM instance configured for the Claude Messages API

    Raises:
        ValueError: If the model is not offered or the TTL is not supported
        RuntimeError: If server mode is used without ANTHROPIC_API_KEY set
    """
    from langchain_anthropic import ChatAnthropic

    resolved_model = normalize_model(model)
    resolved_ttl = cache_ttl or os.getenv("HEDIT_PROMPT_CACHE_TTL") or DEFAULT_CACHE_TTL
    if resolved_ttl not in CACHE_TTLS:
        raise ValueError(
            f"Unsupported prompt cache TTL '{resolved_ttl}'. Supported: {', '.join(CACHE_TTLS)}"
        )

    kwargs: dict[str, Any] = {}
    if api_key:
        # BYOK: first-party endpoint, no workspace header. The base URL must
        # be passed explicitly: ChatAnthropic otherwise falls back to the
        # process-wide ANTHROPIC_BASE_URL env var, which in server mode points
        # at the AWS endpoint that rejects keys without the workspace header.
        kwargs["api_key"] = api_key
        kwargs["base_url"] = "https://api.anthropic.com"
    else:
        server_key = os.getenv("ANTHROPIC_API_KEY")
        if not server_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (server mode requires it)")
        kwargs["api_key"] = server_key
        base_url = os.getenv("ANTHROPIC_BASE_URL")
        workspace_id = os.getenv("ANTHROPIC_WORKSPACE_ID")
        if base_url:
            kwargs["base_url"] = base_url
        if workspace_id:
            kwargs["default_headers"] = {"anthropic-workspace-id": workspace_id}

    if resolved_model in _SAMPLING_MODELS:
        kwargs["temperature"] = temperature

    if disable_reasoning:
        if resolved_model in _ADAPTIVE_THINKING_MODELS:
            kwargs["thinking"] = {"type": "disabled"}
        elif resolved_model in _ALWAYS_THINKING_MODELS:
            kwargs["output_config"] = {"effort": LOWEST_REASONING_EFFORT}

    llm = ChatAnthropic(
        model=resolved_model,
        max_tokens=max_tokens or 8000,
        timeout=timeout,
        **kwargs,
    )

    if enable_caching:
        return CachingLLMWrapper(llm=llm, role=role, cache_ttl=resolved_ttl)

    return llm


class CachingLLMWrapper(BaseChatModel):
    """Wrapper that adds cache_control to system messages for prompt caching.

    Intercepts messages before they are sent and transforms system messages
    to the multipart format with cache_control markers. Cache hits cost
    ~10% of the normal input price (after a 25% cache-write premium).

    Minimum cacheable prompt: 1024 tokens for Sonnet 5, 4096 for Haiku 4.5.
    A shorter prefix keeps the marker but never creates an entry, so no
    write premium is charged either. Only the annotation prompt (~21.8k
    tokens) clears the Haiku minimum; the evaluation, assessment, feedback,
    keyword, and vision prompts (186-623 tokens) do not.

    Cache TTL: 5 minutes by default, refreshed on each hit; "1h" is
    available for traffic spaced further apart (see CACHE_TTLS).

    Only plain-text System/Human/AI turns are supported; other message
    types (tool calls, tool results) raise TypeError rather than being
    silently relabeled, since the transformation would corrupt them.
    """

    llm: BaseChatModel
    """The underlying LLM to wrap."""

    role: str = "unspecified"
    """Agent role used to label token and cache usage."""

    cache_ttl: str = DEFAULT_CACHE_TTL
    """Prompt-cache lifetime for the system prefix ("5m" or "1h")."""

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        llm: BaseChatModel,
        role: str = "unspecified",
        cache_ttl: str = DEFAULT_CACHE_TTL,
        **kwargs,
    ) -> None:
        super().__init__(llm=llm, role=role, cache_ttl=cache_ttl, **kwargs)  # type: ignore[call-arg]

    @property
    def _llm_type(self) -> str:
        return "caching_llm_wrapper"

    def _add_cache_control(self, messages: list[BaseMessage]) -> list[dict]:
        """Transform messages to add cache_control to system messages."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        cache_control: dict[str, str] = {"type": "ephemeral"}
        if self.cache_ttl != DEFAULT_CACHE_TTL:
            cache_control["ttl"] = self.cache_ttl

        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append(
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": msg.content,
                                "cache_control": cache_control,
                            }
                        ],
                    }
                )
            elif isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    raise TypeError(
                        "CachingLLMWrapper does not support AI messages with "
                        "tool calls; disable caching for tool-calling agents"
                    )
                result.append({"role": "assistant", "content": msg.content})
            else:
                raise TypeError(
                    f"CachingLLMWrapper does not support {type(msg).__name__}; "
                    "only plain System/Human/AI messages are cacheable"
                )

        return result

    def _record_usage(self, result: Any) -> None:
        """Report one response's token and cache usage to the ledger.

        Accepts either an ``AIMessage`` (from invoke/ainvoke) or a
        ``ChatResult`` (from _generate/_agenerate). Responses without usage
        metadata are skipped rather than counted as zero.
        """
        message = result
        generations = getattr(result, "generations", None)
        if generations is not None:
            message = generations[0].message if generations else None

        usage = getattr(message, "usage_metadata", None)
        if not usage:
            return

        record_usage(role=self.role, model=getattr(self.llm, "model", ""), usage=usage)

    def _generate(  # type: ignore[override]
        self, messages: list[BaseMessage], **kwargs: Any
    ) -> Any:
        cached_messages = self._add_cache_control(messages)
        result = self.llm._generate(cached_messages, **kwargs)  # type: ignore[arg-type]
        self._record_usage(result)
        return result

    async def _agenerate(  # type: ignore[override]
        self, messages: list[BaseMessage], **kwargs: Any
    ) -> Any:
        cached_messages = self._add_cache_control(messages)
        result = await self.llm._agenerate(cached_messages, **kwargs)  # type: ignore[arg-type]
        self._record_usage(result)
        return result

    def invoke(  # type: ignore[override]
        self, messages: list[BaseMessage], **kwargs: Any
    ) -> Any:
        cached_messages = self._add_cache_control(messages)
        result = self.llm.invoke(cached_messages, **kwargs)
        self._record_usage(result)
        return result

    async def ainvoke(  # type: ignore[override]
        self, messages: list[BaseMessage], **kwargs: Any
    ) -> Any:
        cached_messages = self._add_cache_control(messages)
        result = await self.llm.ainvoke(cached_messages, **kwargs)
        self._record_usage(result)
        return result
