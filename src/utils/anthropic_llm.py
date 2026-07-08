"""Native Anthropic LLM integration (direct Messages API, incl. AWS-external billing).

Phase 1 of epic #155: an additive backend that talks to Anthropic's native API
instead of routing through OpenRouter. It supports the AWS-billable (non-Bedrock)
endpoint via an optional base URL + workspace id, and reuses the existing
``CachingLLMWrapper`` for prompt caching on Claude models.

Credentials are read from the environment when not passed explicitly:

- ``ANTHROPIC_API_KEY``      -- API key
- ``ANTHROPIC_BASE_URL``     -- optional endpoint override (the AWS-external URL)
- ``ANTHROPIC_WORKSPACE_ID`` -- optional workspace id (sent as a request header)

The exact workspace header name for the AWS-external endpoint is confirmed by a
one-image smoke test; it is centralised in ``ANTHROPIC_WORKSPACE_HEADER`` so it
can be adjusted in one place.
"""

import os
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.utils.openrouter_llm import CachingLLMWrapper, is_cacheable_model

# Header carrying the workspace id to the AWS-external Anthropic endpoint.
# Adjust here if the smoke test shows the gateway expects a different name.
ANTHROPIC_WORKSPACE_HEADER = "anthropic-workspace-id"

# Minimal alias map so an OpenRouter-style default still resolves to a native id
# on this backend. Full per-provider model mapping is Phase 3 (#158).
_NATIVE_ALIASES = {
    "anthropic/claude-haiku-4.5": "claude-haiku-4-5",
    "claude-haiku-4.5": "claude-haiku-4-5",
    "anthropic/claude-opus-4.5": "claude-opus-4-5",
    "claude-opus-4.5": "claude-opus-4-5",
    "anthropic/claude-sonnet-4.5": "claude-sonnet-4-5",
    "claude-sonnet-4.5": "claude-sonnet-4-5",
}


def to_native_anthropic_id(model: str) -> str:
    """Normalise a model id to a native Anthropic id (no ``anthropic/`` prefix).

    Known OpenRouter-style aliases are mapped; anything else (e.g. an explicit
    ``claude-opus-4-8``) is passed through unchanged aside from stripping a
    leading ``anthropic/``.
    """
    if model in _NATIVE_ALIASES:
        return _NATIVE_ALIASES[model]
    if model.startswith("anthropic/"):
        return model[len("anthropic/") :]
    return model


def create_anthropic_llm(
    model: str = "claude-haiku-4-5",
    api_key: str | None = None,
    base_url: str | None = None,
    workspace_id: str | None = None,
    temperature: float = 0.1,
    max_tokens: int | None = None,
    enable_caching: bool | None = None,
    request_timeout: int = 120,
) -> BaseChatModel:
    """Create a native-Anthropic LLM (optionally AWS-billable) with prompt caching.

    Uses LiteLLM's ``anthropic/`` provider. Extended thinking is off by default on
    native Anthropic, so short structured roles (evaluation, keyword extraction)
    need no explicit reasoning-disable knob here.

    Args:
        model: Native Anthropic model id (e.g. ``claude-opus-4-8``,
            ``claude-haiku-4-5``). OpenRouter-style aliases are normalised.
        api_key: Anthropic API key (defaults to ``ANTHROPIC_API_KEY``).
        base_url: Endpoint override (defaults to ``ANTHROPIC_BASE_URL``); set to
            the AWS-external URL to bill usage to AWS.
        workspace_id: Workspace id (defaults to ``ANTHROPIC_WORKSPACE_ID``); sent
            as the ``ANTHROPIC_WORKSPACE_HEADER`` request header when present.
        temperature: Sampling temperature.
        max_tokens: Optional max output tokens.
        enable_caching: Force caching on/off. If None, auto-enables for Claude
            models (reuses ``CachingLLMWrapper``).
        request_timeout: Per-request timeout in seconds. More generous than the
            OpenRouter path because Opus with the large HED guide can be slow.

    Returns:
        A ``BaseChatModel`` configured for the native Anthropic API.
    """
    from langchain_litellm import ChatLiteLLM

    native_model = to_native_anthropic_id(model)
    litellm_model = f"anthropic/{native_model}"

    resolved_base = base_url or os.getenv("ANTHROPIC_BASE_URL")
    resolved_workspace = workspace_id or os.getenv("ANTHROPIC_WORKSPACE_ID")

    model_kwargs: dict[str, Any] = {}
    # LiteLLM's Anthropic provider reads ``ANTHROPIC_API_BASE`` (not the SDK's
    # ``ANTHROPIC_BASE_URL``), so pass the endpoint explicitly rather than relying
    # on the env-var name matching.
    if resolved_base:
        model_kwargs["api_base"] = resolved_base
    if resolved_workspace:
        model_kwargs["extra_headers"] = {ANTHROPIC_WORKSPACE_HEADER: resolved_workspace}

    llm = ChatLiteLLM(
        model=litellm_model,
        api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
        temperature=temperature,
        max_tokens=max_tokens,
        model_kwargs=model_kwargs,
        request_timeout=request_timeout,
    )

    if enable_caching is None:
        enable_caching = is_cacheable_model(native_model)

    if enable_caching:
        return CachingLLMWrapper(llm=llm)

    return llm
