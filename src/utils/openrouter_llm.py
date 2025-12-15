"""OpenRouter LLM integration for cloud model access."""

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI


def create_openrouter_llm(
    model: str = "openai/gpt-oss-120b",
    api_key: str | None = None,
    temperature: float = 0.1,
    max_tokens: int | None = None,
    provider: str | None = None,
) -> BaseChatModel:
    """Create an OpenRouter LLM instance.

    Args:
        model: Model identifier (e.g., "openai/gpt-oss-120b")
        api_key: OpenRouter API key
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate
        provider: Specific provider to use (e.g., "Cerebras")

    Returns:
        ChatOpenAI instance configured for OpenRouter
    """
    # OpenRouter requires these headers for app identification
    # Using default_headers (not model_kwargs) to ensure headers are sent
    default_headers = {
        "HTTP-Referer": "https://github.com/Annotation-Garden/hedit",
        "X-Title": "HEDit",
    }

    # Build extra_body for provider preference
    extra_body = None
    if provider:
        extra_body = {"provider": {"only": [provider]}}

    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=temperature,
        max_tokens=max_tokens,
        default_headers=default_headers,
        extra_body=extra_body,
    )


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
