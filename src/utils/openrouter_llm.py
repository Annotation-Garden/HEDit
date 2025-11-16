"""OpenRouter LLM integration for cloud model access."""

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel


def create_openrouter_llm(
    model: str = "openai/gpt-5-mini",
    api_key: str | None = None,
    temperature: float = 0.1,
    max_tokens: int | None = None,
    provider: str | None = None,
) -> BaseChatModel:
    """Create an OpenRouter LLM instance.

    Args:
        model: Model identifier (e.g., "openai/gpt-5-mini", "anthropic/claude-haiku-4.5")
        api_key: OpenRouter API key
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate
        provider: Specific provider to use (e.g., "Cerebras", "SambaNova", "Groq")

    Returns:
        ChatOpenAI instance configured for OpenRouter
    """
    # Build model kwargs with extra headers
    model_kwargs = {
        "extra_headers": {
            "HTTP-Referer": "https://github.com/hed-standard/hed-bot",
            "X-Title": "HED-BOT",
        }
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
        model_kwargs=model_kwargs,
        extra_body=extra_body,
    )


# Common model configurations
OPENROUTER_MODELS = {
    # Fast, cheap models for annotation
    "gpt-5-mini": "openai/gpt-5-mini",
    "claude-haiku": "anthropic/claude-haiku-4.5",
    
    # Ultra-cheap models for feedback summarization
    "gpt-5-nano": "openai/gpt-5-nano",
    
    # Premium models for complex tasks
    "gpt-5": "openai/gpt-5",
    "claude-sonnet": "anthropic/claude-sonnet-4.5",
    "claude-opus": "anthropic/claude-opus-4.5",
}


def get_model_name(alias: str) -> str:
    """Get full model name from alias.
    
    Args:
        alias: Model alias (e.g., "gpt-5-mini")
    
    Returns:
        Full model identifier for OpenRouter
    """
    return OPENROUTER_MODELS.get(alias, alias)
