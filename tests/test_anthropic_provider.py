"""Phase 1 (epic #155): native Anthropic backend.

The end-to-end test makes a REAL annotation call and is skipped unless
``ANTHROPIC_API_KEY`` is set (no mocks, per repo policy). The fast unit checks
cover model-id normalisation, cache detection, and object construction without
any network access.
"""

import os

import pytest


def test_is_cacheable_model_native_claude():
    from src.utils.openrouter_llm import is_cacheable_model

    # Native ids (new) and OpenRouter-namespaced ids (existing) both cache.
    assert is_cacheable_model("claude-opus-4-8")
    assert is_cacheable_model("claude-haiku-4-5")
    assert is_cacheable_model("anthropic/claude-haiku-4.5")
    # Non-Claude models do not.
    assert not is_cacheable_model("qwen/qwen3.5-122b-a10b")


def test_to_native_anthropic_id():
    from src.utils.anthropic_llm import to_native_anthropic_id

    # Explicit native ids pass through untouched.
    assert to_native_anthropic_id("claude-opus-4-8") == "claude-opus-4-8"
    # OpenRouter-style aliases are normalised (prefix dropped, dots -> dashes).
    assert to_native_anthropic_id("anthropic/claude-haiku-4.5") == "claude-haiku-4-5"
    assert to_native_anthropic_id("claude-haiku-4.5") == "claude-haiku-4-5"
    # A bare prefix strip for anything else.
    assert to_native_anthropic_id("anthropic/claude-opus-4-8") == "claude-opus-4-8"


def test_create_anthropic_llm_builds_and_wraps_for_caching():
    from src.utils.anthropic_llm import create_anthropic_llm

    llm = create_anthropic_llm(
        model="claude-opus-4-8",
        api_key="sk-ant-test-not-a-real-key",
        base_url="https://aws-external-anthropic.us-east-2.api.aws",
        workspace_id="wrkspc_test",
    )
    # Claude models auto-enable caching -> wrapped, no network call made here.
    assert llm.__class__.__name__ == "CachingLLMWrapper"


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; skipping real Anthropic call",
)
def test_annotate_anthropic_backend_end_to_end():
    """One real annotation through the anthropic backend (AWS-billable).

    Requires ANTHROPIC_API_KEY (and ANTHROPIC_BASE_URL / ANTHROPIC_WORKSPACE_ID
    for the AWS-external endpoint) in the environment.
    """
    from src.cli.local_executor import LocalExecutionBackend

    executor = LocalExecutionBackend(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-opus-4-8",
        backend="anthropic",
    )
    result = executor.annotate(
        description="a person riding a bicycle on a city street",
        max_validation_attempts=5,
    )
    assert result["status"] == "success"
    assert result["is_valid"] is True
    assert result["hed_string"]
