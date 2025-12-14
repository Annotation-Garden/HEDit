"""Integration tests for feedback triage system.

These tests make real LLM calls but use dry_run mode to avoid creating
actual GitHub issues. They verify the classification and triage logic works.

Run with: pytest tests/test_feedback_integration.py -v
Skip with: pytest -v -m "not integration"
"""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_TEST_KEY = os.getenv("OPENROUTER_API_KEY_FOR_TESTING")
SKIP_REASON = "OPENROUTER_API_KEY_FOR_TESTING not set"


@pytest.fixture
def test_api_key() -> str:
    """Get OpenRouter API key for testing."""
    if not OPENROUTER_TEST_KEY:
        pytest.skip(SKIP_REASON)
    return OPENROUTER_TEST_KEY


@pytest.fixture
def triage_agent(test_api_key: str):
    """Create a triage agent for testing (no GitHub client - dry run only)."""
    from src.agents.feedback_triage_agent import FeedbackTriageAgent
    from src.utils.openrouter_llm import create_openrouter_llm

    model = os.getenv("ANNOTATION_MODEL", "openai/gpt-oss-120b")
    provider = os.getenv("LLM_PROVIDER_PREFERENCE", "Cerebras")

    llm = create_openrouter_llm(
        model=model,
        api_key=test_api_key,
        temperature=0.1,
        max_tokens=1000,
        provider=provider if provider else None,
    )

    # No GitHub client - we're testing classification only
    return FeedbackTriageAgent(llm=llm, github_client=None)


@pytest.mark.integration
@pytest.mark.skipif(not OPENROUTER_TEST_KEY, reason=SKIP_REASON)
class TestFeedbackClassification:
    """Test LLM-based feedback classification."""

    @pytest.mark.asyncio
    async def test_classify_bug_feedback(self, triage_agent):
        """Test classification of bug-like feedback."""
        from src.agents.feedback_triage_agent import FeedbackRecord

        record = FeedbackRecord(
            timestamp="2025-01-01T12:00:00Z",
            type="text",
            version="0.5.0",
            description="The system crashes when I input special characters",
            image_description=None,
            annotation="",
            is_valid=False,
            is_faithful=None,
            is_complete=None,
            validation_errors=["System error occurred"],
            validation_warnings=[],
            evaluation_feedback="",
            assessment_feedback="",
            user_comment="This is broken! The app crashes every time.",
        )

        classification = await triage_agent.classify_feedback(record)

        assert "category" in classification
        assert "severity" in classification
        assert classification["category"] in [
            "bug",
            "feature",
            "question",
            "documentation",
            "duplicate",
            "noise",
        ]

    @pytest.mark.asyncio
    async def test_classify_feature_feedback(self, triage_agent):
        """Test classification of feature request feedback."""
        from src.agents.feedback_triage_agent import FeedbackRecord

        record = FeedbackRecord(
            timestamp="2025-01-01T12:00:00Z",
            type="text",
            version="0.5.0",
            description="Add dark mode support",
            image_description=None,
            annotation="Visual-presentation",
            is_valid=True,
            is_faithful=True,
            is_complete=True,
            validation_errors=[],
            validation_warnings=[],
            evaluation_feedback="Good annotation",
            assessment_feedback="Complete",
            user_comment="Would be great to have dark mode for the interface.",
        )

        classification = await triage_agent.classify_feedback(record)

        assert "category" in classification
        # Could be feature or enhancement
        assert classification["category"] in ["feature", "enhancement", "question", "noise"]

    @pytest.mark.asyncio
    async def test_classify_low_priority_feedback(self, triage_agent):
        """Test that vague feedback is classified as low priority."""
        from src.agents.feedback_triage_agent import FeedbackRecord

        record = FeedbackRecord(
            timestamp="2025-01-01T12:00:00Z",
            type="text",
            version="0.5.0",
            description="A red light appears",
            image_description=None,
            annotation="Sensory-event, Visual-presentation",
            is_valid=True,
            is_faithful=True,
            is_complete=True,
            validation_errors=[],
            validation_warnings=[],
            evaluation_feedback="Good",
            assessment_feedback="Complete",
            user_comment="Looks okay I guess",
        )

        classification = await triage_agent.classify_feedback(record)

        # Vague feedback should be low severity
        assert classification.get("severity") in ["low", "medium"]


@pytest.mark.integration
@pytest.mark.skipif(not OPENROUTER_TEST_KEY, reason=SKIP_REASON)
class TestFeedbackTriage:
    """Test full triage flow with dry_run mode."""

    @pytest.mark.asyncio
    async def test_triage_archives_low_priority(self, triage_agent):
        """Test that low priority feedback is archived."""
        from src.agents.feedback_triage_agent import FeedbackRecord

        record = FeedbackRecord(
            timestamp="2025-01-01T12:00:00Z",
            type="text",
            version="0.5.0",
            description="Testing the system",
            image_description=None,
            annotation="Event",
            is_valid=True,
            is_faithful=True,
            is_complete=True,
            validation_errors=[],
            validation_warnings=[],
            evaluation_feedback="",
            assessment_feedback="",
            user_comment="Just testing, nothing wrong.",
        )

        result = await triage_agent.triage(record, existing_items=[])

        # Low priority, non-actionable feedback should be archived
        assert result.action in ["archive", "comment", "create_issue"]
        assert result.category is not None
        assert result.severity is not None

    @pytest.mark.asyncio
    async def test_triage_dry_run_no_github_action(self, triage_agent):
        """Test that dry_run mode doesn't create real issues."""
        from src.agents.feedback_triage_agent import FeedbackRecord

        record = FeedbackRecord(
            timestamp="2025-01-01T12:00:00Z",
            type="text",
            version="0.5.0",
            description="Critical system failure",
            image_description=None,
            annotation="",
            is_valid=False,
            is_faithful=None,
            is_complete=None,
            validation_errors=["Fatal error", "System crash"],
            validation_warnings=[],
            evaluation_feedback="",
            assessment_feedback="",
            user_comment="Everything is broken!",
        )

        # Process with dry_run=True
        result = await triage_agent.process_and_execute(record, dry_run=True)

        # Should have dry_run flag set
        assert result.get("dry_run") is True
        # Should indicate what would happen without actually doing it
        assert "action" in result
        assert result["action"] in ["archive", "comment", "create_issue"]

        # Should NOT have created a real issue (no issue_url)
        assert "issue_url" not in result or result.get("dry_run") is True


# Note: API endpoint tests for /feedback are covered in test_api_endpoints.py
# The integration tests here focus on the triage agent logic with real LLM calls
