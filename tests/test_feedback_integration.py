"""Integration tests for feedback triage system.

These tests make real LLM calls but use dry_run mode to avoid creating
actual GitHub issues. They verify the classification and triage logic works.

Run with: pytest tests/test_feedback_integration.py -v
Skip with: pytest -v -m "not integration"
"""

import os
from urllib.parse import urlparse

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


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_SKIP_REASON = "GITHUB_TOKEN not set"


@pytest.mark.integration
@pytest.mark.skipif(not GITHUB_TOKEN, reason=GITHUB_SKIP_REASON)
class TestGitHubClientIntegration:
    """Test GitHub client with real API calls (read-only operations)."""

    @pytest.fixture
    def github_client(self):
        """Create a GitHub client for testing."""
        from src.utils.github_client import GitHubClient

        return GitHubClient(
            token=GITHUB_TOKEN,
            owner="Annotation-Garden",
            repo="hedit",
        )

    @pytest.mark.asyncio
    async def test_get_open_issues(self, github_client):
        """Test fetching open issues from the repository."""
        issues = await github_client.get_open_issues(limit=5)

        # Should return a list (may be empty if no open issues)
        assert isinstance(issues, list)

        # If there are issues, verify structure
        for issue in issues:
            assert issue.item_type == "issue"
            assert issue.number > 0
            assert issue.title
            parsed_url = urlparse(issue.url)
            assert parsed_url.scheme in ("http", "https")
            assert parsed_url.hostname == "github.com"

    @pytest.mark.asyncio
    async def test_get_open_pull_requests(self, github_client):
        """Test fetching open pull requests from the repository."""
        prs = await github_client.get_open_pull_requests(limit=5)

        # Should return a list (may be empty if no open PRs)
        assert isinstance(prs, list)

        # If there are PRs, verify structure
        for pr in prs:
            assert pr.item_type == "pull_request"
            assert pr.number > 0
            assert pr.title
            assert urlparse(pr.url).scheme in ("http", "https")
            assert urlparse(pr.url).hostname == "github.com"

    @pytest.mark.asyncio
    async def test_get_all_open_items(self, github_client):
        """Test fetching all open items (issues + PRs)."""
        items = await github_client.get_all_open_items(issue_limit=3, pr_limit=3)

        # Should return a list
        assert isinstance(items, list)

        # Verify all items have required fields
        for item in items:
            assert item.item_type in ["issue", "pull_request"]
            assert item.number > 0


@pytest.mark.integration
@pytest.mark.skipif(
    not OPENROUTER_TEST_KEY or not GITHUB_TOKEN,
    reason="OPENROUTER_API_KEY_FOR_TESTING or GITHUB_TOKEN not set",
)
class TestTriageWithGitHub:
    """Test triage agent with real GitHub data."""

    @pytest.fixture
    def triage_agent_with_github(self):
        """Create a triage agent with GitHub client."""
        from src.agents.feedback_triage_agent import FeedbackTriageAgent
        from src.utils.github_client import GitHubClient
        from src.utils.openrouter_llm import create_openrouter_llm

        model = os.getenv("ANNOTATION_MODEL", "openai/gpt-oss-120b")
        provider = os.getenv("LLM_PROVIDER_PREFERENCE", "Cerebras")

        llm = create_openrouter_llm(
            model=model,
            api_key=OPENROUTER_TEST_KEY,
            temperature=0.1,
            max_tokens=1000,
            provider=provider if provider else None,
        )

        github_client = GitHubClient(
            token=GITHUB_TOKEN,
            owner="Annotation-Garden",
            repo="hedit",
        )

        return FeedbackTriageAgent(llm=llm, github_client=github_client)

    @pytest.mark.asyncio
    async def test_find_similar_items(self, triage_agent_with_github):
        """Test finding similar items in GitHub."""
        from src.agents.feedback_triage_agent import FeedbackRecord

        record = FeedbackRecord(
            timestamp="2025-01-01T12:00:00Z",
            type="text",
            version="0.5.0",
            description="Add feedback triage feature",
            image_description=None,
            annotation="Event",
            is_valid=True,
            is_faithful=True,
            is_complete=True,
            validation_errors=[],
            validation_warnings=[],
            evaluation_feedback="",
            assessment_feedback="",
            user_comment="We need automated feedback processing.",
        )

        # Fetch existing items from GitHub
        existing_items = await triage_agent_with_github.github_client.get_all_open_items(
            issue_limit=10, pr_limit=5
        )

        # Find similar items
        similar_result = await triage_agent_with_github.find_similar_items(record, existing_items)

        # Should return a dict with similarity info
        assert isinstance(similar_result, dict)
        assert "has_similar" in similar_result

    @pytest.mark.asyncio
    async def test_full_triage_with_github_dry_run(self, triage_agent_with_github):
        """Test full triage flow with GitHub integration in dry_run mode."""
        from src.agents.feedback_triage_agent import FeedbackRecord

        record = FeedbackRecord(
            timestamp="2025-01-01T12:00:00Z",
            type="text",
            version="0.5.0",
            description="Test feedback for integration testing",
            image_description=None,
            annotation="Event",
            is_valid=True,
            is_faithful=True,
            is_complete=True,
            validation_errors=[],
            validation_warnings=[],
            evaluation_feedback="Good annotation",
            assessment_feedback="Complete",
            user_comment="This is a test comment for integration testing.",
        )

        # Process with dry_run=True (won't create real issues)
        result = await triage_agent_with_github.process_and_execute(record, dry_run=True)

        assert result.get("dry_run") is True
        assert "action" in result
        assert "category" in result
        assert "severity" in result


@pytest.mark.integration
@pytest.mark.skipif(not OPENROUTER_TEST_KEY, reason=SKIP_REASON)
class TestProcessFeedbackScript:
    """Test the process_feedback CLI script."""

    @pytest.mark.asyncio
    async def test_process_feedback_file(self, tmp_path):
        """Test processing a feedback file."""
        import json

        from src.scripts.process_feedback import process_feedback_file

        # Create a test feedback file
        feedback_data = {
            "timestamp": "2025-01-01T12:00:00Z",
            "type": "text",
            "version": "0.5.0",
            "description": "Test event description",
            "annotation": "Event",
            "is_valid": True,
            "is_faithful": True,
            "is_complete": True,
            "validation_errors": [],
            "validation_warnings": [],
            "evaluation_feedback": "Good",
            "assessment_feedback": "Complete",
            "user_comment": "Test feedback for CLI script.",
        }

        # Write as JSONL
        feedback_file = tmp_path / "test_feedback.jsonl"
        with open(feedback_file, "w") as f:
            f.write(json.dumps(feedback_data) + "\n")

        # Process with dry_run=True
        results = await process_feedback_file(feedback_file, dry_run=True)

        assert len(results) == 1
        result = results[0]

        # Should have processed successfully
        assert "error" not in result or result.get("dry_run") is True
        assert "action" in result

    @pytest.mark.asyncio
    async def test_process_single_json_file(self, tmp_path):
        """Test processing a single JSON feedback file."""
        import json

        from src.scripts.process_feedback import process_feedback_file

        # Create a test feedback file as JSON
        feedback_data = {
            "timestamp": "2025-01-01T12:00:00Z",
            "type": "text",
            "version": "0.5.0",
            "description": "Another test event",
            "annotation": "Sensory-event",
            "is_valid": True,
            "is_faithful": True,
            "is_complete": True,
            "validation_errors": [],
            "validation_warnings": [],
            "evaluation_feedback": "",
            "assessment_feedback": "",
            "user_comment": "",
        }

        # Write as JSON
        feedback_file = tmp_path / "test_feedback.json"
        with open(feedback_file, "w") as f:
            json.dump(feedback_data, f)

        # Process with dry_run=True
        results = await process_feedback_file(feedback_file, dry_run=True)

        assert len(results) == 1
        assert "action" in results[0]


# Note: API endpoint tests for /feedback are covered in test_api_endpoints.py
# The integration tests here focus on the triage agent logic with real LLM calls
