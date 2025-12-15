"""Tests for GitHub API client.

These tests verify the GitHub client data structures and formatting functions.
Integration tests that make real API calls are marked with @pytest.mark.integration.
"""

from src.utils.github_client import GitHubClient, GitHubItem, format_items_for_prompt


class TestGitHubItem:
    """Tests for GitHubItem dataclass."""

    def test_github_item_creation(self):
        """Test creating a GitHubItem."""
        item = GitHubItem(
            number=123,
            title="Test Issue",
            body="This is a test issue body",
            state="open",
            item_type="issue",
            labels=["bug", "priority: high"],
            url="https://github.com/test/repo/issues/123",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-02T00:00:00Z",
        )
        assert item.number == 123
        assert item.title == "Test Issue"
        assert item.item_type == "issue"
        assert len(item.labels) == 2

    def test_github_item_summary(self):
        """Test GitHubItem summary property."""
        item = GitHubItem(
            number=42,
            title="Feature Request",
            body="A long body text that should be truncated when displayed in summary",
            state="open",
            item_type="pull_request",
            labels=["enhancement"],
            url="https://github.com/test/repo/pull/42",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-02T00:00:00Z",
        )
        summary = item.summary
        assert "#42" in summary
        assert "[pull_request]" in summary
        assert "Feature Request" in summary

    def test_github_item_summary_truncation(self):
        """Test that long body is truncated in summary."""
        long_body = "x" * 600
        item = GitHubItem(
            number=1,
            title="Test",
            body=long_body,
            state="open",
            item_type="issue",
            labels=[],
            url="https://github.com/test/repo/issues/1",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-02T00:00:00Z",
        )
        summary = item.summary
        assert "..." in summary
        assert len(summary) < len(long_body)

    def test_github_item_empty_body(self):
        """Test GitHubItem with empty body."""
        item = GitHubItem(
            number=1,
            title="No Description",
            body="",
            state="open",
            item_type="issue",
            labels=[],
            url="https://github.com/test/repo/issues/1",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-02T00:00:00Z",
        )
        summary = item.summary
        assert "#1" in summary
        assert "No Description" in summary


class TestGitHubClient:
    """Tests for GitHubClient initialization."""

    def test_client_initialization(self):
        """Test client initialization with parameters."""
        client = GitHubClient(
            token="test-token",
            owner="test-owner",
            repo="test-repo",
        )
        assert client.token == "test-token"
        assert client.owner == "test-owner"
        assert client.repo == "test-repo"
        assert "Bearer test-token" in client.headers["Authorization"]

    def test_client_default_values(self):
        """Test client uses default owner and repo."""
        client = GitHubClient(token="test-token")
        assert client.owner == "Annotation-Garden"
        assert client.repo == "hedit"

    def test_client_headers(self):
        """Test client sets correct headers."""
        client = GitHubClient(token="abc123")
        assert client.headers["Accept"] == "application/vnd.github+json"
        assert client.headers["Authorization"] == "Bearer abc123"
        assert "X-GitHub-Api-Version" in client.headers


class TestFormatItemsForPrompt:
    """Tests for format_items_for_prompt function."""

    def test_format_empty_list(self):
        """Test formatting empty item list."""
        result = format_items_for_prompt([])
        assert "No open issues" in result

    def test_format_single_issue(self):
        """Test formatting a single issue."""
        items = [
            GitHubItem(
                number=10,
                title="Bug in validation",
                body="The validator returns wrong results",
                state="open",
                item_type="issue",
                labels=["bug"],
                url="https://github.com/test/repo/issues/10",
                created_at="2025-01-01T00:00:00Z",
                updated_at="2025-01-02T00:00:00Z",
            )
        ]
        result = format_items_for_prompt(items)
        assert "#10" in result
        assert "[issue]" in result
        assert "Bug in validation" in result
        assert "bug" in result

    def test_format_multiple_items(self):
        """Test formatting multiple items."""
        items = [
            GitHubItem(
                number=1,
                title="Issue One",
                body="First issue",
                state="open",
                item_type="issue",
                labels=["bug"],
                url="https://github.com/test/repo/issues/1",
                created_at="2025-01-01T00:00:00Z",
                updated_at="2025-01-02T00:00:00Z",
            ),
            GitHubItem(
                number=2,
                title="PR Two",
                body="Second item",
                state="open",
                item_type="pull_request",
                labels=["enhancement"],
                url="https://github.com/test/repo/pull/2",
                created_at="2025-01-01T00:00:00Z",
                updated_at="2025-01-02T00:00:00Z",
            ),
        ]
        result = format_items_for_prompt(items)
        assert "#1" in result
        assert "#2" in result
        assert "[issue]" in result
        assert "[pull_request]" in result

    def test_format_item_no_labels(self):
        """Test formatting item without labels."""
        items = [
            GitHubItem(
                number=1,
                title="Unlabeled Issue",
                body="No labels here",
                state="open",
                item_type="issue",
                labels=[],
                url="https://github.com/test/repo/issues/1",
                created_at="2025-01-01T00:00:00Z",
                updated_at="2025-01-02T00:00:00Z",
            )
        ]
        result = format_items_for_prompt(items)
        assert "no labels" in result

    def test_format_truncates_long_body(self):
        """Test that long body is truncated in formatted output."""
        long_body = "A" * 300
        items = [
            GitHubItem(
                number=1,
                title="Long Body Issue",
                body=long_body,
                state="open",
                item_type="issue",
                labels=[],
                url="https://github.com/test/repo/issues/1",
                created_at="2025-01-01T00:00:00Z",
                updated_at="2025-01-02T00:00:00Z",
            )
        ]
        result = format_items_for_prompt(items)
        assert "..." in result
        # Should be truncated to 200 chars
        assert "A" * 201 not in result
