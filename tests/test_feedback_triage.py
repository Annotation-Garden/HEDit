"""Tests for feedback triage agent.

These tests verify the feedback record parsing, file loading, and saving functions.
Integration tests that make real LLM calls are marked with @pytest.mark.integration.
"""

import json
from urllib.parse import urlparse

from src.agents.feedback_triage_agent import (
    FeedbackRecord,
    TriageResult,
    load_feedback_file,
    save_processed_feedback,
)
from src.utils.github_client import GitHubItem


class TestFeedbackRecord:
    """Tests for FeedbackRecord dataclass."""

    def test_from_json_text_mode(self):
        """Test creating FeedbackRecord from text mode JSON."""
        data = {
            "timestamp": "2025-01-01T12:00:00Z",
            "type": "text",
            "version": "0.5.0",
            "description": "A red light flashes",
            "annotation": "Sensory-event, Visual-presentation",
            "is_valid": True,
            "is_faithful": True,
            "is_complete": True,
            "validation_errors": [],
            "validation_warnings": [],
            "evaluation_feedback": "Good annotation",
            "assessment_feedback": "Complete",
            "user_comment": "Works great!",
        }
        record = FeedbackRecord.from_json(data)

        assert record.type == "text"
        assert record.description == "A red light flashes"
        assert record.annotation == "Sensory-event, Visual-presentation"
        assert record.is_valid is True
        assert record.user_comment == "Works great!"

    def test_from_json_image_mode(self):
        """Test creating FeedbackRecord from image mode JSON."""
        data = {
            "timestamp": "2025-01-01T12:00:00Z",
            "type": "image",
            "version": "0.5.0",
            "image_description": "A person standing in a field",
            "annotation": "Agent-action, Visual-presentation",
            "is_valid": False,
            "validation_errors": ["Invalid tag"],
        }
        record = FeedbackRecord.from_json(data)

        assert record.type == "image"
        assert record.image_description == "A person standing in a field"
        assert record.is_valid is False
        assert len(record.validation_errors) == 1

    def test_from_json_minimal(self):
        """Test creating FeedbackRecord with minimal data."""
        data = {
            "annotation": "Event",
        }
        record = FeedbackRecord.from_json(data)

        assert record.type == "text"
        assert record.annotation == "Event"
        assert record.is_valid is False
        assert record.validation_errors == []
        assert record.user_comment is None

    def test_to_summary(self):
        """Test generating summary from FeedbackRecord."""
        record = FeedbackRecord(
            timestamp="2025-01-01T12:00:00Z",
            type="text",
            version="0.5.0",
            description="A button press",
            image_description=None,
            annotation="Agent-action, (Press, Button)",
            is_valid=True,
            is_faithful=True,
            is_complete=True,
            validation_errors=[],
            validation_warnings=[],
            evaluation_feedback="Good",
            assessment_feedback="Complete",
            user_comment="The annotation is correct",
        )
        summary = record.to_summary()

        assert "text" in summary.lower()
        assert "0.5.0" in summary
        assert "A button press" in summary
        assert "Agent-action" in summary
        assert "valid" in summary.lower()
        assert "The annotation is correct" in summary

    def test_to_summary_with_errors(self):
        """Test summary includes validation errors."""
        record = FeedbackRecord(
            timestamp="2025-01-01T12:00:00Z",
            type="text",
            version="0.5.0",
            description="Test",
            image_description=None,
            annotation="Invalid-tag",
            is_valid=False,
            is_faithful=None,
            is_complete=None,
            validation_errors=["Error 1", "Error 2", "Error 3", "Error 4"],
            validation_warnings=[],
            evaluation_feedback="",
            assessment_feedback="",
            user_comment=None,
        )
        summary = record.to_summary()

        assert "invalid" in summary.lower()
        assert "Validation Errors" in summary
        # Should only show first 3 errors
        assert "Error 1" in summary or "Error 2" in summary


class TestTriageResult:
    """Tests for TriageResult dataclass."""

    def test_triage_result_comment(self):
        """Test TriageResult for comment action."""
        similar_item = GitHubItem(
            number=42,
            title="Existing Issue",
            body="Similar issue",
            state="open",
            item_type="issue",
            labels=["bug"],
            url="https://github.com/test/repo/issues/42",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-02T00:00:00Z",
        )
        result = TriageResult(
            action="comment",
            reason="Similar to existing issue",
            category="duplicate",
            severity="medium",
            similar_item=similar_item,
            issue_title=None,
            issue_body=None,
            labels=["validation"],
        )

        assert result.action == "comment"
        assert result.similar_item is not None
        assert result.similar_item.number == 42

    def test_triage_result_create_issue(self):
        """Test TriageResult for create_issue action."""
        result = TriageResult(
            action="create_issue",
            reason="Novel bug with high severity",
            category="bug",
            severity="high",
            similar_item=None,
            issue_title="[Bug] Validation returns wrong result",
            issue_body="## Description\n\nValidation issue...",
            labels=["type: bug", "priority: high"],
        )

        assert result.action == "create_issue"
        assert result.issue_title is not None
        assert "Bug" in result.issue_title
        assert len(result.labels) == 2

    def test_triage_result_archive(self):
        """Test TriageResult for archive action."""
        result = TriageResult(
            action="archive",
            reason="Low priority question",
            category="question",
            severity="low",
            similar_item=None,
            issue_title=None,
            issue_body=None,
            labels=[],
        )

        assert result.action == "archive"
        assert result.severity == "low"

    def test_triage_result_to_dict(self):
        """Test converting TriageResult to dict."""
        result = TriageResult(
            action="create_issue",
            reason="New feature request",
            category="feature",
            severity="medium",
            similar_item=None,
            issue_title="Add dark mode",
            issue_body="## Feature\n\nPlease add dark mode",
            labels=["enhancement"],
        )
        data = result.to_dict()

        assert data["action"] == "create_issue"
        assert data["reason"] == "New feature request"
        assert data["category"] == "feature"
        assert data["issue_title"] == "Add dark mode"
        assert "similar_item_number" not in data

    def test_triage_result_to_dict_with_similar_item(self):
        """Test converting TriageResult with similar item to dict."""
        similar_item = GitHubItem(
            number=10,
            title="Related",
            body="",
            state="open",
            item_type="issue",
            labels=[],
            url="https://github.com/test/repo/issues/10",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-02T00:00:00Z",
        )
        result = TriageResult(
            action="comment",
            reason="Related",
            category="duplicate",
            severity="low",
            similar_item=similar_item,
            issue_title=None,
            issue_body=None,
            labels=[],
        )
        data = result.to_dict()

        assert data["similar_item_number"] == 10
        assert urlparse(data["similar_item_url"]).hostname == "github.com"


class TestLoadFeedbackFile:
    """Tests for load_feedback_file function."""

    def test_load_single_record(self, tmp_path):
        """Test loading a JSONL file with single record."""
        feedback_file = tmp_path / "feedback.jsonl"
        record = {
            "timestamp": "2025-01-01T00:00:00Z",
            "type": "text",
            "version": "0.5.0",
            "description": "Test description",
            "annotation": "Event",
        }
        feedback_file.write_text(json.dumps(record) + "\n")

        records = load_feedback_file(feedback_file)

        assert len(records) == 1
        assert records[0].description == "Test description"

    def test_load_multiple_records(self, tmp_path):
        """Test loading a JSONL file with multiple records."""
        feedback_file = tmp_path / "feedback.jsonl"
        records = [
            {"annotation": "Event1", "type": "text"},
            {"annotation": "Event2", "type": "image"},
            {"annotation": "Event3", "type": "text"},
        ]
        content = "\n".join(json.dumps(r) for r in records) + "\n"
        feedback_file.write_text(content)

        loaded = load_feedback_file(feedback_file)

        assert len(loaded) == 3
        assert loaded[0].annotation == "Event1"
        assert loaded[1].type == "image"
        assert loaded[2].annotation == "Event3"

    def test_load_empty_lines_ignored(self, tmp_path):
        """Test that empty lines are ignored."""
        feedback_file = tmp_path / "feedback.jsonl"
        content = json.dumps({"annotation": "Event"}) + "\n\n\n"
        feedback_file.write_text(content)

        records = load_feedback_file(feedback_file)

        assert len(records) == 1


class TestSaveProcessedFeedback:
    """Tests for save_processed_feedback function."""

    def test_save_creates_file(self, tmp_path):
        """Test that save creates a file in the output directory."""
        record = FeedbackRecord(
            timestamp="2025-01-01T12:00:00Z",
            type="text",
            version="0.5.0",
            description="Test",
            image_description=None,
            annotation="Event",
            is_valid=True,
            is_faithful=True,
            is_complete=True,
            validation_errors=[],
            validation_warnings=[],
            evaluation_feedback="",
            assessment_feedback="",
            user_comment=None,
        )
        result = {"action": "archive", "reason": "Test"}

        output_path = save_processed_feedback(record, result, tmp_path)

        assert output_path.exists()
        assert output_path.suffix == ".json"

    def test_save_creates_output_dir(self, tmp_path):
        """Test that save creates output directory if it doesn't exist."""
        record = FeedbackRecord(
            timestamp="2025-01-01T12:00:00Z",
            type="text",
            version="0.5.0",
            description="Test",
            image_description=None,
            annotation="Event",
            is_valid=True,
            is_faithful=None,
            is_complete=None,
            validation_errors=[],
            validation_warnings=[],
            evaluation_feedback="",
            assessment_feedback="",
            user_comment=None,
        )
        result = {"action": "create_issue"}
        new_dir = tmp_path / "nested" / "output"

        output_path = save_processed_feedback(record, result, new_dir)

        assert new_dir.exists()
        assert output_path.exists()

    def test_save_content_structure(self, tmp_path):
        """Test that saved content has correct structure."""
        record = FeedbackRecord(
            timestamp="2025-01-01T12:00:00Z",
            type="text",
            version="0.5.0",
            description="Original description",
            image_description=None,
            annotation="Sensory-event",
            is_valid=True,
            is_faithful=True,
            is_complete=True,
            validation_errors=[],
            validation_warnings=[],
            evaluation_feedback="Good",
            assessment_feedback="Complete",
            user_comment="Nice!",
        )
        result = {"action": "archive", "reason": "Low priority"}

        output_path = save_processed_feedback(record, result, tmp_path)

        with open(output_path) as f:
            saved = json.load(f)

        assert "original_feedback" in saved
        assert "processing_result" in saved
        assert "processed_at" in saved

        assert saved["original_feedback"]["description"] == "Original description"
        assert saved["original_feedback"]["annotation"] == "Sensory-event"
        assert saved["original_feedback"]["user_comment"] == "Nice!"
        assert saved["processing_result"]["action"] == "archive"


class TestGenerateCommentBody:
    """Tests for FeedbackTriageAgent._generate_comment_body method."""

    def test_generate_comment_with_all_fields(self):
        """Test comment generation with all fields populated."""
        from src.agents.feedback_triage_agent import FeedbackTriageAgent

        agent = FeedbackTriageAgent(llm=None, github_client=None)

        record = FeedbackRecord(
            timestamp="2025-01-01T12:00:00Z",
            type="text",
            version="0.5.0",
            description="Test description for comment",
            image_description=None,
            annotation="Event",
            is_valid=True,
            is_faithful=True,
            is_complete=True,
            validation_errors=["Error 1", "Error 2"],
            validation_warnings=[],
            evaluation_feedback="",
            assessment_feedback="",
            user_comment="This is my feedback comment.",
        )

        similar_item = GitHubItem(
            number=42,
            title="Related Issue",
            body="Issue body",
            state="open",
            item_type="issue",
            labels=["bug"],
            url="https://github.com/test/repo/issues/42",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-02T00:00:00Z",
        )

        result = TriageResult(
            action="comment",
            reason="Similar to existing issue",
            category="bug",
            severity="medium",
            similar_item=similar_item,
            issue_title=None,
            issue_body=None,
            labels=[],
        )

        comment = agent._generate_comment_body(record, result)

        assert "Related User Feedback" in comment
        assert "text" in comment
        assert "0.5.0" in comment
        assert "Test description for comment" in comment
        assert "This is my feedback comment" in comment
        assert "2 errors" in comment
        assert "Similar to existing issue" in comment

    def test_generate_comment_minimal_fields(self):
        """Test comment generation with minimal fields."""
        from src.agents.feedback_triage_agent import FeedbackTriageAgent

        agent = FeedbackTriageAgent(llm=None, github_client=None)

        record = FeedbackRecord(
            timestamp="2025-01-01T12:00:00Z",
            type="image",
            version="0.5.0",
            description=None,
            image_description="An image",
            annotation="Visual-presentation",
            is_valid=True,
            is_faithful=True,
            is_complete=True,
            validation_errors=[],
            validation_warnings=[],
            evaluation_feedback="",
            assessment_feedback="",
            user_comment=None,
        )

        result = TriageResult(
            action="comment",
            reason="Related to existing PR",
            category="feature",
            severity="low",
            similar_item=None,
            issue_title=None,
            issue_body=None,
            labels=[],
        )

        comment = agent._generate_comment_body(record, result)

        assert "Related User Feedback" in comment
        assert "image" in comment
        assert "0.5.0" in comment
        # Should not contain description or user comment since they're None
        assert "Related to existing PR" in comment


class TestFeedbackRecordToSummary:
    """Tests for FeedbackRecord.to_summary method."""

    def test_to_summary_text_mode(self):
        """Test to_summary for text mode feedback."""
        record = FeedbackRecord(
            timestamp="2025-01-01T12:00:00Z",
            type="text",
            version="0.5.0",
            description="A participant presses a button",
            image_description=None,
            annotation="Action, Agent-action",
            is_valid=True,
            is_faithful=True,
            is_complete=True,
            validation_errors=[],
            validation_warnings=["Minor warning"],
            evaluation_feedback="Good annotation",
            assessment_feedback="Complete",
            user_comment="Works great!",
        )

        summary = record.to_summary()

        assert "text" in summary
        assert "A participant presses a button" in summary
        assert "Action, Agent-action" in summary
        assert "Works great!" in summary

    def test_to_summary_image_mode(self):
        """Test to_summary for image mode feedback."""
        record = FeedbackRecord(
            timestamp="2025-01-01T12:00:00Z",
            type="image",
            version="0.5.0",
            description=None,
            image_description="A red circle on white background",
            annotation="Visual-presentation",
            is_valid=False,
            is_faithful=None,
            is_complete=None,
            validation_errors=["Invalid tag"],
            validation_warnings=[],
            evaluation_feedback="",
            assessment_feedback="",
            user_comment="The annotation is wrong",
        )

        summary = record.to_summary()

        assert "image" in summary
        assert "A red circle on white background" in summary
        assert "Invalid tag" in summary
        assert "The annotation is wrong" in summary
