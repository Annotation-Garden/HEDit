"""Feedback triage agent for automated feedback processing.

This agent processes user feedback submissions and determines the appropriate action:
1. Add as comment to existing similar issue/PR
2. Create new GitHub issue if novel and significant
3. Archive to feedback directory for manual review
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.utils.github_client import GitHubClient, GitHubItem, format_items_for_prompt

logger = logging.getLogger(__name__)


@dataclass
class FeedbackRecord:
    """Parsed feedback record from user submission."""

    timestamp: str
    type: Literal["text", "image"]
    version: str
    description: str | None  # For text mode
    image_description: str | None  # For image mode
    annotation: str
    is_valid: bool
    is_faithful: bool | None
    is_complete: bool | None
    validation_errors: list[str]
    validation_warnings: list[str]
    evaluation_feedback: str
    assessment_feedback: str
    user_comment: str | None

    @classmethod
    def from_json(cls, data: dict) -> "FeedbackRecord":
        """Create FeedbackRecord from JSON dict."""
        return cls(
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            type=data.get("type", "text"),
            version=data.get("version", "unknown"),
            description=data.get("description"),
            image_description=data.get("image_description"),
            annotation=data.get("annotation", ""),
            is_valid=data.get("is_valid", False),
            is_faithful=data.get("is_faithful"),
            is_complete=data.get("is_complete"),
            validation_errors=data.get("validation_errors", []),
            validation_warnings=data.get("validation_warnings", []),
            evaluation_feedback=data.get("evaluation_feedback", ""),
            assessment_feedback=data.get("assessment_feedback", ""),
            user_comment=data.get("user_comment"),
        )

    def to_summary(self) -> str:
        """Generate a summary for LLM classification."""
        lines = [
            f"Feedback Type: {self.type}",
            f"Version: {self.version}",
        ]

        if self.description:
            lines.append(f"Input Description: {self.description}")
        if self.image_description:
            lines.append(f"Image Description: {self.image_description}")

        lines.append(f"Generated Annotation: {self.annotation}")
        lines.append(f"Validation Status: {'valid' if self.is_valid else 'invalid'}")

        if self.validation_errors:
            lines.append(f"Validation Errors: {', '.join(self.validation_errors[:3])}")
        if self.evaluation_feedback:
            lines.append(f"Evaluation Feedback: {self.evaluation_feedback[:200]}")
        if self.user_comment:
            lines.append(f"User Comment: {self.user_comment}")

        return "\n".join(lines)


@dataclass
class TriageResult:
    """Result of feedback triage processing."""

    action: Literal["comment", "create_issue", "archive"]
    reason: str
    category: str  # bug, feature, question, duplicate, etc.
    severity: Literal["high", "medium", "low"]
    similar_item: GitHubItem | None  # If action is comment
    issue_title: str | None  # If action is create_issue
    issue_body: str | None  # If action is create_issue
    labels: list[str]  # Suggested labels

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "action": self.action,
            "reason": self.reason,
            "category": self.category,
            "severity": self.severity,
            "labels": self.labels,
        }
        if self.similar_item:
            result["similar_item_number"] = self.similar_item.number
            result["similar_item_url"] = self.similar_item.url
        if self.issue_title:
            result["issue_title"] = self.issue_title
        if self.issue_body:
            result["issue_body"] = self.issue_body
        return result


CLASSIFICATION_PROMPT = """You are analyzing user feedback for HEDit, a multi-agent system that converts natural language descriptions into HED (Hierarchical Event Descriptors) annotations.

IMPORTANT: Be CONSERVATIVE in your classification. Most feedback should be archived for manual review rather than automatically creating GitHub issues. Users can always open issues manually if they believe something is important.

Only classify as HIGH severity if:
- The system is completely broken or unusable
- There is data loss or corruption
- There is a security vulnerability
- A core feature is fundamentally not working

Only mark as "actionable" if:
- The feedback clearly describes a specific, reproducible problem
- The feedback contains a concrete, well-defined feature request
- There is enough detail to act on without further clarification

Analyze the following feedback and classify it:

{feedback_summary}

Classify as one of:
- bug: Something isn't working correctly (validation errors, wrong output, crashes)
- feature: New feature request or enhancement suggestion
- question: User question or clarification needed
- documentation: Documentation improvement needed
- duplicate: Similar to an existing issue
- noise: Not actionable (spam, unclear, off-topic)

Determine severity (be conservative - when in doubt, use "low"):
- high: System completely broken, data loss, security issue, core feature unusable
- medium: Feature not working as expected, but workaround available
- low: Minor inconvenience, cosmetic issue, unclear feedback, general comments

Extract:
- Key concepts (as tags/labels)
- Affected components (validation, annotation, evaluation, api, frontend)
- A 1-2 sentence summary

Respond in JSON format:
{{
    "category": "bug|feature|question|documentation|duplicate|noise",
    "severity": "high|medium|low",
    "summary": "Brief summary of the feedback",
    "concepts": ["tag1", "tag2"],
    "components": ["validation", "annotation"],
    "actionable": true|false
}}"""


SIMILARITY_PROMPT = """Compare the following user feedback with existing GitHub issues/PRs to find potential duplicates or related items.

USER FEEDBACK:
{feedback_summary}

EXISTING ISSUES AND PRs:
{existing_items}

Find the most similar existing issue or PR. Consider:
1. Same type of problem (validation error, annotation quality, etc.)
2. Same component affected
3. Similar error messages or symptoms
4. Related feature requests

Respond in JSON format:
{{
    "has_similar": true|false,
    "similar_number": 123,  // Issue/PR number if found, null otherwise
    "similarity_score": 0.85,  // 0.0-1.0, where 1.0 is identical
    "similarity_reason": "Brief explanation of why they are similar"
}}

If no similar item exists (similarity < 0.5), set has_similar to false."""


ISSUE_GENERATION_PROMPT = """Generate a GitHub issue for the following user feedback about HEDit.

FEEDBACK:
{feedback_summary}

CLASSIFICATION:
Category: {category}
Severity: {severity}
Components: {components}

Create a well-formatted GitHub issue with:
1. Clear, concise title (max 80 chars)
2. Description with proper formatting
3. Steps to reproduce (if applicable)
4. Expected vs actual behavior
5. Version information

Respond in JSON format:
{{
    "title": "Issue title",
    "body": "Full issue body in markdown format",
    "labels": ["label1", "label2"]
}}

Use these label categories:
- type: bug, feature, enhancement, question
- priority: high, medium, low
- component: validation, annotation, evaluation, api, frontend"""


class FeedbackTriageAgent:
    """Agent for triaging user feedback and determining appropriate action."""

    def __init__(
        self,
        llm: BaseChatModel,
        github_client: GitHubClient | None = None,
    ):
        """Initialize the feedback triage agent.

        Args:
            llm: Language model for classification and generation
            github_client: Optional GitHub client for issue operations
        """
        self.llm = llm
        self.github_client = github_client

    async def classify_feedback(self, feedback: FeedbackRecord) -> dict:
        """Classify the feedback using LLM.

        Args:
            feedback: Parsed feedback record

        Returns:
            Classification result dict
        """
        prompt = CLASSIFICATION_PROMPT.format(feedback_summary=feedback.to_summary())

        messages = [
            SystemMessage(
                content="You are a feedback classification assistant. Respond only with valid JSON."
            ),
            HumanMessage(content=prompt),
        ]

        response = await self.llm.ainvoke(messages)
        content = response.content.strip()

        # Extract JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse classification response: {content}")
            return {
                "category": "noise",
                "severity": "low",
                "summary": "Could not classify feedback",
                "concepts": [],
                "components": [],
                "actionable": False,
            }

    async def find_similar_items(
        self,
        feedback: FeedbackRecord,
        existing_items: list[GitHubItem],
    ) -> dict:
        """Find similar existing issues/PRs.

        Args:
            feedback: Parsed feedback record
            existing_items: List of existing GitHub items

        Returns:
            Similarity result dict
        """
        if not existing_items:
            return {"has_similar": False, "similar_number": None, "similarity_score": 0.0}

        prompt = SIMILARITY_PROMPT.format(
            feedback_summary=feedback.to_summary(),
            existing_items=format_items_for_prompt(existing_items),
        )

        messages = [
            SystemMessage(
                content="You are a similarity analysis assistant. Respond only with valid JSON."
            ),
            HumanMessage(content=prompt),
        ]

        response = await self.llm.ainvoke(messages)
        content = response.content.strip()

        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse similarity response: {content}")
            return {"has_similar": False, "similar_number": None, "similarity_score": 0.0}

    async def generate_issue_content(
        self,
        feedback: FeedbackRecord,
        classification: dict,
    ) -> dict:
        """Generate issue title and body.

        Args:
            feedback: Parsed feedback record
            classification: Classification result

        Returns:
            Issue content dict with title, body, labels
        """
        prompt = ISSUE_GENERATION_PROMPT.format(
            feedback_summary=feedback.to_summary(),
            category=classification.get("category", "unknown"),
            severity=classification.get("severity", "medium"),
            components=", ".join(classification.get("components", [])),
        )

        messages = [
            SystemMessage(
                content="You are a GitHub issue writing assistant. Respond only with valid JSON."
            ),
            HumanMessage(content=prompt),
        ]

        response = await self.llm.ainvoke(messages)
        content = response.content.strip()

        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse issue content response: {content}")
            return {
                "title": f"[Feedback] {classification.get('summary', 'User feedback')}",
                "body": f"## User Feedback\n\n{feedback.to_summary()}",
                "labels": ["type: feedback"],
            }

    async def triage(
        self,
        feedback: FeedbackRecord,
        existing_items: list[GitHubItem] | None = None,
        similarity_threshold: float = 0.8,
    ) -> TriageResult:
        """Perform complete triage on feedback.

        Args:
            feedback: Parsed feedback record
            existing_items: Optional list of existing GitHub items
            similarity_threshold: Threshold for considering items similar (0.0-1.0)

        Returns:
            TriageResult with action and details
        """
        # Step 1: Classify feedback
        classification = await self.classify_feedback(feedback)
        logger.info(f"Classification: {classification}")

        # If not actionable, archive
        if not classification.get("actionable", True) or classification.get("category") == "noise":
            return TriageResult(
                action="archive",
                reason="Feedback classified as not actionable",
                category=classification.get("category", "noise"),
                severity="low",
                similar_item=None,
                issue_title=None,
                issue_body=None,
                labels=[],
            )

        # Step 2: Check for similar existing items
        if existing_items is None and self.github_client:
            existing_items = await self.github_client.get_all_open_items()

        similarity_result = {"has_similar": False}
        similar_item = None

        if existing_items:
            similarity_result = await self.find_similar_items(feedback, existing_items)
            logger.info(f"Similarity result: {similarity_result}")

            if (
                similarity_result.get("has_similar")
                and similarity_result.get("similarity_score", 0) >= similarity_threshold
            ):
                # Find the matching item
                similar_number = similarity_result.get("similar_number")
                similar_item = next(
                    (item for item in existing_items if item.number == similar_number),
                    None,
                )

        # Step 3: Determine action
        if similar_item and similarity_result.get("similarity_score", 0) >= similarity_threshold:
            # Add as comment to existing issue
            return TriageResult(
                action="comment",
                reason=similarity_result.get("similarity_reason", "Similar to existing issue"),
                category=classification.get("category", "duplicate"),
                severity=classification.get("severity", "medium"),
                similar_item=similar_item,
                issue_title=None,
                issue_body=None,
                labels=classification.get("concepts", []),
            )

        # Check if serious enough to create issue
        # Be stringent: only create issues for critical bugs or explicit feature requests
        # Users can always open issues manually if they think it's important
        severity = classification.get("severity", "low")
        category = classification.get("category", "")
        is_actionable = classification.get("actionable", False)

        # Only create issues for:
        # 1. High severity bugs (critical issues that break functionality)
        # 2. Explicit, actionable feature requests with high severity
        should_create_issue = (category == "bug" and severity == "high") or (
            category == "feature" and severity == "high" and is_actionable
        )

        if should_create_issue:
            # Generate issue content
            issue_content = await self.generate_issue_content(feedback, classification)
            return TriageResult(
                action="create_issue",
                reason=f"Novel {category} with {severity} severity",
                category=category,
                severity=severity,
                similar_item=None,
                issue_title=issue_content.get("title"),
                issue_body=issue_content.get("body"),
                labels=issue_content.get("labels", []),
            )

        # Archive for manual review
        return TriageResult(
            action="archive",
            reason=f"Low priority {category}, archived for manual review",
            category=category,
            severity=severity,
            similar_item=None,
            issue_title=None,
            issue_body=None,
            labels=classification.get("concepts", []),
        )

    async def process_and_execute(
        self,
        feedback: FeedbackRecord,
        dry_run: bool = False,
    ) -> dict:
        """Process feedback and execute the determined action.

        Args:
            feedback: Parsed feedback record
            dry_run: If True, don't actually create/comment on issues

        Returns:
            Result dict with action taken and details
        """
        # Get existing items if we have a client
        existing_items = None
        if self.github_client:
            try:
                existing_items = await self.github_client.get_all_open_items()
            except Exception as e:
                logger.warning(f"Failed to fetch GitHub items: {e}")

        # Perform triage
        result = await self.triage(feedback, existing_items)
        logger.info(f"Triage result: {result}")

        output = {
            "action": result.action,
            "reason": result.reason,
            "category": result.category,
            "severity": result.severity,
            "dry_run": dry_run,
        }

        if dry_run:
            output["would_execute"] = result.to_dict()
            return output

        # Execute action
        if result.action == "comment" and result.similar_item and self.github_client:
            comment_body = self._generate_comment_body(feedback, result)
            response = await self.github_client.add_comment(
                result.similar_item.number,
                comment_body,
            )
            output["comment_url"] = response.get("html_url")
            output["issue_number"] = result.similar_item.number

        elif result.action == "create_issue" and result.issue_title and self.github_client:
            response = await self.github_client.create_issue(
                title=result.issue_title,
                body=result.issue_body,
                labels=result.labels,
            )
            output["issue_url"] = response.get("html_url")
            output["issue_number"] = response.get("number")

        elif result.action == "archive":
            # Archive is handled by the caller
            output["archive_data"] = result.to_dict()

        return output

    def _generate_comment_body(
        self,
        feedback: FeedbackRecord,
        result: TriageResult,
    ) -> str:
        """Generate comment body for adding to existing issue."""
        lines = [
            "## Related User Feedback",
            "",
            "A user submitted feedback that appears related to this issue.",
            "",
            "### Feedback Summary",
            f"- **Type**: {feedback.type}",
            f"- **Version**: {feedback.version}",
        ]

        if feedback.description:
            lines.append(f"- **Input**: {feedback.description[:100]}...")
        if feedback.user_comment:
            lines.append(f"- **User Comment**: {feedback.user_comment}")

        if feedback.validation_errors:
            lines.append(f"- **Validation Errors**: {len(feedback.validation_errors)} errors")

        lines.extend(
            [
                "",
                f"**Similarity Reason**: {result.reason}",
                "",
                "---",
                "*This comment was automatically generated by HEDit feedback processing.*",
            ]
        )

        return "\n".join(lines)


def load_feedback_file(file_path: Path) -> list[FeedbackRecord]:
    """Load feedback records from a JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of FeedbackRecord objects
    """
    records = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                records.append(FeedbackRecord.from_json(data))
    return records


def save_processed_feedback(
    feedback: FeedbackRecord,
    result: dict,
    output_dir: Path,
) -> Path:
    """Save processed feedback with metadata.

    Args:
        feedback: Original feedback record
        result: Processing result
        output_dir: Directory to save to

    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create filename from timestamp
    timestamp = feedback.timestamp.replace(":", "-").replace(".", "-")
    filename = f"processed-{timestamp}.json"
    output_path = output_dir / filename

    output_data = {
        "original_feedback": {
            "timestamp": feedback.timestamp,
            "type": feedback.type,
            "version": feedback.version,
            "description": feedback.description,
            "annotation": feedback.annotation,
            "user_comment": feedback.user_comment,
        },
        "processing_result": result,
        "processed_at": datetime.now().isoformat(),
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    return output_path
