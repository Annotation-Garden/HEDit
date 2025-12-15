"""GitHub API client for repository operations.

This module provides async operations for interacting with GitHub's REST API,
specifically for retrieving and managing issues and pull requests.
"""

from dataclasses import dataclass
from typing import Literal

import httpx


@dataclass
class GitHubItem:
    """Represents a GitHub issue or pull request."""

    number: int
    title: str
    body: str
    state: str
    item_type: Literal["issue", "pull_request"]
    labels: list[str]
    url: str
    created_at: str
    updated_at: str

    @property
    def summary(self) -> str:
        """Get a brief summary of the item for LLM comparison."""
        body_preview = self.body[:500] if self.body else ""
        if len(self.body) > 500:
            body_preview += "..."
        return f"#{self.number} [{self.item_type}] {self.title}\n{body_preview}"


class GitHubClient:
    """Async client for GitHub API operations."""

    BASE_URL = "https://api.github.com"

    def __init__(
        self,
        token: str,
        owner: str = "Annotation-Garden",
        repo: str = "hedit",
    ):
        """Initialize GitHub client.

        Args:
            token: GitHub personal access token or GITHUB_TOKEN
            owner: Repository owner (default: Annotation-Garden)
            repo: Repository name (default: hedit)
        """
        self.token = token
        self.owner = owner
        self.repo = repo
        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def get_open_issues(self, limit: int = 50) -> list[GitHubItem]:
        """Fetch open issues from the repository.

        Args:
            limit: Maximum number of issues to fetch

        Returns:
            List of GitHubItem objects representing open issues
        """
        url = f"{self.BASE_URL}/repos/{self.owner}/{self.repo}/issues"
        params = {
            "state": "open",
            "per_page": min(limit, 100),
            "sort": "updated",
            "direction": "desc",
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()

        items = []
        for item in data:
            # Skip pull requests (they appear in issues endpoint too)
            if "pull_request" in item:
                continue

            items.append(
                GitHubItem(
                    number=item["number"],
                    title=item["title"],
                    body=item.get("body") or "",
                    state=item["state"],
                    item_type="issue",
                    labels=[label["name"] for label in item.get("labels", [])],
                    url=item["html_url"],
                    created_at=item["created_at"],
                    updated_at=item["updated_at"],
                )
            )

        return items

    async def get_open_pull_requests(self, limit: int = 20) -> list[GitHubItem]:
        """Fetch open pull requests from the repository.

        Args:
            limit: Maximum number of PRs to fetch

        Returns:
            List of GitHubItem objects representing open PRs
        """
        url = f"{self.BASE_URL}/repos/{self.owner}/{self.repo}/pulls"
        params = {
            "state": "open",
            "per_page": min(limit, 100),
            "sort": "updated",
            "direction": "desc",
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()

        items = []
        for item in data:
            items.append(
                GitHubItem(
                    number=item["number"],
                    title=item["title"],
                    body=item.get("body") or "",
                    state=item["state"],
                    item_type="pull_request",
                    labels=[label["name"] for label in item.get("labels", [])],
                    url=item["html_url"],
                    created_at=item["created_at"],
                    updated_at=item["updated_at"],
                )
            )

        return items

    async def get_all_open_items(
        self,
        issue_limit: int = 50,
        pr_limit: int = 20,
    ) -> list[GitHubItem]:
        """Fetch both open issues and pull requests.

        Args:
            issue_limit: Maximum number of issues to fetch
            pr_limit: Maximum number of PRs to fetch

        Returns:
            Combined list of GitHubItem objects
        """
        issues = await self.get_open_issues(limit=issue_limit)
        prs = await self.get_open_pull_requests(limit=pr_limit)
        return issues + prs

    async def add_comment(self, item_number: int, body: str) -> dict:
        """Add a comment to an issue or pull request.

        Args:
            item_number: Issue or PR number
            body: Comment body text

        Returns:
            API response data
        """
        url = f"{self.BASE_URL}/repos/{self.owner}/{self.repo}/issues/{item_number}/comments"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=self.headers,
                json={"body": body},
            )
            response.raise_for_status()
            return response.json()

    async def create_issue(
        self,
        title: str,
        body: str,
        labels: list[str] | None = None,
    ) -> dict:
        """Create a new issue in the repository.

        Args:
            title: Issue title
            body: Issue body (markdown)
            labels: Optional list of label names

        Returns:
            API response data including issue number and URL
        """
        url = f"{self.BASE_URL}/repos/{self.owner}/{self.repo}/issues"

        payload = {
            "title": title,
            "body": body,
        }
        if labels:
            payload["labels"] = labels

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            return response.json()


def format_items_for_prompt(items: list[GitHubItem]) -> str:
    """Format GitHub items for inclusion in an LLM prompt.

    Args:
        items: List of GitHub issues/PRs

    Returns:
        Formatted string with item summaries
    """
    if not items:
        return "No open issues or pull requests found."

    lines = []
    for item in items:
        labels_str = ", ".join(item.labels) if item.labels else "no labels"
        lines.append(f"- #{item.number} [{item.item_type}] ({labels_str}): {item.title}")
        if item.body:
            # Include first 200 chars of body
            preview = item.body[:200].replace("\n", " ")
            if len(item.body) > 200:
                preview += "..."
            lines.append(f"  Description: {preview}")

    return "\n".join(lines)
