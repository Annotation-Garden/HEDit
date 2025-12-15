#!/usr/bin/env python3
"""CLI script for processing feedback files.

This script is called by the GitHub Actions workflow to process
feedback submissions and triage them appropriately.

Usage:
    python -m src.scripts.process_feedback feedback/file.jsonl
    python -m src.scripts.process_feedback feedback/file.jsonl --dry-run=true
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def process_feedback_file(
    file_path: Path,
    dry_run: bool = False,
) -> list[dict]:
    """Process a single feedback file.

    Args:
        file_path: Path to feedback file (JSON or JSONL)
        dry_run: If True, don't execute actions

    Returns:
        List of processing results
    """
    from src.agents.feedback_triage_agent import (
        FeedbackRecord,
        FeedbackTriageAgent,
        load_feedback_file,
        save_processed_feedback,
    )
    from src.utils.github_client import GitHubClient
    from src.utils.openrouter_llm import create_openrouter_llm

    # Get API keys from environment (prefer testing key for CI/tests)
    openrouter_key = os.getenv("OPENROUTER_API_KEY_FOR_TESTING") or os.getenv("OPENROUTER_API_KEY")
    github_token = os.getenv("GITHUB_TOKEN")

    if not openrouter_key:
        logger.error("OPENROUTER_API_KEY or OPENROUTER_API_KEY_FOR_TESTING not set")
        sys.exit(1)

    # Get model configuration from environment
    model = os.getenv("ANNOTATION_MODEL", "openai/gpt-oss-120b")
    provider = os.getenv("LLM_PROVIDER_PREFERENCE", "Cerebras")

    # Create LLM
    llm = create_openrouter_llm(
        model=model,
        api_key=openrouter_key,
        temperature=0.1,
        max_tokens=1000,
        provider=provider if provider else None,
    )

    # Create GitHub client if token available
    github_client = None
    if github_token:
        github_client = GitHubClient(
            token=github_token,
            owner=os.getenv("GITHUB_REPOSITORY_OWNER", "Annotation-Garden"),
            repo=os.getenv("GITHUB_REPOSITORY", "hedit").split("/")[-1],
        )
        logger.info("GitHub client initialized")
    else:
        logger.warning("GITHUB_TOKEN not set, running without GitHub integration")

    # Create agent
    agent = FeedbackTriageAgent(llm=llm, github_client=github_client)

    # Load feedback records
    if file_path.suffix == ".jsonl":
        records = load_feedback_file(file_path)
    else:
        # Single JSON file
        with open(file_path) as f:
            data = json.load(f)
        records = [FeedbackRecord.from_json(data)]

    logger.info(f"Loaded {len(records)} feedback record(s) from {file_path}")

    results = []
    for i, record in enumerate(records):
        logger.info(f"Processing record {i + 1}/{len(records)}")

        try:
            result = await agent.process_and_execute(record, dry_run=dry_run)
            results.append(result)

            # Log result
            action = result.get("action", "unknown")
            logger.info(f"Record {i + 1}: Action={action}, Reason={result.get('reason', 'N/A')}")

            if "issue_url" in result:
                logger.info(f"  Created issue: {result['issue_url']}")
            if "comment_url" in result:
                logger.info(f"  Added comment: {result['comment_url']}")

            # Save processed feedback
            if not dry_run:
                output_dir = Path("feedback/processed")
                save_processed_feedback(record, result, output_dir)

        except Exception as e:
            logger.error(f"Failed to process record {i + 1}: {e}")
            results.append({"error": str(e)})

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process feedback files and triage to GitHub issues"
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to feedback file (JSON or JSONL)",
    )
    parser.add_argument(
        "--dry-run",
        type=str,
        default="false",
        help="Dry run mode (true/false)",
    )

    args = parser.parse_args()

    # Parse dry_run as boolean
    dry_run = args.dry_run.lower() in ("true", "1", "yes")

    if not args.file.exists():
        logger.error(f"File not found: {args.file}")
        sys.exit(1)

    logger.info(f"Processing feedback file: {args.file}")
    logger.info(f"Dry run mode: {dry_run}")

    # Run async processing
    results = asyncio.run(process_feedback_file(args.file, dry_run=dry_run))

    # Print summary
    print("\n" + "=" * 50)
    print("PROCESSING SUMMARY")
    print("=" * 50)

    actions = {}
    for result in results:
        action = result.get("action", "error")
        actions[action] = actions.get(action, 0) + 1

    for action, count in actions.items():
        print(f"  {action}: {count}")

    print("=" * 50)

    # Output results as JSON for potential downstream processing
    print("\nFull results (JSON):")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
