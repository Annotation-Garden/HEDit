"""Keyword Extraction Agent for semantic search preprocessing.

This agent extracts key concepts from natural language event descriptions
using a fast LLM, which are then used for semantic search to find relevant
HED tags.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a keyword extraction specialist for neuroscience event annotation.

Extract the 5-10 most important concepts and keywords from event descriptions.
Focus on:
1. Event types (stimulus, response, action, state)
2. Agents and entities (human, animal, objects)
3. Sensory modalities (visual, auditory, tactile)
4. Actions and movements (press, look, walk)
5. Cognitive/emotional states (attention, fear, reward)
6. Equipment and devices (screen, button, keyboard)
7. Temporal aspects (onset, duration, trial)

Return ONLY a JSON array of lowercase keywords, no explanations.

Example:
Input: "Participant viewed a red circle on screen and pressed the spacebar"
Output: ["participant", "view", "red", "circle", "screen", "press", "spacebar", "visual"]

Input: "Reward delivery: juice given to monkey after correct response"
Output: ["reward", "juice", "monkey", "animal", "correct", "response", "feedback"]"""


class KeywordExtractionAgent:
    """Agent that extracts key concepts from event descriptions.

    Uses a fast, cheap model to identify important keywords and concepts
    that are then used for semantic search to find relevant HED tags.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialize the keyword extraction agent.

        Args:
            llm: Language model for keyword extraction (should be fast/cheap)
        """
        self.llm = llm

    async def extract(self, description: str) -> list[str]:
        """Extract keywords from an event description.

        Args:
            description: Natural language event description

        Returns:
            List of extracted keywords (lowercase)
        """
        user_prompt = f"""Extract key concepts from this event description:

{description}

Return ONLY a JSON array of lowercase keywords."""

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            keywords = self._parse_keywords(response.content)
            logger.debug(f"Extracted keywords: {keywords}")
            return keywords
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            # Fallback: extract simple words from description
            return self._fallback_extraction(description)

    def _parse_keywords(self, content: str) -> list[str]:
        """Parse keywords from LLM response.

        Args:
            content: Raw LLM response content

        Returns:
            List of keywords
        """
        content = content.strip()

        # Try to parse as JSON array
        try:
            # Remove markdown code blocks if present
            content = re.sub(r"```(?:json)?\s*\n?", "", content)
            content = re.sub(r"```\s*$", "", content)
            content = content.strip()

            # Find JSON array in content
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                keywords = json.loads(match.group())
                if isinstance(keywords, list):
                    return [str(k).lower().strip() for k in keywords if k]
        except json.JSONDecodeError:
            pass

        # Fallback: extract quoted strings or comma-separated values
        quoted = re.findall(r'"([^"]+)"', content)
        if quoted:
            return [q.lower().strip() for q in quoted]

        # Last resort: split by comma
        if "," in content:
            return [w.lower().strip() for w in content.split(",") if w.strip()]

        return []

    def _fallback_extraction(self, description: str) -> list[str]:
        """Simple fallback keyword extraction without LLM.

        Args:
            description: Event description

        Returns:
            List of extracted words
        """
        # Remove common stop words and extract meaningful terms
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "and",
            "but",
            "if",
            "or",
            "because",
            "until",
            "while",
            "this",
            "that",
            "these",
            "those",
        }

        # Extract words
        words = re.findall(r"\b[a-zA-Z]{3,}\b", description.lower())
        keywords = [w for w in words if w not in stop_words]

        # Return unique keywords
        seen = set()
        unique = []
        for k in keywords:
            if k not in seen:
                seen.add(k)
                unique.append(k)

        return unique[:10]  # Limit to 10 keywords
