"""Annotation Agent for generating HED tags from natural language.

This agent is responsible for converting natural language event descriptions
into HED annotation strings, using vocabulary constraints and best practices.
"""

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.state import HedAnnotationState
from src.utils.hed_comprehensive_guide import get_comprehensive_hed_guide
from src.utils.json_schema_loader import HedJsonSchemaLoader, load_latest_schema


class AnnotationAgent:
    """Agent that generates HED annotations from natural language descriptions.

    This agent uses an LLM with specialized prompts and vocabulary constraints
    to generate syntactically and semantically correct HED annotations.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        schema_dir: Path | str | None = None,
    ) -> None:
        """Initialize the annotation agent.

        Args:
            llm: Language model for generation
            schema_dir: Directory containing JSON schemas (optional)
        """
        self.llm = llm
        self.schema_dir = schema_dir
        self.json_schema_loader: HedJsonSchemaLoader | None = None

    def _load_json_schema(self, schema_version: str) -> HedJsonSchemaLoader | None:
        """Load JSON schema for given version.

        Args:
            schema_version: Schema version (currently uses latest)

        Returns:
            Loaded JSON schema, or None if schema_dir not provided
        """
        if self.schema_dir is None:
            # No local schema directory - will use HED library for vocabulary
            return None
        # For now, always load latest
        # TODO: Support version-specific loading
        return load_latest_schema(self.schema_dir)

    def _build_system_prompt(
        self,
        vocabulary: list[str],
        extendable_tags: list[str],
    ) -> str:
        """Build the system prompt for the annotation agent.

        Args:
            vocabulary: List of valid short-form HED tags
            extendable_tags: Tags that allow extension

        Returns:
            Complete system prompt with all HED rules
        """
        return get_comprehensive_hed_guide(vocabulary, extendable_tags)

    def _build_user_prompt(
        self,
        description: str,
        validation_errors: list[str] | None = None,
    ) -> str:
        """Build the user prompt for annotation.

        Args:
            description: Natural language event description
            validation_errors: Previous validation errors (if retrying)

        Returns:
            User prompt string
        """
        if validation_errors:
            errors_str = "\n".join(f"- {error}" for error in validation_errors)
            return f"""Previous annotation had validation errors:
{errors_str}

Please fix these errors and generate a corrected HED annotation for:
{description}

Remember to use only valid HED tags and follow proper grouping rules.

CRITICAL: Output ONLY the raw HED annotation string.
Do NOT include:
- Markdown headers (##, ###)
- Code blocks (```)
- Explanatory text like "Here is", "Corrected", "Refined"
- Any other commentary

Just output the HED string directly."""

        return f"""Generate a HED annotation for this event description:
{description}

CRITICAL: Output ONLY the raw HED annotation string.
Do NOT include:
- Markdown headers (##, ###)
- Code blocks (```)
- Explanatory text
- Any commentary

Just output the HED string directly."""

    async def annotate(self, state: HedAnnotationState) -> dict:
        """Generate or refine a HED annotation.

        Args:
            state: Current annotation workflow state

        Returns:
            State update with new annotation
        """
        # Load JSON schema (if schema_dir provided) or use HED library
        if self.json_schema_loader is None:
            self.json_schema_loader = self._load_json_schema(state["schema_version"])

        # Get vocabulary and extensionAllowed tags
        if self.json_schema_loader is not None:
            vocabulary = self.json_schema_loader.get_vocabulary()
            extendable_tags_dict = self.json_schema_loader.get_extendable_tags()
            extendable_tags = list(extendable_tags_dict.keys())
        else:
            # Use HED library to get vocabulary when no local schemas
            from src.utils.schema_loader import get_schema_loader

            schema_loader = get_schema_loader()
            schema = schema_loader.load_schema(state["schema_version"])
            vocabulary = schema_loader.get_schema_vocabulary(schema)
            # Without JSON schema, we don't know which tags are extendable
            # Use empty list - LLM will still generate valid annotations
            extendable_tags = []

        # Build prompts with complete HED rules
        system_prompt = self._build_system_prompt(vocabulary, extendable_tags)

        # Build user prompt with any feedback (use augmented errors with remediation for LLM)
        feedbacks = []
        if state.get("validation_errors_augmented"):
            feedbacks.extend(state["validation_errors_augmented"])
        if state.get("evaluation_feedback") and not state.get("is_faithful"):
            feedbacks.append(f"Evaluation feedback: {state['evaluation_feedback']}")
        if state.get("assessment_feedback") and not state.get("is_complete"):
            feedbacks.append(f"Assessment feedback: {state['assessment_feedback']}")

        user_prompt = self._build_user_prompt(
            state["input_description"],
            feedbacks if feedbacks else None,
        )

        # Generate annotation
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = await self.llm.ainvoke(messages)
        raw_annotation = response.content.strip()

        # Clean up LLM output - extract just the HED annotation
        annotation = self._extract_hed_annotation(raw_annotation)

        # Update state
        return {
            "current_annotation": annotation,
            "messages": messages + [response],
        }

    def _extract_hed_annotation(self, text: str) -> str:
        """Extract HED annotation from LLM response, removing markdown and explanations.

        Args:
            text: Raw LLM response

        Returns:
            Clean HED annotation string
        """
        import re

        # Remove markdown code blocks
        text = re.sub(r"```(?:hed|HED)?\s*\n?", "", text)
        text = re.sub(r"```\s*$", "", text)

        # Remove markdown headers
        text = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.MULTILINE)

        # Split by lines and find HED-like content
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        # Look for lines that look like HED annotations:
        # - Contain HED keywords (Sensory-event, Agent-action, etc.)
        # - Have parentheses and commas typical of HED
        # - Don't start with explanatory words (The, Here, This, etc.)
        hed_keywords = [
            "Sensory-event",
            "Agent-action",
            "Event",
            "Visual-presentation",
            "Participant-response",
            "Experimental-stimulus",
        ]

        for line in lines:
            # Skip obvious explanation lines
            if line.lower().startswith(("the ", "here ", "this ", "note:", "corrected", "refined")):
                continue
            # Check if line contains HED structure
            if any(keyword in line for keyword in hed_keywords) or ("(" in line and "," in line):
                return line

        # If no clear HED line found, return the longest non-empty line
        # (likely to be the annotation)
        if lines:
            return max(lines, key=len)

        return text
