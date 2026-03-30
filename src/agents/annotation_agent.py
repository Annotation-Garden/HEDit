"""Annotation Agent for generating HED tags from natural language.

This agent is responsible for converting natural language event descriptions
into HED annotation strings, using vocabulary constraints and best practices.
"""

import logging
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.state import HedAnnotationState
from src.utils import extract_text_content
from src.utils.hed_comprehensive_guide import get_comprehensive_hed_guide
from src.utils.json_schema_loader import HedJsonSchemaLoader, load_latest_schema

logger = logging.getLogger(__name__)


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
        no_extend: bool = False,
    ) -> str:
        """Build the system prompt for the annotation agent.

        The system prompt is static per schema version (vocabulary + rules)
        to enable prompt caching across requests. Semantic hints are placed
        in the user prompt instead.

        Args:
            vocabulary: List of valid short-form HED tags
            extendable_tags: Tags that allow extension
            no_extend: If True, prohibit tag extensions

        Returns:
            Complete system prompt with all HED rules
        """
        return get_comprehensive_hed_guide(vocabulary, extendable_tags, no_extend)

    def _format_tag_suggestions(self, tag_suggestions: dict[str, list[str]]) -> str:
        """Format tag suggestions into a clear instruction block.

        Args:
            tag_suggestions: Mapping of invalid tags to suggested valid alternatives

        Returns:
            Formatted suggestion string
        """
        if not tag_suggestions:
            return ""

        lines = ["\nSuggested VALID HED tag replacements for invalid tags:"]
        for invalid_tag, suggestions in tag_suggestions.items():
            if suggestions:
                lines.append(f"  Instead of '{invalid_tag}', use: {', '.join(suggestions[:3])}")
            else:
                lines.append(
                    f"  '{invalid_tag}' has no direct match; check the HED schema vocabulary"
                )
        return "\n".join(lines)

    def _format_semantic_hints(self, semantic_hints: list[dict] | None) -> str:
        """Format semantic hints for inclusion in the user prompt.

        Args:
            semantic_hints: List of hint dicts with tag, score, source keys

        Returns:
            Formatted hints section, or empty string if no hints
        """
        if not semantic_hints:
            return ""

        from src.utils.hed_comprehensive_guide import _format_semantic_hints

        return "\n" + _format_semantic_hints(semantic_hints)

    def _build_user_prompt(
        self,
        description: str,
        validation_errors: list[str] | None = None,
        tag_suggestions: dict[str, list[str]] | None = None,
        previous_annotation: str | None = None,
        semantic_hints: list[dict] | None = None,
    ) -> str:
        """Build the user prompt for annotation.

        Semantic hints are included here (not in system prompt) so the
        system prompt stays static and cacheable across requests.

        Args:
            description: Natural language event description
            validation_errors: Previous validation errors (if retrying)
            tag_suggestions: LSP-suggested valid tags for invalid tags
            previous_annotation: The previous annotation attempt (for targeted correction)
            semantic_hints: Optional semantic search hints for relevant tags

        Returns:
            User prompt string
        """
        hints_str = self._format_semantic_hints(semantic_hints)

        if validation_errors:
            errors_str = "\n".join(f"- {error}" for error in validation_errors)
            suggestions_str = self._format_tag_suggestions(tag_suggestions or {})
            replacement_note = (
                "IMPORTANT: Replace invalid tags with the suggested alternatives above.\n"
                if suggestions_str
                else ""
            )
            previous_section = (
                f"Previous annotation:\n{previous_annotation}\n\n" if previous_annotation else ""
            )
            return f"""{previous_section}Validation errors found:
{errors_str}
{suggestions_str}

Fix these errors and generate a corrected HED annotation for:
{description}
{hints_str}
{replacement_note}CRITICAL: Output ONLY the raw HED annotation string."""

        return f"""Generate a HED annotation for this event description:
{description}
{hints_str}
CRITICAL: Output ONLY the raw HED annotation string."""

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

        # Build system prompt with HED rules (static per schema version for caching)
        no_extend = state.get("no_extend", False)
        system_prompt = self._build_system_prompt(
            vocabulary,
            extendable_tags,
            no_extend,
        )

        # Build user prompt with feedback and semantic hints
        semantic_hints = state.get("semantic_hints", [])
        feedbacks = []
        if state.get("validation_errors_augmented"):
            feedbacks.extend(state["validation_errors_augmented"])
        if state.get("evaluation_feedback") and not state.get("is_faithful"):
            feedbacks.append(f"Evaluation feedback: {state['evaluation_feedback']}")
        if state.get("assessment_feedback") and not state.get("is_complete"):
            feedbacks.append(f"Assessment feedback: {state['assessment_feedback']}")

        # Pass tag suggestions directly (survives feedback summarization)
        tag_suggestions = state.get("tag_suggestions", {})

        # Include previous annotation for targeted corrections
        previous_annotation = state.get("current_annotation") if feedbacks else None

        user_prompt = self._build_user_prompt(
            state["input_description"],
            feedbacks or None,
            tag_suggestions or None,
            previous_annotation,
            semantic_hints if semantic_hints else None,
        )

        # Generate annotation
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = await self.llm.ainvoke(messages)
        except Exception as e:
            logger.error("LLM invocation failed: %s", e, exc_info=True)
            raise
        raw_annotation = extract_text_content(response.content)

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
