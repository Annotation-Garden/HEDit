"""Evaluation Agent for assessing annotation faithfulness.

This agent evaluates how faithfully a HED annotation captures
the original natural language event description.
"""

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.state import HedAnnotationState
from src.utils.json_schema_loader import load_latest_schema


class EvaluationAgent:
    """Agent that evaluates the faithfulness of HED annotations.

    This agent compares the generated HED annotation against the original
    description to assess completeness, accuracy, and semantic fidelity.
    Also suggests schema matches for tags that might not exist.
    """

    def __init__(self, llm: BaseChatModel, schema_dir: Path | str | None = None) -> None:
        """Initialize the evaluation agent.

        Args:
            llm: Language model for evaluation
            schema_dir: Directory containing JSON schemas
        """
        self.llm = llm
        self.schema_dir = schema_dir
        self.json_schema_loader = None

    def _build_system_prompt(self) -> str:
        """Build the system prompt for evaluation.

        Returns:
            System prompt string
        """
        return """You are an expert HED annotation evaluator.

Your task is to assess how faithfully a HED annotation captures the original natural language event description.

## Evaluation Philosophy

- **Be practical, not perfectionist**: Annotations don't need to capture EVERY detail
- **Focus on core event elements**: Event type, main objects, key actions
- **Accept reasonable variations**: Multiple valid ways to annotate the same event
- **Prioritize correctness over completeness**: Better to be accurate than exhaustive

## Evaluation Criteria

### 1. Core Elements (REQUIRED)
- Event type identified (Sensory-event, Agent-action, etc.)
- Main objects/stimuli included
- Key actions captured (if any)

### 2. Important Details (SHOULD HAVE)
- Colors, shapes if explicitly mentioned
- Spatial relationships if specified
- Task role (Experimental-stimulus, Participant-response)

### 3. Optional Enhancements (NICE TO HAVE)
- Fine-grained attributes
- Implicit details
- Additional context

## Decision Guidelines

**ACCEPT if**:
- Core elements are present ✓
- No major inaccuracies
- Can translate back to similar English description

**REFINE only if**:
- Missing critical information (event type, main object, key action)
- Contains clear errors or misrepresentations
- Would fail reversibility test (can't translate back)

## Response Format

FAITHFUL: [yes/partial/no]

DECISION: [ACCEPT/REFINE]

FEEDBACK:
- [Brief feedback if refinement needed]
"""

    def _build_user_prompt(self, description: str, annotation: str) -> str:
        """Build the user prompt for evaluation.

        Args:
            description: Original natural language description
            annotation: Generated HED annotation

        Returns:
            User prompt string
        """
        return f"""Evaluate this HED annotation:

ORIGINAL DESCRIPTION:
{description}

HED ANNOTATION:
{annotation}

Provide a thorough evaluation following the specified format."""

    async def evaluate(self, state: HedAnnotationState) -> dict:
        """Evaluate the faithfulness of the current annotation.

        Args:
            state: Current annotation workflow state

        Returns:
            State update with evaluation feedback
        """
        # Load schema if needed (only if schema_dir provided)
        if self.json_schema_loader is None and self.schema_dir is not None:
            self.json_schema_loader = load_latest_schema(self.schema_dir)

        # Check for potentially invalid tags and suggest matches
        annotation = state["current_annotation"]
        suggestions = self._check_tags_and_suggest(annotation)

        # Build prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            state["input_description"],
            annotation,
        )

        # Add tag suggestions if any
        if suggestions:
            user_prompt += f"\n\n**Tag Suggestions**:\n{suggestions}"

        # Generate evaluation
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = await self.llm.ainvoke(messages)
        feedback = response.content.strip()

        # Parse decision with multiple fallbacks
        is_faithful = self._parse_decision(feedback)

        # Update state
        return {
            "evaluation_feedback": feedback,
            "is_faithful": is_faithful,
            "messages": state.get("messages", []) + messages + [response],
        }

    def _parse_decision(self, feedback: str) -> bool:
        """Parse evaluation decision from LLM feedback.

        Args:
            feedback: LLM evaluation feedback

        Returns:
            True if annotation should be accepted, False if needs refinement
        """
        import re

        feedback_lower = feedback.lower()

        # Check for explicit DECISION line
        decision_match = re.search(r"decision:\s*(accept|refine)", feedback_lower)
        if decision_match:
            return decision_match.group(1) == "accept"

        # Check for FAITHFUL field - accept "yes" or "partial"
        faithful_match = re.search(r"faithful:\s*(yes|partial|no)", feedback_lower)
        if faithful_match:
            result = faithful_match.group(1)
            return result in ["yes", "partial"]  # Accept partial as good enough!

        # Fallback: look for positive indicators
        positive_indicators = ["accept", "good", "sufficient", "adequate", "captures well"]
        negative_indicators = ["refine", "missing", "incorrect", "inaccurate", "lacks"]

        positive_score = sum(1 for indicator in positive_indicators if indicator in feedback_lower)
        negative_score = sum(1 for indicator in negative_indicators if indicator in feedback_lower)

        # If more positive than negative, accept
        if positive_score > negative_score:
            return True

        # Default to refine if ambiguous (conservative)
        return False

    def _check_tags_and_suggest(self, annotation: str) -> str:
        """Check annotation for invalid tags and suggest alternatives.

        Args:
            annotation: HED annotation string

        Returns:
            Suggestion text (empty if all tags valid)
        """
        if not self.json_schema_loader:
            return ""

        # Extract tags from annotation (simple tokenization)
        # Remove parentheses, split by comma
        cleaned = annotation.replace("(", "").replace(")", "")
        tags = [t.strip() for t in cleaned.split(",")]

        vocabulary = set(self.json_schema_loader.get_vocabulary())
        suggestions = []

        for tag in tags:
            # Skip empty, value placeholders, or column references
            if not tag or "#" in tag or "{" in tag:
                continue

            # Check if tag or its base (before /) is in vocabulary
            base_tag = tag.split("/")[0]
            if base_tag not in vocabulary:
                # Find closest matches
                matches = self.json_schema_loader.find_closest_match(base_tag)
                if matches:
                    suggestions.append(
                        f"- '{base_tag}' not in schema. Did you mean: {', '.join(matches)}?"
                    )
                else:
                    # Check if it's a valid extension
                    if "/" in tag:
                        if self.json_schema_loader.is_extendable(base_tag):
                            suggestions.append(
                                f"- '{tag}' uses extension (dataset-specific, non-portable)"
                            )
                        else:
                            suggestions.append(
                                f"- '{base_tag}' doesn't allow extension. Use schema tag instead."
                            )

        return "\n".join(suggestions) if suggestions else ""
