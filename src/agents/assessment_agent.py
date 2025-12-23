"""Assessment Agent for final annotation comparison.

This agent performs the final assessment to identify any still-missing
elements or dimensions in the HED annotation.
"""

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.state import HedAnnotationState


class AssessmentAgent:
    """Agent that performs final assessment of HED annotations.

    This agent compares the final annotation against the original description
    to provide annotators with feedback on completeness and missing elements.
    """

    def __init__(self, llm: BaseChatModel, schema_dir: Path | str | None = None) -> None:
        """Initialize the assessment agent.

        Args:
            llm: Language model for assessment
            schema_dir: Directory containing JSON schemas (for future use)
        """
        self.llm = llm
        self.schema_dir = schema_dir

    def _build_system_prompt(self) -> str:
        """Build the system prompt for assessment.

        Returns:
            System prompt string
        """
        return """You are a HED annotation assessment specialist.

Compare the HED annotation against the original description and provide brief feedback.

## Assessment Criteria

### 1. Completeness
- Are all key elements from the description captured?
- Event type, objects, actions, attributes

### 2. Semantic Grouping Quality
Check that parentheses correctly group related concepts:
- Properties grouped with their objects: (Red, Circle) not Red, Circle
- Agent-action nesting: ((Agent), (Action, (Object)))
- Spatial relations: ((A), (Relation, (B)))

**Reversibility test**: Can you translate the annotation back to the original English?
If meaning is ambiguous due to ungrouped properties, note this.

### 3. Accuracy
- Does the annotation accurately represent the described event?
- No misrepresentations or incorrect relationships

Response format:
COMPLETENESS: [complete/incomplete]
GROUPING: [correct/needs-improvement]
NOTES: [Brief note on what's captured, missing, or needs grouping fixes, 1-2 sentences]
STATUS: [COMPLETE/NEEDS-REVIEW]
"""

    def _build_user_prompt(self, description: str, annotation: str) -> str:
        """Build the user prompt for assessment.

        Args:
            description: Original natural language description
            annotation: Final HED annotation

        Returns:
            User prompt string
        """
        return f"""Assess this annotation:

DESCRIPTION: {description}
ANNOTATION: {annotation}

Provide brief assessment in the specified format."""

    async def assess(self, state: HedAnnotationState) -> dict:
        """Perform final assessment of the annotation.

        Args:
            state: Current annotation workflow state

        Returns:
            State update with assessment feedback
        """
        # Build prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            state["input_description"],
            state["current_annotation"],
        )

        # Generate assessment
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = await self.llm.ainvoke(messages)
        feedback = response.content.strip()

        # Parse completion status
        is_complete = "FINAL STATUS: COMPLETE" in feedback

        # Update state
        return {
            "assessment_feedback": feedback,
            "is_complete": is_complete,
            "messages": state.get("messages", []) + messages + [response],
        }
