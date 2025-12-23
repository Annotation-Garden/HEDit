"""Feedback summarization agent for condensing validation errors and feedback.

This agent summarizes validation errors and evaluation/assessment feedback
into concise, actionable points for the annotation agent.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.state import HedAnnotationState


class FeedbackSummarizer:
    """Agent that summarizes validation errors and feedback.

    Uses a cheap, fast model to condense verbose error messages and
    feedback into concise, actionable summaries for the annotation agent.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialize the feedback summarizer.

        Args:
            llm: Language model for summarization
        """
        self.llm = llm

    def _build_system_prompt(self) -> str:
        """Build the system prompt for feedback summarization.

        Returns:
            System prompt string
        """
        return """You are a feedback summarizer for HED annotation generation.

Your task: Condense validation errors and feedback into concise, actionable points.

Guidelines:
1. Extract only the KEY issues that need fixing
2. Remove verbose error messages and stack traces
3. Group similar errors together
4. Use bullet points for clarity
5. Keep it under 100 words total
6. Focus on WHAT to fix, not technical details

Example 1 - Tag errors:
Input: "TAG_EXTENSION_INVALID: 'Red' does not have 'Property' as its parent..."

Output:
- Remove parent paths from existing tags (use Red, Circle - NOT Property/Red, Item/Circle)

Example 2 - Semantic grouping:
Input: "GROUPING: needs-improvement. Properties like Red and Circle should be grouped together..."

Output:
- Group object properties together: (Red, Circle) not Red, Circle
- Nest agent-action relationships: ((Agent), (Action, (Object)))

Be direct and actionable."""

    def _build_user_prompt(self, state: HedAnnotationState) -> str:
        """Build the user prompt with errors and feedback.

        Args:
            state: Current annotation workflow state

        Returns:
            User prompt string
        """
        feedback_parts = []

        # Add validation errors (use augmented version with remediation guidance for LLM)
        if state.get("validation_errors_augmented"):
            errors = "\n".join(state["validation_errors_augmented"])
            feedback_parts.append(f"VALIDATION ERRORS:\n{errors}")

        # Add evaluation feedback
        if state.get("evaluation_feedback") and not state.get("is_faithful"):
            feedback_parts.append(f"EVALUATION FEEDBACK:\n{state['evaluation_feedback']}")

        # Add assessment feedback
        if state.get("assessment_feedback") and not state.get("is_complete"):
            feedback_parts.append(f"ASSESSMENT FEEDBACK:\n{state['assessment_feedback']}")

        if not feedback_parts:
            return "No feedback to summarize."

        return (
            "\n\n".join(feedback_parts) + "\n\nSummarize the above into concise, actionable points:"
        )

    async def summarize(self, state: HedAnnotationState) -> dict:
        """Summarize validation errors and feedback.

        Args:
            state: Current annotation workflow state

        Returns:
            State update with summarized feedback
        """
        # Build prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(state)

        # Skip if no feedback to summarize
        if user_prompt == "No feedback to summarize.":
            return {}

        # Generate summary
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = await self.llm.ainvoke(messages)
        summarized_feedback = response.content.strip()

        # Replace verbose feedback with summary (only augmented fields for LLM, not raw for users)
        return {
            "validation_errors_augmented": (
                [summarized_feedback] if state.get("validation_errors_augmented") else []
            ),
            "evaluation_feedback": summarized_feedback if state.get("evaluation_feedback") else "",
            "assessment_feedback": summarized_feedback if state.get("assessment_feedback") else "",
        }
