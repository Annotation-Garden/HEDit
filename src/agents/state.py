"""LangGraph state definition for HED annotation workflow.

This module defines the state structure that flows through the
multi-agent annotation pipeline.
"""

from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict


class HedAnnotationState(TypedDict):
    """State for the HED annotation workflow.

    This state is passed between agents and tracks the entire
    annotation generation and validation process.

    Attributes:
        messages: Conversation history and agent communications
        input_description: Original natural language event description
        current_annotation: Current HED annotation string
        validation_status: Current validation status
        validation_errors: List of validation error messages
        validation_warnings: List of validation warning messages
        validation_attempts: Number of validation attempts made
        total_iterations: Total number of annotation attempts (validation + evaluation)
        evaluation_feedback: Feedback from evaluation agent
        assessment_feedback: Feedback from assessment agent
        is_valid: Whether current annotation is valid
        is_faithful: Whether annotation faithfully captures description
        is_complete: Whether annotation is complete
        max_validation_attempts: Maximum allowed validation attempts
        max_total_iterations: Maximum total iterations to prevent infinite loops
        schema_version: HED schema version being used
    """

    # Message history for agent communication
    messages: Annotated[list[BaseMessage], add_messages]

    # Input and output
    input_description: str
    current_annotation: str

    # Validation tracking
    validation_status: Literal["pending", "valid", "invalid", "max_attempts_reached"]
    validation_errors: list[str]
    validation_warnings: list[str]
    validation_attempts: int
    total_iterations: int  # Total annotation attempts

    # Agent feedback
    evaluation_feedback: str
    assessment_feedback: str

    # Status flags
    is_valid: bool
    is_faithful: bool
    is_complete: bool

    # Configuration
    max_validation_attempts: int
    max_total_iterations: int  # Max total iterations
    schema_version: str
    run_assessment: bool  # Whether to run final assessment


def create_initial_state(
    input_description: str,
    schema_version: str = "8.3.0",
    max_validation_attempts: int = 5,
    max_total_iterations: int = 10,
    run_assessment: bool = False,
) -> HedAnnotationState:
    """Create an initial state for a new annotation workflow.

    Args:
        input_description: Natural language event description to annotate
        schema_version: HED schema version to use (default: "8.3.0")
        max_validation_attempts: Maximum validation retry attempts (default: 5)
        max_total_iterations: Maximum total iterations to prevent infinite loops (default: 10)
        run_assessment: Whether to run final assessment (default: False)

    Returns:
        Initial HedAnnotationState
    """
    return HedAnnotationState(
        messages=[],
        input_description=input_description,
        current_annotation="",
        validation_status="pending",
        validation_errors=[],
        validation_warnings=[],
        validation_attempts=0,
        total_iterations=0,
        evaluation_feedback="",
        assessment_feedback="",
        is_valid=False,
        is_faithful=False,
        is_complete=False,
        max_validation_attempts=max_validation_attempts,
        max_total_iterations=max_total_iterations,
        schema_version=schema_version,
        run_assessment=run_assessment,
    )
