"""LangGraph workflow for HED annotation generation.

This module defines the multi-agent workflow that orchestrates
annotation, validation, evaluation, and assessment.
"""

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph

from src.agents.annotation_agent import AnnotationAgent
from src.agents.assessment_agent import AssessmentAgent
from src.agents.evaluation_agent import EvaluationAgent
from src.agents.feedback_summarizer import FeedbackSummarizer
from src.agents.state import HedAnnotationState
from src.agents.validation_agent import ValidationAgent
from src.utils.schema_loader import HedSchemaLoader


class HedAnnotationWorkflow:
    """Multi-agent workflow for HED annotation generation and validation.

    The workflow follows this pattern:
    1. Annotation: Generate HED tags from natural language
    2. Validation: Check HED compliance
    3. If errors and attempts < max: Return to annotation with feedback
    4. If valid: Proceed to evaluation
    5. Evaluation: Assess faithfulness to original description
    6. If needs refinement: Return to annotation
    7. If faithful: Proceed to assessment
    8. Assessment: Final comparison for completeness
    9. End: Return final annotation with feedback
    """

    def __init__(
        self,
        llm: BaseChatModel,
        evaluation_llm: BaseChatModel | None = None,
        assessment_llm: BaseChatModel | None = None,
        feedback_llm: BaseChatModel | None = None,
        schema_dir: Path | str | None = None,
        validator_path: Path | None = None,
        use_js_validator: bool = True,
    ) -> None:
        """Initialize the workflow.

        Args:
            llm: Language model for annotation agent
            evaluation_llm: Language model for evaluation agent (defaults to llm)
            assessment_llm: Language model for assessment agent (defaults to llm)
            feedback_llm: Language model for feedback summarization (defaults to llm)
            schema_dir: Directory containing JSON schemas
            validator_path: Path to hed-javascript for validation
            use_js_validator: Whether to use JavaScript validator
        """
        # Store schema directory (None means use HED library to fetch from GitHub)
        self.schema_dir = schema_dir

        # Initialize legacy schema loader for validation
        self.schema_loader = HedSchemaLoader()

        # Use provided LLMs or default to main llm
        eval_llm = evaluation_llm or llm
        assess_llm = assessment_llm or llm
        feed_llm = feedback_llm or llm

        # Initialize agents with JSON schema support and per-agent LLMs
        self.annotation_agent = AnnotationAgent(llm, schema_dir=self.schema_dir)
        self.validation_agent = ValidationAgent(
            self.schema_loader,
            use_javascript=use_js_validator,
            validator_path=validator_path,
        )
        self.evaluation_agent = EvaluationAgent(eval_llm, schema_dir=self.schema_dir)
        self.assessment_agent = AssessmentAgent(assess_llm, schema_dir=self.schema_dir)
        self.feedback_summarizer = FeedbackSummarizer(feed_llm)

        # Build graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow.

        Returns:
            Compiled StateGraph
        """
        # Create graph
        workflow = StateGraph(HedAnnotationState)

        # Add nodes
        workflow.add_node("annotate", self._annotate_node)
        workflow.add_node("validate", self._validate_node)
        workflow.add_node("summarize_feedback", self._summarize_feedback_node)
        workflow.add_node("evaluate", self._evaluate_node)
        workflow.add_node("assess", self._assess_node)

        # Add edges
        workflow.set_entry_point("annotate")

        # After annotation, always validate
        workflow.add_edge("annotate", "validate")

        # After validation, route based on result
        workflow.add_conditional_edges(
            "validate",
            self._route_after_validation,
            {
                "summarize_feedback": "summarize_feedback",  # Summarize feedback if invalid
                "evaluate": "evaluate",  # Proceed if valid
                "end": END,  # End if max attempts reached
            },
        )

        # After feedback summarization, go to annotation
        workflow.add_edge("summarize_feedback", "annotate")

        # After evaluation, route based on faithfulness
        workflow.add_conditional_edges(
            "evaluate",
            self._route_after_evaluation,
            {
                "summarize_feedback": "summarize_feedback",  # Summarize feedback if not faithful
                "assess": "assess",  # Proceed to assessment if needed
                "end": END,  # Skip assessment if valid and faithful
            },
        )

        # After assessment, always end
        workflow.add_edge("assess", END)

        return workflow.compile()

    async def _annotate_node(self, state: HedAnnotationState) -> dict:
        """Annotation node: Generate or refine HED annotation.

        Args:
            state: Current workflow state

        Returns:
            State update
        """
        total_iters = state.get("total_iterations", 0) + 1
        print(
            f"[WORKFLOW] Entering annotate node (validation attempt {state['validation_attempts']}, total iteration {total_iters})"
        )
        result = await self.annotation_agent.annotate(state)
        result["total_iterations"] = total_iters  # Increment counter
        print(f"[WORKFLOW] Annotation generated: {result.get('current_annotation', '')[:100]}...")
        return result

    async def _validate_node(self, state: HedAnnotationState) -> dict:
        """Validation node: Validate HED annotation.

        Args:
            state: Current workflow state

        Returns:
            State update
        """
        print("[WORKFLOW] Entering validate node")
        result = await self.validation_agent.validate(state)
        print(
            f"[WORKFLOW] Validation result: {result.get('validation_status')}, is_valid: {result.get('is_valid')}"
        )
        if not result.get("is_valid"):
            print(f"[WORKFLOW] Validation errors: {result.get('validation_errors', [])}")
        return result

    async def _evaluate_node(self, state: HedAnnotationState) -> dict:
        """Evaluation node: Evaluate annotation faithfulness.

        Args:
            state: Current workflow state

        Returns:
            State update
        """
        print("[WORKFLOW] Entering evaluate node")
        result = await self.evaluation_agent.evaluate(state)
        print(f"[WORKFLOW] Evaluation result: is_faithful={result.get('is_faithful')}")

        # Set default assessment values if assessment will be skipped
        run_assessment = state.get("run_assessment", False)
        if not run_assessment:
            result["is_complete"] = result.get("is_faithful", False) and state.get(
                "is_valid", False
            )
            if result["is_complete"]:
                result["assessment_feedback"] = (
                    "Annotation is valid and faithful to the original description."
                )
            else:
                result["assessment_feedback"] = ""

        return result

    async def _assess_node(self, state: HedAnnotationState) -> dict:
        """Assessment node: Final assessment.

        Args:
            state: Current workflow state

        Returns:
            State update
        """
        return await self.assessment_agent.assess(state)

    async def _summarize_feedback_node(self, state: HedAnnotationState) -> dict:
        """Summarize feedback node: Condense errors and feedback.

        Args:
            state: Current workflow state

        Returns:
            State update with summarized feedback
        """
        print("[WORKFLOW] Entering summarize_feedback node")
        result = await self.feedback_summarizer.summarize(state)
        print(
            f"[WORKFLOW] Feedback summarized: {result.get('validation_errors_augmented', [''])[0][:100] if result.get('validation_errors_augmented') else 'No feedback'}..."
        )
        return result

    def _route_after_validation(
        self,
        state: HedAnnotationState,
    ) -> str:
        """Route after validation based on result.

        Args:
            state: Current workflow state

        Returns:
            Next node name
        """
        if state["validation_status"] == "valid":
            print("[WORKFLOW] Routing to evaluate (validation passed)")
            return "evaluate"
        elif state["validation_status"] == "max_attempts_reached":
            print("[WORKFLOW] Routing to end (max validation attempts reached)")
            return "end"
        else:
            print(
                f"[WORKFLOW] Routing to summarize_feedback (validation failed, attempts: {state['validation_attempts']}/{state['max_validation_attempts']})"
            )
            return "summarize_feedback"

    def _route_after_evaluation(
        self,
        state: HedAnnotationState,
    ) -> str:
        """Route after evaluation based on faithfulness.

        Args:
            state: Current workflow state

        Returns:
            Next node name
        """
        # Check if max total iterations reached
        total_iters = state.get("total_iterations", 0)
        max_iters = state.get("max_total_iterations", 10)
        run_assessment = state.get("run_assessment", False)

        if total_iters >= max_iters:
            # Only run assessment at max iterations if explicitly requested
            if run_assessment:
                print(f"[WORKFLOW] Routing to assess (max total iterations {max_iters} reached)")
                return "assess"
            else:
                print(
                    "[WORKFLOW] Skipping assessment (max iterations reached, assessment not requested) - routing to END"
                )
                return "end"

        if state["is_faithful"]:
            # Only run assessment if explicitly requested
            if state.get("is_valid") and run_assessment:
                print(
                    "[WORKFLOW] Routing to assess (annotation is valid and faithful, assessment requested)"
                )
                return "assess"
            elif state.get("is_valid"):
                print(
                    "[WORKFLOW] Skipping assessment (annotation is valid and faithful, assessment not requested) - routing to END"
                )
                return "end"
            elif run_assessment:
                print(
                    "[WORKFLOW] Routing to assess (annotation is faithful but has validation issues)"
                )
                return "assess"
            else:
                print(
                    "[WORKFLOW] Skipping assessment (has validation issues, assessment not requested) - routing to END"
                )
                return "end"
        else:
            print(
                f"[WORKFLOW] Routing to summarize_feedback (annotation needs refinement, iteration {total_iters}/{max_iters})"
            )
            return "summarize_feedback"

    async def run(
        self,
        input_description: str,
        schema_version: str = "8.3.0",
        max_validation_attempts: int = 5,
        max_total_iterations: int = 10,
        run_assessment: bool = False,
        config: dict | None = None,
    ) -> HedAnnotationState:
        """Run the complete annotation workflow.

        Args:
            input_description: Natural language event description
            schema_version: HED schema version to use
            max_validation_attempts: Maximum validation retry attempts
            max_total_iterations: Maximum total iterations to prevent infinite loops
            run_assessment: Whether to run final assessment (default: False)
            config: Optional LangGraph config (e.g., recursion_limit)

        Returns:
            Final workflow state with annotation and feedback
        """
        from src.agents.state import create_initial_state

        # Create initial state
        initial_state = create_initial_state(
            input_description,
            schema_version,
            max_validation_attempts,
            max_total_iterations,
            run_assessment,
        )

        # Run workflow
        final_state = await self.graph.ainvoke(initial_state, config=config)

        return final_state
