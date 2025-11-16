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
        schema_dir: Path | str | None = None,
        validator_path: Path | None = None,
        use_js_validator: bool = True,
    ) -> None:
        """Initialize the workflow.

        Args:
            llm: Language model for agents
            schema_dir: Directory containing JSON schemas
            validator_path: Path to hed-javascript for validation
            use_js_validator: Whether to use JavaScript validator
        """
        # Store schema directory
        self.schema_dir = schema_dir or Path(
            "/Users/yahya/Documents/git/HED/hed-schemas/schemas_latest_json"
        )

        # Initialize legacy schema loader for validation
        self.schema_loader = HedSchemaLoader()

        # Initialize agents with JSON schema support
        self.annotation_agent = AnnotationAgent(llm, schema_dir=self.schema_dir)
        self.validation_agent = ValidationAgent(
            self.schema_loader,
            use_javascript=use_js_validator,
            validator_path=validator_path,
        )
        self.evaluation_agent = EvaluationAgent(llm, schema_dir=self.schema_dir)
        self.assessment_agent = AssessmentAgent(llm, schema_dir=self.schema_dir)

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
                "annotate": "annotate",  # Retry if invalid and attempts remain
                "evaluate": "evaluate",  # Proceed if valid
                "end": END,  # End if max attempts reached
            },
        )

        # After evaluation, route based on faithfulness
        workflow.add_conditional_edges(
            "evaluate",
            self._route_after_evaluation,
            {
                "annotate": "annotate",  # Refine if not faithful
                "assess": "assess",  # Proceed to assessment if faithful
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
        print(f"[WORKFLOW] Entering annotate node (attempt {state['validation_attempts']})")
        result = await self.annotation_agent.annotate(state)
        print(f"[WORKFLOW] Annotation generated: {result.get('current_annotation', '')[:100]}...")
        return result

    async def _validate_node(self, state: HedAnnotationState) -> dict:
        """Validation node: Validate HED annotation.

        Args:
            state: Current workflow state

        Returns:
            State update
        """
        print(f"[WORKFLOW] Entering validate node")
        result = await self.validation_agent.validate(state)
        print(f"[WORKFLOW] Validation result: {result.get('validation_status')}, is_valid: {result.get('is_valid')}")
        if not result.get('is_valid'):
            print(f"[WORKFLOW] Validation errors: {result.get('validation_errors', [])}")
        return result

    async def _evaluate_node(self, state: HedAnnotationState) -> dict:
        """Evaluation node: Evaluate annotation faithfulness.

        Args:
            state: Current workflow state

        Returns:
            State update
        """
        print(f"[WORKFLOW] Entering evaluate node")
        result = await self.evaluation_agent.evaluate(state)
        print(f"[WORKFLOW] Evaluation result: is_faithful={result.get('is_faithful')}")
        return result

    async def _assess_node(self, state: HedAnnotationState) -> dict:
        """Assessment node: Final assessment.

        Args:
            state: Current workflow state

        Returns:
            State update
        """
        return await self.assessment_agent.assess(state)

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
            print(f"[WORKFLOW] Routing to evaluate (validation passed)")
            return "evaluate"
        elif state["validation_status"] == "max_attempts_reached":
            print(f"[WORKFLOW] Routing to end (max validation attempts reached)")
            return "end"
        else:
            print(f"[WORKFLOW] Routing to annotate (validation failed, attempts: {state['validation_attempts']}/{state['max_validation_attempts']})")
            return "annotate"

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
        if state["is_faithful"]:
            print(f"[WORKFLOW] Routing to assess (annotation is faithful)")
            return "assess"
        else:
            print(f"[WORKFLOW] Routing to annotate (annotation needs refinement)")
            return "annotate"

    async def run(
        self,
        input_description: str,
        schema_version: str = "8.3.0",
        max_validation_attempts: int = 5,
        config: dict | None = None,
    ) -> HedAnnotationState:
        """Run the complete annotation workflow.

        Args:
            input_description: Natural language event description
            schema_version: HED schema version to use
            max_validation_attempts: Maximum validation retry attempts
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
        )

        # Run workflow
        final_state = await self.graph.ainvoke(initial_state, config=config)

        return final_state
