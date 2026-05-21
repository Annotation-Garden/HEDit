"""LangGraph workflow for HED annotation generation.

This module defines the multi-agent workflow that orchestrates
annotation, validation, evaluation, and assessment.
"""

import logging
import time
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from src.agents.annotation_agent import AnnotationAgent
from src.agents.assessment_agent import AssessmentAgent
from src.agents.evaluation_agent import EvaluationAgent
from src.agents.feedback_summarizer import FeedbackSummarizer
from src.agents.state import HedAnnotationState
from src.agents.validation_agent import ValidationAgent
from src.lsp import HedLspClient
from src.utils import extract_text_content
from src.utils.schema_loader import HedSchemaLoader

logger = logging.getLogger(__name__)


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
        enable_semantic_search: bool = True,
        lsp_client: HedLspClient | None = None,
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
            enable_semantic_search: Whether to include the semantic_preprocess
                node in the graph. The node is only useful when an
                `lsp_client` is also provided.
            lsp_client: Pre-built, already-initialized HedLspClient. The
                caller (FastAPI lifespan, CLI executor) owns the client's
                lifetime. None disables LSP-backed tag enrichment.
        """
        # Store schema directory (None means use HED library to fetch from GitHub)
        self.schema_dir = schema_dir
        # Keyword extraction always runs; LSP enrichment requires an injected client
        self.enable_semantic_search = enable_semantic_search
        self.lsp_client: HedLspClient | None = lsp_client

        # Initialize legacy schema loader for validation
        self.schema_loader = HedSchemaLoader()

        # Use provided LLMs or default to main llm
        eval_llm = evaluation_llm or llm
        assess_llm = assessment_llm or llm
        feed_llm = feedback_llm or llm

        # Store feedback LLM for keyword extraction (cheap/fast model)
        self.feedback_llm = feed_llm

        # Initialize agents with JSON schema support and per-agent LLMs
        self.annotation_agent = AnnotationAgent(llm, schema_dir=self.schema_dir)
        self.validation_agent = ValidationAgent(
            self.schema_loader,
            use_javascript=use_js_validator,
            validator_path=validator_path,
            lsp_client=self.lsp_client,
        )
        self.evaluation_agent = EvaluationAgent(eval_llm, schema_dir=self.schema_dir)
        self.assessment_agent = AssessmentAgent(assess_llm, schema_dir=self.schema_dir)
        self.feedback_summarizer = FeedbackSummarizer(feed_llm)

        if self.enable_semantic_search and self.lsp_client is not None:
            logger.info("[WORKFLOW] hed-lsp client connected for semantic tag suggestions")
        elif self.enable_semantic_search:
            logger.info(
                "[WORKFLOW] semantic_preprocess enabled but no lsp_client provided; "
                "keyword extraction will run without LSP enrichment"
            )

        # Build graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow.

        Returns:
            Compiled StateGraph
        """
        # Create graph
        workflow = StateGraph(HedAnnotationState)  # type: ignore[arg-type]  # LangGraph typing limitation

        # Add nodes
        if self.enable_semantic_search:
            workflow.add_node("semantic_preprocess", self._semantic_preprocess_node)
        workflow.add_node("annotate", self._annotate_node)
        workflow.add_node("validate", self._validate_node)
        workflow.add_node("summarize_feedback", self._summarize_feedback_node)
        workflow.add_node("evaluate", self._evaluate_node)
        workflow.add_node("assess", self._assess_node)

        # Add edges
        if self.enable_semantic_search:
            workflow.set_entry_point("semantic_preprocess")
            workflow.add_edge("semantic_preprocess", "annotate")
        else:
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

        return workflow.compile()  # type: ignore[return-value]

    async def _extract_keywords(self, description: str) -> list[str]:
        """Extract HED-relevant keywords from a natural language description.

        Uses the feedback LLM (cheap/fast model) to identify key concepts
        that can be mapped to HED tags via the LSP suggest tool.

        Args:
            description: Natural language event or image description

        Returns:
            List of extracted keywords (max 20)
        """
        system_prompt = (
            "You are a keyword extractor for neuroscience event descriptions. "
            "Extract the most important concepts that could map to HED "
            "(Hierarchical Event Descriptors) tags.\n\n"
            "Extract:\n"
            "- Objects/entities (person, car, button, screen, face, etc.)\n"
            "- Actions/events (pressing, flashing, appearing, moving, etc.)\n"
            "- Properties/attributes (red, large, fast, loud, etc.)\n"
            "- Spatial relationships (left, center, above, etc.)\n"
            "- Temporal aspects (onset, offset, duration, etc.)\n"
            "- Sensory modalities (visual, auditory, tactile, etc.)\n\n"
            "Return ONLY a comma-separated list of single words or short phrases "
            "(2-3 words max). Return at most 20 keywords. "
            "Do not include any other text, explanation, or formatting."
        )

        try:
            response = await self.feedback_llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Description: {description}"),
                ]
            )
            raw_text = extract_text_content(response.content)
            # Parse comma-separated keywords, strip whitespace, filter empty
            keywords = [kw.strip() for kw in raw_text.split(",") if kw.strip()]
            # Limit to 20 keywords
            keywords = keywords[:20]
            logger.info(f"[WORKFLOW] Extracted {len(keywords)} keywords: {keywords}")
            return keywords
        except Exception as e:
            logger.warning("[WORKFLOW] Keyword extraction failed: %s", e, exc_info=True)
            return []

    async def _semantic_preprocess_node(self, state: HedAnnotationState) -> dict:
        """Semantic preprocessing node: Extract keywords and suggest HED tags.

        This node runs before annotation to provide semantic hints based on
        the input description. It uses the feedback LLM to extract keywords,
        then passes those keywords to hed-lsp CLI for tag suggestions.
        Only runs on the first iteration.

        Args:
            state: Current workflow state

        Returns:
            State update with extracted_keywords and semantic_hints
        """
        # Only run preprocessing on first iteration
        if state.get("total_iterations", 0) > 0:
            logger.debug("[WORKFLOW] Skipping semantic preprocessing (not first iteration)")
            return {}

        logger.info("[WORKFLOW] Entering semantic_preprocess node")

        # Step 1: Extract keywords from the description using LLM
        keywords = await self._extract_keywords(state["input_description"])

        # Step 2: One batched hed/suggest call over the persistent LSP
        # connection. The previous per-keyword loop spawned the hed-suggest
        # CLI once per keyword and dominated request latency (~6 s/spawn).
        semantic_hints: list[dict] = []

        if keywords and self.lsp_client is not None:
            try:
                result = await self.lsp_client.suggest(*keywords)
            except Exception as e:
                logger.warning("[WORKFLOW] hed-lsp suggest failed: %s", e)
                result = None

            if result is not None and result.success:
                seen_tags: dict[str, dict] = {}
                for keyword, tags in result.raw.items():
                    for idx, tag in enumerate(tags):
                        # Higher score for earlier (more relevant) hits.
                        score = max(0.0, 1.0 - idx * 0.05)
                        existing = seen_tags.get(tag)
                        if existing is None or score > existing["score"]:
                            seen_tags[tag] = {
                                "tag": tag,
                                "keyword": keyword,
                                "score": score,
                                "source": "hed-lsp",
                            }
                semantic_hints = sorted(seen_tags.values(), key=lambda h: h["score"], reverse=True)
                logger.info(
                    "[WORKFLOW] hed-lsp suggested %d unique tags from %d keywords",
                    len(semantic_hints),
                    len(keywords),
                )
            elif result is not None:
                logger.debug("[WORKFLOW] hed-lsp suggest returned failure: %s", result.error)
        elif keywords:
            logger.info(
                "[WORKFLOW] no lsp_client; storing %d extracted keywords without enrichment",
                len(keywords),
            )

        return {
            "extracted_keywords": keywords,
            "semantic_hints": semantic_hints,
        }

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
        t0 = time.monotonic()
        result = await self.annotation_agent.annotate(state)
        elapsed = time.monotonic() - t0
        result["total_iterations"] = total_iters  # Increment counter
        print(
            f"[WORKFLOW] Annotation generated in {elapsed:.1f}s: {result.get('current_annotation', '')[:100]}..."
        )
        return result

    async def _validate_node(self, state: HedAnnotationState) -> dict:
        """Validation node: Validate HED annotation.

        Args:
            state: Current workflow state

        Returns:
            State update
        """
        print("[WORKFLOW] Entering validate node")
        t0 = time.monotonic()
        result = await self.validation_agent.validate(state)
        elapsed = time.monotonic() - t0
        print(
            f"[WORKFLOW] Validation result in {elapsed:.1f}s: {result.get('validation_status')}, is_valid: {result.get('is_valid')}"
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
        t0 = time.monotonic()
        result = await self.evaluation_agent.evaluate(state)
        elapsed = time.monotonic() - t0
        print(
            f"[WORKFLOW] Evaluation result in {elapsed:.1f}s: is_faithful={result.get('is_faithful')}"
        )

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
        print("[WORKFLOW] Entering assess node")
        t0 = time.monotonic()
        result = await self.assessment_agent.assess(state)
        elapsed = time.monotonic() - t0
        print(f"[WORKFLOW] Assessment completed in {elapsed:.1f}s")
        return result

    async def _summarize_feedback_node(self, state: HedAnnotationState) -> dict:
        """Summarize feedback node: Condense errors and feedback.

        Args:
            state: Current workflow state

        Returns:
            State update with summarized feedback
        """
        print("[WORKFLOW] Entering summarize_feedback node")
        t0 = time.monotonic()
        result = await self.feedback_summarizer.summarize(state)
        elapsed = time.monotonic() - t0
        print(
            f"[WORKFLOW] Feedback summarized in {elapsed:.1f}s: {result.get('validation_errors_augmented', [''])[0][:100] if result.get('validation_errors_augmented') else 'No feedback'}..."
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
        """Route after evaluation based on faithfulness and assessment mode.

        When run_assessment=False (default), evaluation is informational only;
        the result is reported but never triggers refinement loops.
        When run_assessment=True, evaluation can trigger refinement and the
        assessment node runs at the end.

        Args:
            state: Current workflow state

        Returns:
            Next node name
        """
        run_assessment = state.get("run_assessment", False)

        # When assessment is off, evaluation is informational -- always end
        if not run_assessment:
            print(
                f"[WORKFLOW] Evaluation complete (informational, is_faithful={state['is_faithful']}) - routing to END"
            )
            return "end"

        # Assessment mode: allow refinement loops with iteration cap
        total_iters = state.get("total_iterations", 0)
        max_iters = state.get("max_total_iterations", 4)

        if total_iters >= max_iters:
            print(f"[WORKFLOW] Routing to assess (max total iterations {max_iters} reached)")
            return "assess"

        if state["is_faithful"]:
            print("[WORKFLOW] Routing to assess (annotation is faithful)")
            return "assess"
        else:
            print(
                f"[WORKFLOW] Routing to summarize_feedback (annotation needs refinement, iteration {total_iters}/{max_iters})"
            )
            return "summarize_feedback"

    async def run(
        self,
        input_description: str,
        schema_version: str = "8.4.0",
        max_validation_attempts: int = 3,
        max_total_iterations: int | None = None,
        run_assessment: bool = False,
        no_extend: bool = False,
        config: dict | None = None,
    ) -> HedAnnotationState:
        """Run the complete annotation workflow.

        Args:
            input_description: Natural language event description
            schema_version: HED schema version to use
            max_validation_attempts: Maximum validation retry attempts
            max_total_iterations: Maximum total iterations (default: max_validation_attempts + 1)
            run_assessment: Whether to run final assessment (default: False)
            no_extend: If True, prohibit tag extensions (use only existing vocabulary)
            config: Optional LangGraph config (e.g., recursion_limit)

        Returns:
            Final workflow state with annotation and feedback
        """
        from src.agents.state import create_initial_state

        if max_total_iterations is None:
            max_total_iterations = max_validation_attempts + 1

        # Create initial state
        initial_state = create_initial_state(
            input_description,
            schema_version,
            max_validation_attempts,
            max_total_iterations,
            run_assessment,
            no_extend=no_extend,
        )

        # Run workflow
        final_state = await self.graph.ainvoke(initial_state, config=config)  # type: ignore[attr-defined]

        return final_state
