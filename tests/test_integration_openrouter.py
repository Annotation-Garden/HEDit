"""Integration tests that make real calls to OpenRouter.

These tests use OPENROUTER_API_KEY_FOR_TESTING to track testing costs separately.
Tests are skipped if the key is not present (for local development without API key).

Run with: pytest tests/test_integration_openrouter.py -v
Run all tests including integration: pytest -v
Skip integration tests: pytest -v -m "not integration"
"""

import os

import pytest
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if OpenRouter testing key is available
OPENROUTER_TEST_KEY = os.getenv("OPENROUTER_API_KEY_FOR_TESTING")
SKIP_REASON = "OPENROUTER_API_KEY_FOR_TESTING not set"

# Use the same models as configured in .env for consistency
# Default to the environment-configured models
TEST_MODEL = os.getenv("ANNOTATION_MODEL", "openai/gpt-oss-120b")
TEST_PROVIDER = os.getenv("LLM_PROVIDER_PREFERENCE", "Cerebras")


@pytest.fixture
def test_api_key() -> str:
    """Get OpenRouter API key for testing."""
    if not OPENROUTER_TEST_KEY:
        pytest.skip(SKIP_REASON)
    return OPENROUTER_TEST_KEY


@pytest.fixture
def test_llm(test_api_key: str):
    """Create an LLM instance for testing using env-configured model."""
    from src.utils.openrouter_llm import create_openrouter_llm

    return create_openrouter_llm(
        model=TEST_MODEL,
        api_key=test_api_key,
        temperature=0.1,
        max_tokens=500,
        provider=TEST_PROVIDER if TEST_PROVIDER else None,
    )


@pytest.mark.integration
@pytest.mark.skipif(not OPENROUTER_TEST_KEY, reason=SKIP_REASON)
class TestOpenRouterConnection:
    """Test that we can connect to OpenRouter and get responses."""

    @pytest.mark.asyncio
    async def test_basic_llm_call(self, test_llm) -> None:
        """Test a basic LLM call returns a response."""
        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content="Say 'hello' and nothing else.")]
        response = await test_llm.ainvoke(messages)

        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
        assert "hello" in response.content.lower()

    @pytest.mark.asyncio
    async def test_llm_follows_instructions(self, test_llm) -> None:
        """Test that LLM follows specific instructions."""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content="You must respond with exactly one word."),
            HumanMessage(content="What color is the sky on a clear day?"),
        ]
        response = await test_llm.ainvoke(messages)

        assert response is not None
        assert response.content is not None
        # Response should be short (one word instruction)
        assert len(response.content.split()) <= 3


@pytest.mark.integration
@pytest.mark.skipif(not OPENROUTER_TEST_KEY, reason=SKIP_REASON)
class TestAnnotationAgentIntegration:
    """Test the annotation agent with real LLM calls."""

    @pytest.fixture
    def annotation_agent(self, test_api_key: str):
        """Create an annotation agent for testing using env-configured model."""
        from pathlib import Path

        from src.agents.annotation_agent import AnnotationAgent
        from src.utils.openrouter_llm import create_openrouter_llm

        llm = create_openrouter_llm(
            model=TEST_MODEL,
            api_key=test_api_key,
            temperature=0.1,
            max_tokens=1000,
            provider=TEST_PROVIDER if TEST_PROVIDER else None,
        )

        # Use default schema directory
        schema_dir = Path.home() / "Documents/git/HED/hed-schemas/schemas_latest_json"

        return AnnotationAgent(llm=llm, schema_dir=schema_dir)

    @pytest.mark.asyncio
    async def test_annotation_generates_hed_tags(self, annotation_agent) -> None:
        """Test that annotation agent generates HED-like output."""
        from src.agents.state import create_initial_state

        state = create_initial_state(
            input_description="A red light flashes on the screen",
            schema_version="8.3.0",
        )

        result = await annotation_agent.annotate(state)

        assert "current_annotation" in result
        annotation = result["current_annotation"]

        # Check it looks like HED (contains commas, parentheses, or HED keywords)
        assert annotation is not None
        assert len(annotation) > 0
        # HED annotations typically contain commas or parentheses
        has_hed_structure = "," in annotation or "(" in annotation
        # Or contain common HED tags
        has_hed_keywords = any(
            kw in annotation for kw in ["Sensory", "Visual", "Event", "Red", "Light", "Screen"]
        )
        assert has_hed_structure or has_hed_keywords, f"Output doesn't look like HED: {annotation}"


@pytest.mark.integration
@pytest.mark.skipif(not OPENROUTER_TEST_KEY, reason=SKIP_REASON)
class TestEvaluationAgentIntegration:
    """Test the evaluation agent with real LLM calls."""

    @pytest.fixture
    def evaluation_agent(self, test_api_key: str):
        """Create an evaluation agent for testing using env-configured model."""
        from pathlib import Path

        from src.agents.evaluation_agent import EvaluationAgent
        from src.utils.openrouter_llm import create_openrouter_llm

        llm = create_openrouter_llm(
            model=TEST_MODEL,
            api_key=test_api_key,
            temperature=0.1,
            max_tokens=500,
            provider=TEST_PROVIDER if TEST_PROVIDER else None,
        )

        schema_dir = Path.home() / "Documents/git/HED/hed-schemas/schemas_latest_json"

        return EvaluationAgent(llm=llm, schema_dir=schema_dir)

    @pytest.mark.asyncio
    async def test_evaluation_returns_feedback(self, evaluation_agent) -> None:
        """Test that evaluation agent provides feedback."""
        from src.agents.state import create_initial_state

        state = create_initial_state(
            input_description="A person presses a button",
            schema_version="8.3.0",
        )
        state["current_annotation"] = "Agent-action, (Press, Button)"
        state["is_valid"] = True
        state["validation_errors"] = []

        result = await evaluation_agent.evaluate(state)

        assert "is_faithful" in result
        assert "evaluation_feedback" in result
        assert isinstance(result["is_faithful"], bool)
        assert result["evaluation_feedback"] is not None


@pytest.mark.integration
@pytest.mark.skipif(not OPENROUTER_TEST_KEY, reason=SKIP_REASON)
class TestWorkflowIntegration:
    """Test the complete annotation workflow with real LLM calls.

    Note: This test makes multiple LLM calls and may take longer.
    """

    @pytest.fixture
    def workflow(self, test_api_key: str):
        """Create a workflow for testing using env-configured model."""
        from pathlib import Path

        from src.agents.workflow import HedAnnotationWorkflow
        from src.utils.openrouter_llm import create_openrouter_llm

        llm = create_openrouter_llm(
            model=TEST_MODEL,
            api_key=test_api_key,
            temperature=0.1,
            max_tokens=1000,
            provider=TEST_PROVIDER if TEST_PROVIDER else None,
        )

        schema_dir = Path.home() / "Documents/git/HED/hed-schemas/schemas_latest_json"
        validator_path = Path.home() / "Documents/git/HED/hed-javascript"

        return HedAnnotationWorkflow(
            llm=llm,
            schema_dir=schema_dir,
            validator_path=validator_path,
            use_js_validator=False,  # Use Python validator for speed
        )

    @pytest.mark.asyncio
    async def test_simple_annotation_workflow(self, workflow) -> None:
        """Test a simple annotation through the full workflow."""
        result = await workflow.run(
            input_description="A visual stimulus appears on the screen",
            schema_version="8.3.0",
            max_validation_attempts=3,
            max_total_iterations=5,
            run_assessment=False,
        )

        # Check that workflow completed and returned results
        assert result is not None
        assert "current_annotation" in result
        assert "is_valid" in result
        assert "validation_attempts" in result

        # Annotation should be non-empty
        assert result["current_annotation"] is not None
        assert len(result["current_annotation"]) > 0

        # Check for HED-like structure
        annotation = result["current_annotation"]
        has_structure = "," in annotation or "(" in annotation or "-" in annotation
        assert has_structure, f"Annotation doesn't look like HED: {annotation}"


@pytest.mark.integration
@pytest.mark.skipif(not OPENROUTER_TEST_KEY, reason=SKIP_REASON)
class TestAPIEndpointIntegration:
    """Test API endpoints with real LLM calls.

    Note: These tests require the full API lifespan to initialize.
    The TestClient triggers lifespan events when used as a context manager.
    """

    @pytest.fixture
    def client(self, test_api_key: str):
        """Create a test client for the API using env-configured models."""
        import os
        from pathlib import Path

        from fastapi.testclient import TestClient

        # Set environment variables for the test BEFORE importing app
        # Use the testing API key but keep other env settings
        os.environ["LLM_PROVIDER"] = "openrouter"
        os.environ["OPENROUTER_API_KEY"] = test_api_key
        # Use env-configured models (loaded from .env via dotenv)
        os.environ["ANNOTATION_MODEL"] = TEST_MODEL
        os.environ["EVALUATION_MODEL"] = os.getenv("EVALUATION_MODEL", TEST_MODEL)
        os.environ["ASSESSMENT_MODEL"] = os.getenv("ASSESSMENT_MODEL", TEST_MODEL)
        os.environ["FEEDBACK_MODEL"] = os.getenv("FEEDBACK_MODEL", TEST_MODEL)
        if TEST_PROVIDER:
            os.environ["LLM_PROVIDER_PREFERENCE"] = TEST_PROVIDER
        os.environ["REQUIRE_API_AUTH"] = "false"
        os.environ["USE_JS_VALIDATOR"] = "false"

        # Set local schema paths for testing (override Docker defaults)
        schema_dir = Path.home() / "Documents/git/HED/hed-schemas/schemas_latest_json"
        validator_path = Path.home() / "Documents/git/HED/hed-javascript"
        os.environ["HED_SCHEMA_DIR"] = str(schema_dir)
        os.environ["HED_VALIDATOR_PATH"] = str(validator_path)

        from src.api.main import app

        # Use context manager to ensure lifespan events are triggered
        with TestClient(app) as client:
            yield client

    def test_health_endpoint(self, client) -> None:
        """Test the health endpoint works."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]

    def test_annotate_endpoint(self, client) -> None:
        """Test the annotation endpoint with a real request."""
        response = client.post(
            "/annotate",
            json={
                "description": "A beep sound plays",
                "schema_version": "8.3.0",
                "max_validation_attempts": 2,
                "run_assessment": False,
            },
        )

        # Check response structure (may fail validation but should return result)
        assert response.status_code == 200
        data = response.json()
        assert "annotation" in data
        assert "is_valid" in data
        assert "status" in data
