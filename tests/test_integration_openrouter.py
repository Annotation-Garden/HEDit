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
        from src.agents.annotation_agent import AnnotationAgent
        from src.utils.openrouter_llm import create_openrouter_llm

        llm = create_openrouter_llm(
            model=TEST_MODEL,
            api_key=test_api_key,
            temperature=0.1,
            max_tokens=1000,
            provider=TEST_PROVIDER if TEST_PROVIDER else None,
        )

        # Always use None to fetch schemas from GitHub via HED library
        # This ensures tests are consistent regardless of local setup
        return AnnotationAgent(llm=llm, schema_dir=None)

    @pytest.mark.asyncio
    async def test_annotation_generates_hed_tags(self, annotation_agent) -> None:
        """Test that annotation agent generates HED-like output."""
        from src.agents.state import create_initial_state

        state = create_initial_state(
            input_description="A red light flashes on the screen",
            schema_version="8.3.0",
        )

        # Retry up to 3 times in case of empty LLM response (rate limiting, etc.)
        max_retries = 3
        annotation = None
        for _attempt in range(max_retries):
            result = await annotation_agent.annotate(state)
            assert "current_annotation" in result
            annotation = result["current_annotation"]
            if annotation and len(annotation) > 0:
                break
            # Small delay before retry
            import asyncio

            await asyncio.sleep(1)

        # Check it looks like HED (contains commas, parentheses, or HED keywords)
        assert annotation is not None, "Annotation is None after retries"
        assert len(annotation) > 0, f"Empty annotation after {max_retries} retries"
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
        from src.agents.evaluation_agent import EvaluationAgent
        from src.utils.openrouter_llm import create_openrouter_llm

        llm = create_openrouter_llm(
            model=TEST_MODEL,
            api_key=test_api_key,
            temperature=0.1,
            max_tokens=500,
            provider=TEST_PROVIDER if TEST_PROVIDER else None,
        )

        # Always use None to fetch schemas from GitHub via HED library
        return EvaluationAgent(llm=llm, schema_dir=None)

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
        from src.agents.workflow import HedAnnotationWorkflow
        from src.utils.openrouter_llm import create_openrouter_llm

        llm = create_openrouter_llm(
            model=TEST_MODEL,
            api_key=test_api_key,
            temperature=0.1,
            max_tokens=1000,
            provider=TEST_PROVIDER if TEST_PROVIDER else None,
        )

        # Always use None to fetch schemas from GitHub via HED library
        # Use Python validator (no JS validator path needed)
        return HedAnnotationWorkflow(
            llm=llm,
            schema_dir=None,
            validator_path=None,
            use_js_validator=False,  # Use Python validator
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
class TestVisionAgentIntegration:
    """Test the vision agent with real vision LLM calls.

    Note: These tests use example images from the examples/ directory.
    Vision model calls may take longer than text-only calls.
    """

    @pytest.fixture
    def vision_agent(self, test_api_key: str):
        """Create a vision agent for testing."""
        from src.agents.vision_agent import VisionAgent
        from src.utils.openrouter_llm import create_openrouter_llm

        # Use default vision model
        vision_model = os.getenv("VISION_MODEL", "qwen/qwen3-vl-30b-a3b-instruct")

        llm = create_openrouter_llm(
            model=vision_model,
            api_key=test_api_key,
            temperature=0.1,
            max_tokens=500,
            # Vision models don't use Cerebras provider
            provider=None,
        )

        return VisionAgent(llm=llm)

    @pytest.fixture
    def example_image_base64(self) -> str:
        """Load example image as base64 data URI."""
        import base64
        from io import BytesIO
        from pathlib import Path

        from PIL import Image

        # Use the first example image
        image_path = Path(__file__).parent.parent / "examples" / "shared0001_nsd02951.jpg"
        if not image_path.exists():
            pytest.skip(f"Example image not found: {image_path}")

        img = Image.open(image_path)
        buffer = BytesIO()
        img_format = img.format or "JPEG"
        img.save(buffer, format=img_format)
        buffer.seek(0)

        base64_str = base64.b64encode(buffer.read()).decode("utf-8")
        mime_type = f"image/{img_format.lower()}"
        return f"data:{mime_type};base64,{base64_str}"

    @pytest.mark.asyncio
    async def test_vision_agent_describes_image(self, vision_agent, example_image_base64) -> None:
        """Test that vision agent generates a description from an image."""
        result = await vision_agent.describe_image(example_image_base64)

        assert result is not None
        assert "description" in result
        assert "prompt_used" in result
        assert "metadata" in result

        # Description should be non-empty
        description = result["description"]
        assert description is not None
        assert len(description) > 20  # Should be a meaningful description

        # Metadata should have image info
        metadata = result["metadata"]
        assert "width" in metadata
        assert "height" in metadata
        assert "format" in metadata

    @pytest.mark.asyncio
    async def test_vision_agent_with_custom_prompt(
        self, vision_agent, example_image_base64
    ) -> None:
        """Test vision agent with a custom prompt."""
        custom_prompt = "List the main objects visible in this image in a comma-separated list."

        result = await vision_agent.describe_image(
            example_image_base64,
            custom_prompt=custom_prompt,
        )

        assert result is not None
        assert result["prompt_used"] == custom_prompt
        assert "description" in result
        # Custom prompt response should contain commas (list format)
        assert len(result["description"]) > 0


@pytest.mark.integration
@pytest.mark.skipif(not OPENROUTER_TEST_KEY, reason=SKIP_REASON)
class TestImageProcessingIntegration:
    """Test image processing utilities with real images."""

    @pytest.fixture
    def example_image_path(self):
        """Get path to example image."""
        from pathlib import Path

        image_path = Path(__file__).parent.parent / "examples" / "shared0060_nsd06432.jpg"
        if not image_path.exists():
            pytest.skip(f"Example image not found: {image_path}")
        return image_path

    def test_decode_base64_image(self, example_image_path) -> None:
        """Test decoding a real image from base64."""
        import base64
        from io import BytesIO

        from PIL import Image

        from src.utils.image_processing import decode_base64_image

        # Load and encode image
        img = Image.open(example_image_path)
        buffer = BytesIO()
        img.save(buffer, format=img.format or "JPEG")
        buffer.seek(0)
        base64_str = base64.b64encode(buffer.read()).decode("utf-8")

        # Decode and validate
        decoded_img, metadata = decode_base64_image(base64_str)

        assert decoded_img is not None
        assert metadata["width"] == img.width
        assert metadata["height"] == img.height
        assert metadata["format"] in ["JPEG", "JPG"]

    def test_validate_image_data(self, example_image_path) -> None:
        """Test image validation with a real image."""
        import base64
        from io import BytesIO

        from PIL import Image

        from src.utils.image_processing import validate_image_data

        # Load and encode image as data URI
        img = Image.open(example_image_path)
        buffer = BytesIO()
        img_format = img.format or "JPEG"
        img.save(buffer, format=img_format)
        buffer.seek(0)
        base64_str = base64.b64encode(buffer.read()).decode("utf-8")
        data_uri = f"data:image/{img_format.lower()};base64,{base64_str}"

        # Validate
        result = validate_image_data(data_uri)

        assert result["valid"] is True
        assert result["error"] is None
        assert result["metadata"] is not None
        assert result["metadata"]["width"] > 0
        assert result["metadata"]["height"] > 0

    def test_prepare_image_for_vision_model(self, example_image_path) -> None:
        """Test preparing an image for vision model processing."""
        import base64
        from io import BytesIO

        from PIL import Image

        from src.utils.image_processing import prepare_image_for_vision_model

        # Load and encode image
        img = Image.open(example_image_path)
        buffer = BytesIO()
        img_format = img.format or "JPEG"
        img.save(buffer, format=img_format)
        buffer.seek(0)
        base64_str = base64.b64encode(buffer.read()).decode("utf-8")

        # Prepare for vision model
        data_uri, metadata = prepare_image_for_vision_model(base64_str)

        assert data_uri.startswith("data:image/")
        assert ";base64," in data_uri
        assert metadata["width"] == img.width
        assert metadata["height"] == img.height


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
        import importlib

        from fastapi.testclient import TestClient

        # Set environment variables for the test BEFORE importing app
        os.environ["LLM_PROVIDER"] = "openrouter"
        os.environ["OPENROUTER_API_KEY"] = test_api_key
        os.environ["ANNOTATION_MODEL"] = TEST_MODEL
        os.environ["EVALUATION_MODEL"] = os.getenv("EVALUATION_MODEL", TEST_MODEL)
        os.environ["ASSESSMENT_MODEL"] = os.getenv("ASSESSMENT_MODEL", TEST_MODEL)
        os.environ["FEEDBACK_MODEL"] = os.getenv("FEEDBACK_MODEL", TEST_MODEL)
        if TEST_PROVIDER:
            os.environ["LLM_PROVIDER_PREFERENCE"] = TEST_PROVIDER
        # Use API key auth to avoid issues with module caching
        os.environ["REQUIRE_API_AUTH"] = "true"
        os.environ["API_KEYS"] = "integration-test-api-key"
        os.environ["USE_JS_VALIDATOR"] = "false"

        # Clear schema paths - app now handles None gracefully and fetches from GitHub
        for key in ["HED_SCHEMA_DIR", "HED_VALIDATOR_PATH"]:
            if key in os.environ:
                del os.environ[key]

        # Reload security module to pick up new env vars
        # (needed when running after other tests that modified security state)
        from src.api import security

        importlib.reload(security)

        # Also reload main to pick up the new security module
        from src.api import main

        importlib.reload(main)

        with TestClient(main.app) as client:
            yield client

    # Auth header for authenticated requests
    AUTH_HEADERS = {"X-API-Key": "integration-test-api-key"}

    def test_health_endpoint(self, client) -> None:
        """Test the health endpoint works (no auth required)."""
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
            headers=self.AUTH_HEADERS,
        )

        # Check response structure (may fail validation but should return result)
        assert response.status_code == 200
        data = response.json()
        assert "annotation" in data
        assert "is_valid" in data
        assert "status" in data
