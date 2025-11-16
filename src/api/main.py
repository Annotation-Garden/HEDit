"""FastAPI application for HED-BOT annotation service.

This module provides REST API endpoints for HED annotation generation
and validation using the multi-agent workflow.
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_community.chat_models import ChatOllama
from langchain_core.language_models import BaseChatModel

from src import __version__
from src.agents.workflow import HedAnnotationWorkflow
from src.utils.openrouter_llm import create_openrouter_llm, get_model_name
from src.api.models import (
    AnnotationRequest,
    AnnotationResponse,
    HealthResponse,
    ValidationRequest,
    ValidationResponse,
)
from src.utils.schema_loader import HedSchemaLoader
from src.validation.hed_validator import HedPythonValidator

# Global workflow instance
workflow: HedAnnotationWorkflow | None = None
schema_loader: HedSchemaLoader | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan (startup and shutdown).

    Args:
        app: FastAPI application
    """
    global workflow, schema_loader

    # Startup: Initialize workflow
    print("Initializing HED-BOT annotation workflow...")

    # Auto-detect environment (Docker vs local)
    def get_default_path(docker_path: str, local_path: str) -> str:
        """Get default path based on environment.

        Args:
            docker_path: Path to use in Docker
            local_path: Path to use in local development

        Returns:
            Appropriate default path
        """
        # Check if running in Docker (look for Docker-specific paths)
        if Path("/app").exists() and Path(docker_path).exists():
            return docker_path
        # Check if local development path exists
        elif Path(local_path).exists():
            return local_path
        # Fall back to Docker path (will fail gracefully if not available)
        return docker_path

    # Get configuration from environment with smart defaults
    llm_provider = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "openrouter"
    llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    # Schema directory with environment detection
    schema_dir = os.getenv(
        "HED_SCHEMA_DIR",
        get_default_path(
            "/app/hed-schemas/schemas_latest_json",  # Docker
            str(Path.home() / "git/hed-schemas/schemas_latest_json"),  # Local Linux/macOS
        ),
    )

    # Validator path with environment detection
    validator_path = os.getenv(
        "HED_VALIDATOR_PATH",
        get_default_path(
            "/app/hed-javascript",  # Docker
            str(Path.home() / "git/hed-javascript"),  # Local Linux/macOS
        ),
    )

    use_js_validator = os.getenv("USE_JS_VALIDATOR", "true").lower() == "true"

    print(f"Environment: {'Docker' if Path('/app').exists() else 'Local'}")
    print(f"Schema directory: {schema_dir}")
    print(f"Validator path: {validator_path}")

    # Initialize LLM based on provider
    if llm_provider == "openrouter":
        # OpenRouter configuration
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required when using OpenRouter")

        # Provider preference (e.g., "Cerebras" for ultra-fast inference)
        provider_preference = os.getenv("LLM_PROVIDER_PREFERENCE")

        # Per-agent model configuration
        annotation_model = get_model_name(os.getenv("ANNOTATION_MODEL", "gpt-5-mini"))
        evaluation_model = get_model_name(os.getenv("EVALUATION_MODEL", "gpt-5-mini"))
        assessment_model = get_model_name(os.getenv("ASSESSMENT_MODEL", "gpt-5-mini"))
        feedback_model = get_model_name(os.getenv("FEEDBACK_MODEL", "gpt-5-nano"))

        print(f"Using OpenRouter with models:")
        print(f"  Annotation: {annotation_model}")
        print(f"  Evaluation: {evaluation_model}")
        print(f"  Assessment: {assessment_model}")
        print(f"  Feedback: {feedback_model}")
        if provider_preference:
            print(f"  Provider: {provider_preference} (ultra-fast)")

        # Create LLMs for each agent
        annotation_llm = create_openrouter_llm(
            model=annotation_model,
            api_key=openrouter_api_key,
            temperature=llm_temperature,
            provider=provider_preference,
        )
        evaluation_llm = create_openrouter_llm(
            model=evaluation_model,
            api_key=openrouter_api_key,
            temperature=llm_temperature,
            provider=provider_preference,
        )
        assessment_llm = create_openrouter_llm(
            model=assessment_model,
            api_key=openrouter_api_key,
            temperature=llm_temperature,
            provider=provider_preference,
        )
        feedback_llm = create_openrouter_llm(
            model=feedback_model,
            api_key=openrouter_api_key,
            temperature=llm_temperature,
            provider=provider_preference,
        )

        # Use annotation_llm as default
        llm = annotation_llm
    else:
        # Ollama configuration (default)
        llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:11435")
        llm_model = os.getenv("LLM_MODEL", "gpt-oss:20b")

        llm = ChatOllama(
            base_url=llm_base_url,
            model=llm_model,
            temperature=llm_temperature,
        )

        # All agents use same model for Ollama
        annotation_llm = evaluation_llm = assessment_llm = feedback_llm = llm

        print(f"Using Ollama: {llm_model} at {llm_base_url}")

    # Initialize workflow with per-agent LLMs
    workflow = HedAnnotationWorkflow(
        llm=annotation_llm,
        evaluation_llm=evaluation_llm if llm_provider == "openrouter" else None,
        assessment_llm=assessment_llm if llm_provider == "openrouter" else None,
        feedback_llm=feedback_llm if llm_provider == "openrouter" else None,
        schema_dir=Path(schema_dir),
        validator_path=Path(validator_path) if use_js_validator else None,
        use_js_validator=use_js_validator,
    )

    # Set global schema_loader from workflow
    schema_loader = workflow.schema_loader

    print(f"Workflow initialized successfully!")
    print(f"  LLM Provider: {llm_provider} (temperature={llm_temperature})")
    print(f"  JavaScript validator: {use_js_validator}")

    yield

    # Shutdown
    print("Shutting down HED-BOT...")


# Create FastAPI app
app = FastAPI(
    title="HED-BOT API",
    description="Multi-agent system for HED annotation generation and validation",
    version=__version__,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns:
        Health status and service availability
    """
    llm_available = workflow is not None
    validator_available = schema_loader is not None

    status = "healthy" if (llm_available and validator_available) else "degraded"

    return HealthResponse(
        status=status,
        version=__version__,
        llm_available=llm_available,
        validator_available=validator_available,
    )


@app.post("/annotate", response_model=AnnotationResponse)
async def annotate(request: AnnotationRequest) -> AnnotationResponse:
    """Generate HED annotation from natural language description.

    Args:
        request: Annotation request with description and parameters

    Returns:
        Generated annotation with validation and assessment feedback

    Raises:
        HTTPException: If workflow fails
    """
    if workflow is None:
        raise HTTPException(status_code=503, detail="Workflow not initialized")

    try:
        # Run annotation workflow with increased recursion limit for long descriptions
        # LangGraph default is 25, increase to 100 for complex workflows
        config = {"recursion_limit": 100}

        final_state = await workflow.run(
            input_description=request.description,
            schema_version=request.schema_version,
            max_validation_attempts=request.max_validation_attempts,
            run_assessment=request.run_assessment,
            config=config,
        )

        # Determine overall status
        status = "success" if final_state["is_valid"] else "failed"

        return AnnotationResponse(
            annotation=final_state["current_annotation"],
            is_valid=final_state["is_valid"],
            is_faithful=final_state["is_faithful"],
            is_complete=final_state["is_complete"],
            validation_attempts=final_state["validation_attempts"],
            validation_errors=final_state["validation_errors"],
            validation_warnings=final_state["validation_warnings"],
            evaluation_feedback=final_state["evaluation_feedback"],
            assessment_feedback=final_state["assessment_feedback"],
            status=status,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Annotation workflow failed: {str(e)}",
        ) from e


@app.post("/annotate/stream")
async def annotate_stream(request: AnnotationRequest):
    """Generate HED annotation with real-time progress updates via Server-Sent Events.

    This endpoint streams progress updates as the workflow runs through different
    stages (annotation, validation, evaluation, assessment), providing real-time
    feedback to the user.

    Args:
        request: Annotation request with description and parameters

    Returns:
        StreamingResponse with Server-Sent Events

    Raises:
        HTTPException: If workflow fails
    """
    if workflow is None:
        raise HTTPException(status_code=503, detail="Workflow not initialized")

    async def event_generator():
        """Generate SSE events for workflow progress."""
        try:
            # Progress queue for receiving updates from workflow
            progress_queue = asyncio.Queue()

            # Helper to send SSE event
            def send_event(event_type: str, data: dict):
                return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

            # Send initial start event
            yield send_event("progress", {
                "stage": "starting",
                "message": "Initializing annotation workflow..."
            })

            # Run workflow with progress monitoring
            # Note: We'll need to modify workflow to accept progress callback
            # For now, we'll use a simple approach with state polling

            # Start workflow in background task
            import asyncio
            from src.agents.state import create_initial_state

            # Note: create_initial_state is called internally by workflow.run()
            # No need to create it here

            # Track workflow progress by monitoring state changes
            # This is a simplified version - ideally we'd use callbacks
            yield send_event("progress", {
                "stage": "annotating",
                "message": "Generating HED annotation...",
                "attempt": 1
            })

            # Run workflow
            final_state = await workflow.run(
                input_description=request.description,
                schema_version=request.schema_version,
                max_validation_attempts=request.max_validation_attempts,
                run_assessment=request.run_assessment,
            )

            # Send final result
            status = "success" if final_state["is_valid"] else "failed"
            result = {
                "annotation": final_state["current_annotation"],
                "is_valid": final_state["is_valid"],
                "is_faithful": final_state["is_faithful"],
                "is_complete": final_state["is_complete"],
                "validation_attempts": final_state["validation_attempts"],
                "validation_errors": final_state["validation_errors"],
                "validation_warnings": final_state["validation_warnings"],
                "evaluation_feedback": final_state["evaluation_feedback"],
                "assessment_feedback": final_state["assessment_feedback"],
                "status": status,
            }

            yield send_event("result", result)
            yield send_event("done", {"message": "Workflow completed"})

        except Exception as e:
            yield send_event("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@app.post("/validate", response_model=ValidationResponse)
async def validate(request: ValidationRequest) -> ValidationResponse:
    """Validate a HED annotation string.

    Args:
        request: Validation request with HED string

    Returns:
        Validation result with errors and warnings

    Raises:
        HTTPException: If validation fails
    """
    if schema_loader is None:
        raise HTTPException(status_code=503, detail="Schema loader not initialized")

    try:
        # Load schema
        schema = schema_loader.load_schema(request.schema_version)

        # Validate using Python validator
        validator = HedPythonValidator(schema)
        result = validator.validate(request.hed_string)

        return ValidationResponse(
            is_valid=result.is_valid,
            errors=[f"[{e.code}] {e.message}" for e in result.errors],
            warnings=[f"[{w.code}] {w.message}" for w in result.warnings],
            parsed_string=result.parsed_string,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Validation failed: {str(e)}",
        ) from e


@app.get("/")
async def root():
    """Root endpoint with API information.

    Returns:
        API information
    """
    return {
        "name": "HED-BOT API",
        "version": __version__,
        "description": "Multi-agent system for HED annotation generation",
        "endpoints": {
            "POST /annotate": "Generate HED annotation from description",
            "POST /validate": "Validate HED annotation string",
            "GET /health": "Health check",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=38427,
        reload=True,
    )
