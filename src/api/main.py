"""FastAPI application for HED-BOT annotation service.

This module provides REST API endpoints for HED annotation generation
and validation using the multi-agent workflow.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.chat_models import ChatOllama

from src import __version__
from src.agents.workflow import HedAnnotationWorkflow
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

    # Get configuration from environment
    llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434")
    llm_model = os.getenv("LLM_MODEL", "gpt-oss:20b")
    schema_dir = os.getenv(
        "HED_SCHEMA_DIR",
        "/Users/yahya/Documents/git/HED/hed-schemas/schemas_latest_json",
    )
    validator_path = os.getenv(
        "HED_VALIDATOR_PATH",
        "/Users/yahya/Documents/git/HED/hed-javascript",
    )
    use_js_validator = os.getenv("USE_JS_VALIDATOR", "true").lower() == "true"

    # Initialize LLM
    llm = ChatOllama(
        base_url=llm_base_url,
        model=llm_model,
        temperature=0.1,  # Low temperature for consistent annotations
    )

    # Initialize workflow with JSON schema support
    workflow = HedAnnotationWorkflow(
        llm=llm,
        schema_dir=Path(schema_dir),
        validator_path=Path(validator_path) if use_js_validator else None,
        use_js_validator=use_js_validator,
    )

    print(f"Workflow initialized with LLM: {llm_model} at {llm_base_url}")
    print(f"Using JavaScript validator: {use_js_validator}")

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
        # Run annotation workflow
        final_state = await workflow.run(
            input_description=request.description,
            schema_version=request.schema_version,
            max_validation_attempts=request.max_validation_attempts,
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
