"""FastAPI application for HEDit annotation service.

This module provides REST API endpoints for HED annotation generation
and validation using the multi-agent workflow.
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import anthropic
from anthropic import APITimeoutError, RateLimitError
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_anthropic.chat_models import AnthropicContextOverflowError

from src import __version__
from src.agents.vision_agent import VisionAgent
from src.agents.workflow import HedAnnotationWorkflow
from src.api.models import (
    AnnotationRequest,
    AnnotationResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ImageAnnotationRequest,
    ImageAnnotationResponse,
    MetricsResponse,
    UsageSummary,
    ValidationRequest,
    ValidationResponse,
)
from src.api.security import api_key_auth, audit_logger
from src.lsp import HedLspClient
from src.telemetry import LocalFileStorage, TelemetryCollector, TelemetryEvent
from src.utils.anthropic_llm import (
    DEFAULT_MODEL,
    annotation_thinking,
    create_anthropic_llm,
    normalize_model,
)
from src.utils.llm_usage import UsageLedger, process_ledger, usage_scope
from src.utils.schema_loader import HedSchemaLoader
from src.validation.hed_validator import HedPythonValidator

# Load environment variables from .env file
load_dotenv()

# Global workflow and vision agent instances
workflow: HedAnnotationWorkflow | None = None
vision_agent: VisionAgent | None = None
schema_loader: HedSchemaLoader | None = None

# Persistent LSP client; lifetime = process lifetime. Spawned in lifespan
# so every request reuses one warm Node child instead of cold-booting
# hed-suggest per keyword.
lsp_client: HedLspClient | None = None

# Telemetry collector (initialized in lifespan)
telemetry_collector: TelemetryCollector | None = None
_startup_time: str = datetime.now(UTC).isoformat()

# Cache for BYOK configuration
_byok_config: dict = {}

# Set during lifespan: False when the server credentials failed the cheap
# startup validation call (server stays up, /health reports degraded).
_llm_credentials_ok: bool = True


def _describe_llm_error(exc: Exception) -> tuple[int, str, str]:
    """Map an LLM exception to (http_status, error_type, user-facing message).

    Keeps the four annotate endpoints consistent: a BYOK user with a revoked
    key gets a 401 pointing at their key, not a generic 500. Order matters --
    AnthropicContextOverflowError subclasses BadRequestError, and the timeout
    and rate-limit types must be checked before their parent classes.

    Every message here is a fixed string. Exception text never reaches the
    client, because a provider error can carry request details and the web app
    renders the message as HTML. Callers log the exception, so the provider's
    own wording stays available in the server log for whoever debugs it.
    """
    if isinstance(exc, APITimeoutError):
        return 504, "timeout", "LLM request timed out. Try again or use a faster model."
    if isinstance(exc, RateLimitError):
        return 429, "rate_limit", "LLM rate limit exceeded. Please wait and try again."
    if isinstance(exc, anthropic.AuthenticationError):
        return 401, "auth", "Invalid or expired Anthropic API key."
    if isinstance(exc, anthropic.PermissionDeniedError):
        return (
            403,
            "permission",
            "The Anthropic API key is not authorized for this workspace or model.",
        )
    if isinstance(exc, AnthropicContextOverflowError):
        return 413, "context_overflow", "The input is too long for this model. Try shortening it."
    if isinstance(exc, anthropic.BadRequestError):
        return (
            400,
            "bad_request",
            "The LLM rejected the request. Check the model and request parameters, then try again.",
        )
    if isinstance(exc, anthropic.APIConnectionError):
        return 502, "upstream_unreachable", "Could not reach the LLM service. Please try again."
    return 500, "internal", "An error occurred during annotation processing."


# Sentinel used by the workflow factories to mean "fall back to the
# server-wide lsp_client global". Per-request callers can pass an
# explicit None to opt out, or another HedLspClient instance to override.
_USE_GLOBAL_LSP: HedLspClient = object()  # type: ignore[assignment]


def _usage_summary(ledger: UsageLedger) -> UsageSummary | None:
    """Build the response's usage figures from a request's usage ledger.

    Returns None when no LLM call was recorded (a fully cached or failed
    request), so clients can tell "no calls" from "zero cost".

    Args:
        ledger: Ledger collected for one request

    Returns:
        UsageSummary, or None when nothing was recorded
    """
    totals = ledger.total()
    if totals.calls == 0:
        return None
    return UsageSummary(**totals.as_dict())


def _override_header(req: Request, name: str) -> str | None:
    """Read a per-request override header, preferring the X-Anthropic-* spelling.

    The X-OpenRouter-* names are the wire spelling from before the Anthropic
    migration. They remain accepted indefinitely so that cached frontends and
    third-party clients keep working; current clients send X-Anthropic-*.

    Args:
        req: Incoming request
        name: Header suffix, e.g. "model" for X-Anthropic-Model

    Returns:
        Header value, or None when neither spelling is present
    """
    return req.headers.get(f"x-anthropic-{name}") or req.headers.get(f"x-openrouter-{name}")


def _resolve_lsp_client(
    explicit: HedLspClient | None,
) -> HedLspClient | None:
    """Pick the per-request LSP client.

    Explicit values (including ``None``) win. The sentinel
    ``_USE_GLOBAL_LSP`` means "use the lifespan-managed global".
    """
    if explicit is _USE_GLOBAL_LSP:
        return lsp_client
    return explicit


def create_anthropic_workflow(
    api_key: str | None = None,
    annotation_model: str | None = None,
    eval_model: str | None = None,
    temperature: float | None = None,
    schema_dir: str | Path | None = None,
    validator_path: str | Path | None = None,
    use_js_validator: bool = True,
    lsp_client: HedLspClient | None = _USE_GLOBAL_LSP,
) -> HedAnnotationWorkflow:
    """Create a workflow with Anthropic Claude LLMs.

    Unified function for both BYOK and server modes. Applies defaults from
    environment variables, then overrides with provided parameters.

    Args:
        api_key: BYOK Anthropic API key (None = server mode, which uses the
            Claude Platform on AWS credentials from the environment)
        annotation_model: Model for annotation (default: ANNOTATION_MODEL env or Claude Haiku 4.5)
        eval_model: Model for eval/assessment/feedback (default: EVALUATION_MODEL env or Claude Haiku 4.5)
        temperature: LLM temperature (default: 0.1)
        schema_dir: Path to HED schemas (None = fetch from GitHub)
        validator_path: Path to hed-javascript (None = use auto fallback chain)
        use_js_validator: Whether to use JavaScript validator
        lsp_client: Explicit LSP client, None to opt out, or the default
            sentinel to reuse the lifespan-managed global client

    Returns:
        Configured HedAnnotationWorkflow

    Raises:
        ValueError: If a requested model is not offered
        RuntimeError: If server mode is used without ANTHROPIC_API_KEY set
    """
    # Apply defaults from environment
    actual_annotation_model = annotation_model or os.getenv("ANNOTATION_MODEL", DEFAULT_MODEL)
    # The evaluation judge stays on Haiku regardless of the annotation model.
    actual_eval_model = eval_model or os.getenv("EVALUATION_MODEL", DEFAULT_MODEL)
    actual_temperature = temperature if temperature is not None else 0.1

    # Validate both models up front so a bad request is rejected with a 400
    # even when server credentials are missing (which would otherwise
    # surface first as a 503 from the annotation LLM's credential check).
    normalize_model(actual_annotation_model)
    normalize_model(actual_eval_model)

    # Create LLMs.
    # Annotation thinks: measured over the benchmark descriptions, a 2048-token
    # budget took first-attempt validity from 5/15 to 13/15 and cut total LLM
    # calls by a third, for 24% more cost and about twice the latency. See
    # annotation_thinking() and docs/prompt-caching.md.
    annotation_llm = create_anthropic_llm(
        model=actual_annotation_model,
        api_key=api_key,
        temperature=actual_temperature,
        thinking=annotation_thinking(actual_annotation_model),
        role="annotation",
    )
    # Evaluation / assessment / feedback / keyword extraction are short
    # structured tasks; reasoning adds 5-10 s per call without
    # measurable quality benefit. See #150.
    evaluation_llm = create_anthropic_llm(
        model=actual_eval_model,
        api_key=api_key,
        temperature=actual_temperature,
        disable_reasoning=True,
        role="evaluation",
    )
    assessment_llm = create_anthropic_llm(
        model=actual_eval_model,
        api_key=api_key,
        temperature=actual_temperature,
        disable_reasoning=True,
        role="assessment",
    )
    feedback_llm = create_anthropic_llm(
        model=actual_eval_model,
        api_key=api_key,
        temperature=actual_temperature,
        disable_reasoning=True,
        role="feedback",
    )
    # Keyword extraction (#148): use the fast annotation model with
    # reasoning explicitly disabled and a small token cap. The
    # task is "list 5-10 keywords"; the heavier eval model used here
    # previously cost ~10 s per call.
    keyword_llm = create_anthropic_llm(
        model=actual_annotation_model,
        api_key=api_key,
        temperature=actual_temperature,
        max_tokens=200,
        disable_reasoning=True,
        role="keyword",
    )

    # Create and return workflow
    # Only use JS validator if validator_path is available.
    # The LSP client (when provided) backs the persistent hed-lsp connection
    # used for both semantic preprocessing and tag-replacement suggestions.
    actual_use_js = use_js_validator and validator_path is not None
    return HedAnnotationWorkflow(
        llm=annotation_llm,
        evaluation_llm=evaluation_llm,
        assessment_llm=assessment_llm,
        feedback_llm=feedback_llm,
        keyword_llm=keyword_llm,
        schema_dir=Path(schema_dir) if schema_dir else None,
        validator_path=Path(validator_path) if validator_path else None,
        use_js_validator=actual_use_js,
        lsp_client=_resolve_lsp_client(lsp_client),
    )


def create_byok_workflow(
    byok_key: str,
    model: str | None = None,
    eval_model: str | None = None,
    temperature: float | None = None,
    lsp_client: HedLspClient | None = _USE_GLOBAL_LSP,
) -> HedAnnotationWorkflow:
    """Create a workflow for BYOK mode using the user's Anthropic key.

    Thin wrapper around create_anthropic_workflow that uses cached server
    config for schema/validator paths. BYOK keys go to the first-party
    Anthropic API, not the server's AWS workspace.

    Args:
        byok_key: User's Anthropic API key
        model: Override annotation model
        eval_model: Override evaluation model (for all eval/assessment/feedback)
        temperature: Override LLM temperature
        lsp_client: Explicit LSP client, None to opt out, or the default
            sentinel to reuse the lifespan-managed global client

    Returns:
        Configured HedAnnotationWorkflow using the user's key
    """
    global _byok_config

    return create_anthropic_workflow(
        api_key=byok_key,
        annotation_model=model,
        eval_model=eval_model,
        temperature=temperature if temperature is not None else _byok_config.get("temperature"),
        schema_dir=_byok_config.get("schema_dir"),
        validator_path=_byok_config.get("validator_path"),
        use_js_validator=_byok_config.get("use_js_validator", True),
        lsp_client=lsp_client,
    )


def create_vision_agent(
    api_key: str | None = None,
    vision_model: str | None = None,
    temperature: float | None = None,
) -> VisionAgent:
    """Create a vision agent instance.

    Args:
        api_key: BYOK Anthropic API key (None = server mode)
        vision_model: Override vision model (uses server default if None)
        temperature: Override temperature (uses 0.3 default if None)

    Returns:
        Configured VisionAgent

    Raises:
        ValueError: If a requested model is not offered
        RuntimeError: If server mode is used without ANTHROPIC_API_KEY set
    """
    actual_model = vision_model or os.getenv("VISION_MODEL", DEFAULT_MODEL)
    actual_temperature = temperature if temperature is not None else 0.3

    vision_llm = create_anthropic_llm(
        model=actual_model,
        api_key=api_key,
        temperature=actual_temperature,
        role="vision",
    )

    return VisionAgent(llm=vision_llm)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan (startup and shutdown).

    Args:
        app: FastAPI application
    """
    global workflow, vision_agent, schema_loader

    # Startup: Initialize workflow
    print("Initializing HEDit annotation workflow...")

    # Auto-detect environment (Docker vs local)
    def get_default_path(docker_path: str, local_path: str) -> str | None:
        """Get default path based on environment.

        Args:
            docker_path: Path to use in Docker
            local_path: Path to use in local development

        Returns:
            Appropriate default path, or None if no paths exist
            (HED library will fetch from GitHub when None)
        """
        # Check if running in Docker (look for Docker-specific paths)
        if Path("/app").exists() and Path(docker_path).exists():
            return docker_path
        # Check if local development path exists
        elif Path(local_path).exists():
            return local_path
        # Return None to trigger HED library to fetch from GitHub
        return None

    # Get configuration from environment with smart defaults.
    # "anthropic" is the only supported provider since the 2026-08-18
    # migration; any other value (including legacy "openrouter"/"ollama")
    # is overridden to "anthropic" with a warning.
    llm_provider = os.getenv("LLM_PROVIDER", "anthropic")
    if llm_provider != "anthropic":
        logging.getLogger("hedit.config").warning(
            "LLM_PROVIDER '%s' is no longer supported; using 'anthropic'", llm_provider
        )
        llm_provider = "anthropic"
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

    # Cache BYOK configuration for on-demand workflow creation
    global _byok_config
    _byok_config = {
        "temperature": llm_temperature,
        "schema_dir": schema_dir,
        "validator_path": validator_path,
        "use_js_validator": use_js_validator,
    }

    print(f"Environment: {'Docker' if Path('/app').exists() else 'Local'}")
    print(f"Schema directory: {schema_dir or 'GitHub (dynamic fetch)'}")
    print(f"Validator path: {validator_path or 'None (using Python validator)'}")

    # Spawn the persistent hed-lsp child once for the lifetime of the
    # process. Per-request workflows reuse this single warm connection
    # instead of cold-booting hed-suggest per keyword. Set HED_LSP_DISABLE=1
    # to fall back to keyword-only preprocessing (no LSP enrichment).
    # Failures here are logged via the structured logger (not print) so
    # operators can detect a silently-degraded server from log
    # aggregation.
    global lsp_client
    lsp_logger = logging.getLogger("hedit.lsp")
    if os.getenv("HED_LSP_DISABLE", "").lower() in ("1", "true", "yes"):
        lsp_logger.info("LSP client disabled via HED_LSP_DISABLE")
        lsp_client = None
    else:
        server_js_env = os.getenv("HED_LSP_SERVER_JS")
        default_server_js = Path("/app/hed-lsp/server/out/server.js")
        server_js_path = Path(server_js_env) if server_js_env else default_server_js
        if not server_js_path.exists():
            lsp_logger.error(
                "LSP server.js not found at %s; set HED_LSP_SERVER_JS or install "
                "hed-lsp. Continuing without LSP enrichment.",
                server_js_path,
            )
            lsp_client = None
        else:
            try:
                lsp_client = await HedLspClient.spawn_stdio(
                    server_js_path,
                    schema_version=os.getenv("HED_SCHEMA_VERSION", "8.4.0"),
                )
                lsp_logger.info("LSP client connected (server.js=%s)", server_js_path)
            except (RuntimeError, OSError, TimeoutError) as exc:
                lsp_logger.error(
                    "LSP client spawn failed; continuing without LSP enrichment. "
                    "Set HED_LSP_DISABLE=1 to suppress this error.",
                    exc_info=exc,
                )
                lsp_client = None

    # Initialize workflow (Claude Platform on AWS - unified workflow creation)
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    # Log configuration (env vars are read by create_anthropic_workflow)
    print("Using Anthropic Claude models (Claude Platform on AWS):")
    print(f"  Annotation: {os.getenv('ANNOTATION_MODEL', DEFAULT_MODEL)}")
    print(f"  Evaluation: {os.getenv('EVALUATION_MODEL', DEFAULT_MODEL)}")

    workflow = create_anthropic_workflow(
        temperature=llm_temperature,
        schema_dir=schema_dir,
        validator_path=validator_path if use_js_validator else None,
        use_js_validator=use_js_validator,
        lsp_client=lsp_client,
    )

    # Validate the credentials with a cheap live call (count_tokens is free).
    # LLM construction never touches the network, so a present-but-wrong key
    # would otherwise boot a server that reports healthy and 500s on every
    # request. Failure keeps the server up but marks /health degraded.
    global _llm_credentials_ok
    try:
        _validation_client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
            default_headers=(
                {"anthropic-workspace-id": workspace_id}
                if (workspace_id := os.getenv("ANTHROPIC_WORKSPACE_ID"))
                else None
            ),
            timeout=10.0,
            max_retries=1,
        )
        _validation_client.messages.count_tokens(
            model=os.getenv("ANNOTATION_MODEL", DEFAULT_MODEL),
            messages=[{"role": "user", "content": "ping"}],
        )
        _llm_credentials_ok = True
        print("Anthropic credentials validated")
    except Exception:
        _llm_credentials_ok = False
        logging.getLogger("hedit.config").error(
            "Anthropic credential validation failed; the server will start "
            "but LLM requests will fail. Check ANTHROPIC_API_KEY, "
            "ANTHROPIC_BASE_URL, and ANTHROPIC_WORKSPACE_ID.",
            exc_info=True,
        )

    # Set global schema_loader from workflow
    schema_loader = workflow.schema_loader

    print("Workflow initialized successfully!")
    print(f"  LLM Provider: {llm_provider} (temperature={llm_temperature})")
    print(f"  JavaScript validator: {use_js_validator}")

    # Initialize vision agent (Claude models are natively multimodal).
    # A bad VISION_MODEL must not take down text annotation: on failure the
    # image endpoints return 503 while the rest of the server stays up.
    vision_model = os.getenv("VISION_MODEL", DEFAULT_MODEL)
    print(f"Initializing vision model: {vision_model}")
    try:
        vision_agent = create_vision_agent()
        print("Vision agent initialized successfully!")
    except Exception:
        vision_agent = None
        logging.getLogger("hedit.config").error(
            "Vision agent initialization failed (VISION_MODEL=%s); image "
            "annotation endpoints will return 503.",
            vision_model,
            exc_info=True,
        )

    global _startup_time
    _startup_time = datetime.now(UTC).isoformat()

    # Initialize telemetry collector
    global telemetry_collector
    # Use /app/telemetry in Docker, otherwise use local .hedit/telemetry
    default_telemetry_dir = "/app/telemetry" if Path("/app").exists() else ".hedit/telemetry"
    telemetry_dir = os.getenv("TELEMETRY_DIR", default_telemetry_dir)
    telemetry_storage = LocalFileStorage(storage_dir=telemetry_dir)
    telemetry_collector = TelemetryCollector(
        storage=telemetry_storage,
        enabled=True,  # Can be configured via env var if needed
    )
    print(f"Telemetry collector initialized (storage: {telemetry_dir})")

    yield

    # Shutdown
    print("Shutting down HEDit...")
    if lsp_client is not None:
        try:
            await lsp_client.shutdown()
            lsp_logger.info("LSP client shut down cleanly")
        except Exception as exc:
            lsp_logger.error("LSP client shutdown error", exc_info=exc)


# Create FastAPI app
app = FastAPI(
    title="HEDit API",
    description="Multi-agent system for HED annotation generation and validation",
    version=__version__,
    lifespan=lifespan,
)

# Configure CORS
# Production: Strict origin validation
# Development: Allow all localhost ports for easy local testing
allowed_origins = [
    "https://hedit.pages.dev",  # Production frontend
    "https://develop.hedit.pages.dev",  # Development frontend
    "https://hedit-api.shirazi-10f.workers.dev",  # Production Worker proxy
    "https://hedit-dev-api.shirazi-10f.workers.dev",  # Development Worker proxy
    "https://annotation.garden",  # Main AGI website
]

# Add common localhost ports for development
# These allow testing with any local dev server
localhost_origins = [
    "http://localhost:3000",  # React default
    "http://localhost:5173",  # Vite default
    "http://localhost:8080",  # Common dev server
    "http://localhost:8000",  # Alternative
    "http://127.0.0.1:3000",  # IPv4 localhost
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:8000",
]

# Add localhost origins (can be disabled via env var for strict production)
if os.getenv("ALLOW_LOCALHOST_CORS", "true").lower() == "true":
    allowed_origins.extend(localhost_origins)

# Add environment-specific origins if configured
if extra_origins := os.getenv("EXTRA_CORS_ORIGINS"):
    allowed_origins.extend(
        [origin.strip() for origin in extra_origins.split(",") if origin.strip()]
    )

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,  # type: ignore[arg-type]  # Starlette typing limitation
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "X-API-Key",
        "X-Anthropic-Key",  # BYOK mode (Anthropic API key)
        "X-Anthropic-Model",  # Model override
        "X-Anthropic-Eval-Model",  # Eval model override
        "X-Anthropic-Vision-Model",  # Vision model override
        "X-Anthropic-Temperature",  # Temperature override
        # Legacy X-OpenRouter-* spellings, still accepted as transport
        "X-OpenRouter-Key",
        "X-OpenRouter-Model",
        "X-OpenRouter-Vision-Model",
        "X-OpenRouter-Vision-Provider",  # Legacy, ignored
        "X-OpenRouter-Provider",  # Legacy, ignored
        "X-OpenRouter-Temperature",
        "X-OpenRouter-Eval-Model",
        "X-OpenRouter-Eval-Provider",  # Legacy, ignored
        "X-User-Id",  # Legacy, ignored
    ],
    max_age=3600,  # Cache preflight requests for 1 hour
)


# Audit logging middleware
@app.middleware("http")
async def audit_logging_middleware(request: Request, call_next):
    """Middleware to log all requests and responses for audit trail."""
    start_time = time.time()

    # Log incoming request
    api_key = request.headers.get("x-api-key")
    api_key_hash = api_key[:8] + "..." if api_key else None
    audit_logger.log_request(request, api_key_hash=api_key_hash)

    # Process request
    try:
        response = await call_next(request)
        processing_time_ms = (time.time() - start_time) * 1000

        # Log response
        audit_logger.log_response(request, response.status_code, processing_time_ms)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response
    except Exception as e:
        # Log error
        audit_logger.log_error(request, e, api_key_hash=api_key_hash)
        raise


# A schema loader existing does not prove the JavaScript validator can run
# (Node missing, hed-javascript not built), so /health runs a real validation
# through the workflow's backend. The probe spawns a Node subprocess on the
# JS path, so its result is cached and refreshed at most once per TTL.
_VALIDATOR_PROBE_TTL_SECONDS = 300.0
_validator_probe: dict[str, float | bool] = {"functional": False, "checked_at": float("-inf")}


async def _validator_functional() -> bool:
    """Return whether the active validator backend passes a functional probe."""
    now = time.monotonic()
    if now - _validator_probe["checked_at"] < _VALIDATOR_PROBE_TTL_SECONDS:
        return bool(_validator_probe["functional"])

    functional = False
    if workflow is not None:
        schema_version = os.getenv("HED_SCHEMA_VERSION", "8.4.0")
        functional = await asyncio.to_thread(workflow.validation_agent.probe, schema_version)
    _validator_probe["functional"] = functional
    _validator_probe["checked_at"] = now
    return functional


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns:
        Health status and service availability
    """
    llm_available = workflow is not None and _llm_credentials_ok
    validator_available = schema_loader is not None and await _validator_functional()

    status = "healthy" if (llm_available and validator_available) else "degraded"

    return HealthResponse(
        status=status,
        version=__version__,
        llm_available=llm_available,
        validator_available=validator_available,
    )


@app.post("/annotate", response_model=AnnotationResponse)
async def annotate(
    request: AnnotationRequest,
    req: Request,
    api_key: str = Depends(api_key_auth),
) -> AnnotationResponse:
    """Generate HED annotation from natural language description.

    Supports two authentication modes:
    - X-API-Key header: Server-level authentication
    - X-Anthropic-Key header: BYOK mode (uses your Anthropic key for billing)

    Args:
        request: Annotation request with description and parameters
        req: FastAPI request to extract headers
        api_key: Authentication result (injected by dependency)

    Returns:
        Generated annotation with validation and assessment feedback

    Raises:
        HTTPException: If workflow fails or authentication fails
    """
    # Determine which workflow to use
    # Check for model override headers (from frontend dropdown or CLI)
    model_override = request.model or _override_header(req, "model")
    eval_model_override = _override_header(req, "eval-model")
    temp_header = _override_header(req, "temperature")
    temperature = request.temperature
    if temperature is None and temp_header:
        try:
            temperature = float(temp_header)
        except ValueError:
            pass  # Invalid header value, use default

    if api_key == "byok":
        # BYOK mode: Create workflow with user's Anthropic key
        byok_key = _override_header(req, "key")
        if not byok_key:
            raise HTTPException(status_code=401, detail="Missing X-Anthropic-Key header")

        try:
            active_workflow = create_byok_workflow(
                byok_key,
                model=model_override,
                eval_model=eval_model_override,
                temperature=temperature,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to initialize BYOK workflow: {str(e)}"
            ) from e
    elif model_override or eval_model_override:
        # Server mode with model overrides: Create custom workflow with server
        # credentials. Supports the frontend model dropdown without requiring
        # the user's own API key.
        try:
            active_workflow = create_anthropic_workflow(
                annotation_model=model_override,
                eval_model=eval_model_override,
                temperature=temperature,
                schema_dir=_byok_config.get("schema_dir"),
                validator_path=_byok_config.get("validator_path"),
                use_js_validator=_byok_config.get("use_js_validator", True),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize workflow with model override: {str(e)}",
            ) from e
    else:
        # Server mode: Use pre-initialized workflow
        if workflow is None:
            raise HTTPException(status_code=503, detail="Workflow not initialized")
        active_workflow = workflow

    try:
        config = {"recursion_limit": 50}

        start_time = time.time()
        with usage_scope() as usage:
            final_state = await active_workflow.run(
                input_description=request.description,
                schema_version=request.schema_version,
                max_validation_attempts=request.max_validation_attempts,
                run_assessment=request.run_assessment,
                no_extend=request.no_extend,
                config=config,
            )
        latency_ms = int((time.time() - start_time) * 1000)

        # Determine overall status
        # IMPORTANT: Ensure is_valid is only True when there are NO validation errors
        # This is a safeguard to prevent inconsistencies in the workflow
        is_valid = final_state["is_valid"] and len(final_state["validation_errors"]) == 0
        status = "success" if is_valid else "failed"

        # Collect telemetry if enabled
        if request.telemetry_enabled and telemetry_collector:
            # Get model info from request body, BYOK headers, or server config
            model_name = (
                request.model
                or _override_header(req, "model")
                or os.getenv("ANNOTATION_MODEL", DEFAULT_MODEL)
            )
            temperature = request.temperature
            if temperature is None:
                temp_header = _override_header(req, "temperature")
                if temp_header is not None:
                    try:
                        temperature = float(temp_header)
                    except ValueError:
                        temperature = None
            if temperature is None:
                temperature = _byok_config.get("temperature", 0.1)

            event = TelemetryEvent.create(
                description=request.description,
                schema_version=request.schema_version,
                hed_string=final_state["current_annotation"],
                iterations=final_state["validation_attempts"],
                validation_errors=final_state["validation_errors"],
                model=model_name,
                provider="anthropic",
                temperature=temperature,
                latency_ms=latency_ms,
                source="api",
                usage=usage.total(),
            )
            await telemetry_collector.collect(event)

        return AnnotationResponse(
            annotation=final_state["current_annotation"],
            is_valid=is_valid,
            is_faithful=final_state["is_faithful"],
            is_complete=final_state["is_complete"],
            validation_attempts=final_state["validation_attempts"],
            validation_errors=final_state["validation_errors"],
            validation_warnings=final_state["validation_warnings"],
            evaluation_feedback=final_state["evaluation_feedback"],
            assessment_feedback=final_state["assessment_feedback"],
            status=status,
            usage=_usage_summary(usage),
        )

    except Exception as e:
        status_code, _error_type, message = _describe_llm_error(e)
        logging.exception("Annotation workflow failed")
        raise HTTPException(status_code=status_code, detail=message) from e


@app.post("/annotate-from-image", response_model=ImageAnnotationResponse)
async def annotate_from_image(
    request: ImageAnnotationRequest,
    req: Request,
    api_key: str = Depends(api_key_auth),
) -> ImageAnnotationResponse:
    """Generate HED annotation from an image.

    Supports two authentication modes:
    - X-API-Key header: Server-level authentication
    - X-Anthropic-Key header: BYOK mode (uses your Anthropic key for billing)

    This endpoint uses a vision-language model to generate a description of the image,
    then passes that description through the standard HED annotation workflow.

    Args:
        request: Image annotation request with base64 image and parameters
        req: FastAPI request to extract headers
        api_key: Authentication result (injected by dependency)

    Returns:
        Generated annotation with image description and validation feedback

    Raises:
        HTTPException: If workflow or vision agent fails or authentication fails
    """
    # Determine which workflow and vision agent to use
    # Check for model override headers (from frontend dropdown or CLI)
    model_override = request.model or _override_header(req, "model")
    vision_model_override = request.vision_model or _override_header(req, "vision-model")
    eval_model_override = _override_header(req, "eval-model")
    temp_header = _override_header(req, "temperature")
    temperature = request.temperature
    if temperature is None and temp_header:
        try:
            temperature = float(temp_header)
        except ValueError:
            pass  # Invalid header value, use default

    if api_key == "byok":
        # BYOK mode: Create workflow and vision agent with user's Anthropic key
        byok_key = _override_header(req, "key")
        if not byok_key:
            raise HTTPException(status_code=401, detail="Missing X-Anthropic-Key header")

        try:
            active_workflow = create_byok_workflow(
                byok_key,
                model=model_override,
                eval_model=eval_model_override,
                temperature=temperature,
            )
            active_vision_agent = create_vision_agent(
                api_key=byok_key,
                vision_model=vision_model_override,
                temperature=temperature,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logging.exception("Failed to initialize BYOK agents")
            raise HTTPException(
                status_code=500, detail=f"Failed to initialize BYOK agents: {str(e)}"
            ) from e
    elif model_override or eval_model_override or vision_model_override:
        # Server mode with model overrides: Create custom workflow with server credentials
        try:
            active_workflow = create_anthropic_workflow(
                annotation_model=model_override,
                eval_model=eval_model_override,
                temperature=temperature,
                schema_dir=_byok_config.get("schema_dir"),
                validator_path=_byok_config.get("validator_path"),
                use_js_validator=_byok_config.get("use_js_validator", True),
            )
            active_vision_agent = create_vision_agent(
                vision_model=vision_model_override,
                temperature=temperature,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize workflow with model override: {str(e)}",
            ) from e
    else:
        # Server mode: Use pre-initialized workflow and vision agent
        if workflow is None:
            raise HTTPException(status_code=503, detail="Workflow not initialized")
        if vision_agent is None:
            raise HTTPException(
                status_code=503,
                detail="Vision model not available. Please use the Anthropic provider.",
            )
        active_workflow = workflow
        active_vision_agent = vision_agent

    try:
        start_time = time.time()

        # The vision call and the annotation workflow share one usage scope
        # so the reported figures cover the whole request.
        with usage_scope() as usage:
            # Step 1: Generate image description using vision model
            vision_result = await active_vision_agent.describe_image(
                image_data=request.image,
                custom_prompt=request.prompt,
            )

            image_description = vision_result["description"]
            image_metadata = vision_result["metadata"]

            # Step 2: Pass description through HED annotation workflow
            config = {"recursion_limit": 50}

            final_state = await active_workflow.run(
                input_description=image_description,
                schema_version=request.schema_version,
                max_validation_attempts=request.max_validation_attempts,
                run_assessment=request.run_assessment,
                no_extend=request.no_extend,
                config=config,
            )
        latency_ms = int((time.time() - start_time) * 1000)

        # Determine overall status
        is_valid = final_state["is_valid"] and len(final_state["validation_errors"]) == 0
        status = "success" if is_valid else "failed"

        # Collect telemetry if enabled
        if request.telemetry_enabled and telemetry_collector:
            # Get model info from request body, BYOK headers, or server config
            model_name = (
                request.model
                or _override_header(req, "model")
                or os.getenv("ANNOTATION_MODEL", DEFAULT_MODEL)
            )
            temperature = request.temperature
            if temperature is None:
                temp_header = _override_header(req, "temperature")
                if temp_header is not None:
                    try:
                        temperature = float(temp_header)
                    except ValueError:
                        temperature = None
            if temperature is None:
                temperature = _byok_config.get("temperature", 0.1)

            event = TelemetryEvent.create(
                description=image_description,  # Use generated image description
                schema_version=request.schema_version,
                hed_string=final_state["current_annotation"],
                iterations=final_state["validation_attempts"],
                validation_errors=final_state["validation_errors"],
                model=model_name,
                provider="anthropic",
                temperature=temperature,
                latency_ms=latency_ms,
                source="api-image",  # Distinguish from text-based annotation
                usage=usage.total(),
            )
            await telemetry_collector.collect(event)

        return ImageAnnotationResponse(
            image_description=image_description,
            annotation=final_state["current_annotation"],
            is_valid=is_valid,
            is_faithful=final_state["is_faithful"],
            is_complete=final_state["is_complete"],
            validation_attempts=final_state["validation_attempts"],
            validation_errors=final_state["validation_errors"],
            validation_warnings=final_state["validation_warnings"],
            evaluation_feedback=final_state["evaluation_feedback"],
            assessment_feedback=final_state["assessment_feedback"],
            status=status,
            image_metadata=image_metadata,
            usage=_usage_summary(usage),
        )

    except Exception as e:
        status_code, _error_type, message = _describe_llm_error(e)
        logging.exception("Image annotation workflow failed")
        raise HTTPException(status_code=status_code, detail=message) from e


async def _collect_stream_telemetry(
    request: AnnotationRequest | ImageAnnotationRequest,
    req: Request,
    current_state: dict,
    start_time: float,
    source: str,
    description: str,
    usage: UsageLedger | None = None,
) -> None:
    """Collect telemetry for streaming endpoints.

    Shared helper used by both /annotate/stream and /annotate-from-image/stream.
    Silently returns if telemetry is disabled or collector is not initialized.

    Args:
        request: The annotation request (text or image)
        req: FastAPI request for header extraction
        current_state: Current workflow state dict
        start_time: Workflow start time (from time.time())
        source: Telemetry source identifier (e.g., "api-stream", "api-image-stream")
        description: Input description text (or image description for image endpoints)
        usage: Usage ledger for this request, when one was collected
    """
    if not request.telemetry_enabled or not telemetry_collector:
        return

    latency_ms = int((time.time() - start_time) * 1000)

    # Get model info from request body, BYOK headers, or server config
    model_name = (
        request.model
        or _override_header(req, "model")
        or os.getenv("ANNOTATION_MODEL", DEFAULT_MODEL)
    )
    temperature = request.temperature
    if temperature is None:
        temp_header = _override_header(req, "temperature")
        if temp_header is not None:
            try:
                temperature = float(temp_header)
            except ValueError:
                temperature = None
    if temperature is None:
        temperature = _byok_config.get("temperature", 0.1)

    event = TelemetryEvent.create(
        description=description,
        schema_version=request.schema_version,
        hed_string=current_state.get("current_annotation", ""),
        iterations=current_state.get("validation_attempts", 0),
        validation_errors=current_state.get("validation_errors", []),
        model=model_name,
        provider="anthropic",
        temperature=temperature,
        latency_ms=latency_ms,
        source=source,
        usage=usage.total() if usage is not None else None,
    )
    await telemetry_collector.collect(event)


@app.post("/annotate/stream")
async def annotate_stream(
    request: AnnotationRequest,
    req: Request,
    api_key: str = Depends(api_key_auth),
):
    """Generate HED annotation with real-time progress updates via Server-Sent Events.

    This endpoint streams progress updates as the workflow runs through different
    stages (annotation, validation, evaluation, assessment), providing real-time
    feedback to the user.

    Supports both server-mode and BYOK (Bring Your Own Key) authentication.

    Args:
        request: Annotation request with description and parameters
        req: FastAPI request to extract headers
        api_key: Authentication result (injected by dependency)

    Returns:
        StreamingResponse with Server-Sent Events

    Raises:
        HTTPException: If workflow fails or authentication fails
    """
    from src.agents.state import create_initial_state

    # Determine which workflow to use (same logic as /annotate)
    model_override = request.model or _override_header(req, "model")
    eval_model_override = _override_header(req, "eval-model")
    temp_header = _override_header(req, "temperature")
    temperature = request.temperature
    if temperature is None and temp_header:
        try:
            temperature = float(temp_header)
        except ValueError:
            pass  # Invalid header value, use default temperature

    if api_key == "byok":
        byok_key = _override_header(req, "key")
        if not byok_key:
            raise HTTPException(status_code=401, detail="Missing X-Anthropic-Key header")
        try:
            active_workflow = create_byok_workflow(
                byok_key,
                model=model_override,
                eval_model=eval_model_override,
                temperature=temperature,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to initialize BYOK workflow: {str(e)}"
            ) from e
    elif model_override or eval_model_override:
        try:
            active_workflow = create_anthropic_workflow(
                annotation_model=model_override,
                eval_model=eval_model_override,
                temperature=temperature,
                schema_dir=_byok_config.get("schema_dir"),
                validator_path=_byok_config.get("validator_path"),
                use_js_validator=_byok_config.get("use_js_validator", True),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to initialize workflow: {str(e)}"
            ) from e
    else:
        if workflow is None:
            raise HTTPException(status_code=503, detail="Workflow not initialized")
        active_workflow = workflow

    # Create initial state (max_total_iterations derived from max_validation_attempts + 1)
    initial_state = create_initial_state(
        request.description,
        request.schema_version,
        request.max_validation_attempts,
        run_assessment=request.run_assessment,
        no_extend=request.no_extend,
    )

    # Node name to user-friendly stage mapping
    node_stage_map = {
        "annotate": ("annotating", "Generating HED annotation..."),
        "validate": ("validating", "Validating HED annotation..."),
        "summarize_feedback": ("refining", "Processing validation feedback..."),
        "evaluate": ("evaluating", "Evaluating annotation faithfulness..."),
        "assess": ("assessing", "Running final assessment..."),
    }

    async def event_generator(usage: UsageLedger):
        """Generate SSE events for workflow progress using LangGraph streaming."""

        def send_event(event_type: str, data: dict) -> str:
            return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

        # SSE padding comment to force Safari to open the stream
        yield ": stream opened\n\n"

        start_time = time.time()
        current_state = initial_state.copy()

        try:
            # Send initial start event
            yield send_event(
                "progress", {"stage": "starting", "message": "Initializing annotation workflow..."}
            )

            # Track state and progress
            last_stage = None
            validation_attempt = 0

            # Use LangGraph's astream_events for real-time streaming
            config = {"recursion_limit": 50}
            async for event in active_workflow.graph.astream_events(  # type: ignore[union-attr]
                initial_state, config=config, version="v2"
            ):
                event_type = event.get("event")
                name = event.get("name", "")

                # Handle node start events
                if event_type == "on_chain_start" and name in node_stage_map:
                    stage, message = node_stage_map[name]

                    # Track validation attempts
                    if name == "validate":
                        validation_attempt += 1

                    # Only send if stage changed
                    if stage != last_stage:
                        last_stage = stage
                        progress_data = {
                            "stage": stage,
                            "message": message,
                        }
                        if name == "validate":
                            progress_data["attempt"] = validation_attempt
                        yield send_event("progress", progress_data)

                # Handle node end events to get intermediate state
                if event_type == "on_chain_end" and name in node_stage_map:
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        current_state.update(output)

                        # Send validation result events
                        if name == "validate":
                            is_valid = output.get("is_valid", False)
                            errors = output.get("validation_errors", [])
                            if is_valid:
                                yield send_event(
                                    "validation",
                                    {
                                        "valid": True,
                                        "attempt": validation_attempt,
                                        "message": "Validation passed",
                                    },
                                )
                            elif errors:
                                tag_suggestions = output.get("tag_suggestions", {})
                                validation_data = {
                                    "valid": False,
                                    "attempt": validation_attempt,
                                    "errors": errors[:3],  # Send first 3 errors
                                    "message": f"Found {len(errors)} validation error(s)",
                                }
                                if tag_suggestions:
                                    validation_data["tag_suggestions"] = tag_suggestions
                                yield send_event("validation", validation_data)

            # Send final result
            is_valid = (
                current_state.get("is_valid", False)
                and len(current_state.get("validation_errors", [])) == 0
            )
            status = "success" if is_valid else "failed"
            result = {
                "annotation": current_state.get("current_annotation", ""),
                "is_valid": is_valid,
                "is_faithful": current_state.get("is_faithful", False),
                "is_complete": current_state.get("is_complete", False),
                "validation_attempts": current_state.get("validation_attempts", 0),
                "validation_errors": current_state.get("validation_errors", []),
                "validation_warnings": current_state.get("validation_warnings", []),
                "tag_suggestions": current_state.get("tag_suggestions", {}),
                "evaluation_feedback": current_state.get("evaluation_feedback", ""),
                "assessment_feedback": current_state.get("assessment_feedback", ""),
                "status": status,
            }
            usage_summary = _usage_summary(usage)
            if usage_summary is not None:
                result["usage"] = usage_summary.model_dump()

            yield send_event("result", result)

            # Collect telemetry after sending result but before done event
            try:
                await _collect_stream_telemetry(
                    request=request,
                    req=req,
                    current_state=current_state,
                    start_time=start_time,
                    source="api-stream",
                    description=request.description,
                    usage=usage,
                )
            except Exception:
                logging.warning("Telemetry collection failed for streaming request", exc_info=True)

            yield send_event("done", {"message": "Workflow completed"})

        except asyncio.CancelledError:
            raise
        except Exception as e:
            _status, error_type, message = _describe_llm_error(e)
            logging.exception("Streaming workflow error (%s)", error_type)
            yield send_event("error", {"message": message, "error_type": error_type})
            # Collect telemetry on error
            try:
                await _collect_stream_telemetry(
                    request=request,
                    req=req,
                    current_state=current_state,
                    start_time=start_time,
                    source="api-stream",
                    description=request.description,
                    usage=usage,
                )
            except Exception:
                logging.warning("Telemetry collection failed on error", exc_info=True)
            yield send_event("done", {"message": "Workflow ended with error"})

    async def streamed_events():
        """Hold one usage scope open for the whole stream.

        The scope lives outside the generator that runs the workflow so that
        every LLM call made while the response streams is attributed to this
        request.
        """
        with usage_scope() as usage:
            async for chunk in event_generator(usage):
                yield chunk

    return StreamingResponse(
        streamed_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "X-Content-Type-Options": "nosniff",  # Prevent MIME-type sniffing; helps Safari trust text/event-stream
        },
    )


@app.post("/annotate-from-image/stream")
async def annotate_from_image_stream(
    request: ImageAnnotationRequest,
    req: Request,
    api_key: str = Depends(api_key_auth),
):
    """Generate HED annotation from an image with real-time progress updates via Server-Sent Events.

    This endpoint streams progress updates as the workflow runs through different
    stages (vision, annotation, validation, evaluation, assessment), providing real-time
    feedback to the user.

    Supports both server-mode and BYOK (Bring Your Own Key) authentication.

    Args:
        request: Image annotation request with base64 image and parameters
        req: FastAPI request to extract headers
        api_key: Authentication result (injected by dependency)

    Returns:
        StreamingResponse with Server-Sent Events

    Raises:
        HTTPException: If workflow or vision agent fails or authentication fails
    """
    from src.agents.state import create_initial_state

    # Determine which workflow and vision agent to use (same logic as /annotate-from-image)
    model_override = request.model or _override_header(req, "model")
    vision_model_override = request.vision_model or _override_header(req, "vision-model")
    eval_model_override = _override_header(req, "eval-model")
    temp_header = _override_header(req, "temperature")
    temperature = request.temperature
    if temperature is None and temp_header:
        try:
            temperature = float(temp_header)
        except ValueError:
            pass

    if api_key == "byok":
        byok_key = _override_header(req, "key")
        if not byok_key:
            raise HTTPException(status_code=401, detail="Missing X-Anthropic-Key header")
        try:
            active_workflow = create_byok_workflow(
                byok_key,
                model=model_override,
                eval_model=eval_model_override,
                temperature=temperature,
            )
            active_vision_agent = create_vision_agent(
                api_key=byok_key,
                vision_model=vision_model_override,
                temperature=temperature,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logging.exception("Failed to initialize BYOK agents")
            raise HTTPException(
                status_code=500, detail=f"Failed to initialize BYOK agents: {str(e)}"
            ) from e
    elif model_override or eval_model_override or vision_model_override:
        try:
            active_workflow = create_anthropic_workflow(
                annotation_model=model_override,
                eval_model=eval_model_override,
                temperature=temperature,
                schema_dir=_byok_config.get("schema_dir"),
                validator_path=_byok_config.get("validator_path"),
                use_js_validator=_byok_config.get("use_js_validator", True),
            )
            active_vision_agent = create_vision_agent(
                vision_model=vision_model_override,
                temperature=temperature,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to initialize workflow: {str(e)}"
            ) from e
    else:
        if workflow is None:
            raise HTTPException(status_code=503, detail="Workflow not initialized")
        if vision_agent is None:
            raise HTTPException(
                status_code=503,
                detail="Vision model not available. Please use the Anthropic provider.",
            )
        active_workflow = workflow
        active_vision_agent = vision_agent

    # Node name to user-friendly stage mapping
    node_stage_map = {
        "annotate": ("annotating", "Generating HED annotation..."),
        "validate": ("validating", "Validating HED annotation..."),
        "summarize_feedback": ("refining", "Processing validation feedback..."),
        "evaluate": ("evaluating", "Evaluating annotation faithfulness..."),
        "assess": ("assessing", "Running final assessment..."),
    }

    async def event_generator(usage: UsageLedger):
        """Generate SSE events for image annotation workflow progress."""

        def send_event(event_type: str, data: dict) -> str:
            return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

        # SSE padding comment to force Safari to open the stream
        yield ": stream opened\n\n"

        start_time = time.time()
        current_state: dict = {}
        image_description = ""

        try:
            # Send initial start event
            yield send_event(
                "progress", {"stage": "starting", "message": "Initializing image annotation..."}
            )

            # Step 1: Generate image description using vision model
            yield send_event(
                "progress", {"stage": "vision", "message": "Analyzing image with vision model..."}
            )

            vision_result = await active_vision_agent.describe_image(
                image_data=request.image,
                custom_prompt=request.prompt,
            )

            image_description = vision_result["description"]
            image_metadata = vision_result["metadata"]

            # Send image description event
            yield send_event(
                "image_description",
                {"description": image_description, "metadata": image_metadata},
            )

            # Step 2: Create initial state for annotation workflow
            initial_state = create_initial_state(
                image_description,
                request.schema_version,
                request.max_validation_attempts,
                run_assessment=request.run_assessment,
                no_extend=request.no_extend,
            )

            # Track state and progress
            current_state = initial_state.copy()
            last_stage = None
            validation_attempt = 0

            # Use LangGraph's astream_events for real-time streaming
            config = {"recursion_limit": 50}
            async for event in active_workflow.graph.astream_events(  # type: ignore[union-attr]
                initial_state, config=config, version="v2"
            ):
                event_type = event.get("event")
                name = event.get("name", "")

                # Handle node start events
                if event_type == "on_chain_start" and name in node_stage_map:
                    stage, message = node_stage_map[name]

                    # Track validation attempts
                    if name == "validate":
                        validation_attempt += 1

                    # Only send if stage changed
                    if stage != last_stage:
                        last_stage = stage
                        progress_data = {
                            "stage": stage,
                            "message": message,
                        }
                        if name == "validate":
                            progress_data["attempt"] = validation_attempt
                        yield send_event("progress", progress_data)

                # Handle node end events to get intermediate state
                if event_type == "on_chain_end" and name in node_stage_map:
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        current_state.update(output)

                        # Send validation result events
                        if name == "validate":
                            is_valid = output.get("is_valid", False)
                            errors = output.get("validation_errors", [])
                            if is_valid:
                                yield send_event(
                                    "validation",
                                    {
                                        "valid": True,
                                        "attempt": validation_attempt,
                                        "message": "Validation passed",
                                    },
                                )
                            elif errors:
                                tag_suggestions = output.get("tag_suggestions", {})
                                validation_data = {
                                    "valid": False,
                                    "attempt": validation_attempt,
                                    "errors": errors[:3],  # Send first 3 errors
                                    "message": f"Found {len(errors)} validation error(s)",
                                }
                                if tag_suggestions:
                                    validation_data["tag_suggestions"] = tag_suggestions
                                yield send_event("validation", validation_data)

            # Send final result
            is_valid = (
                current_state.get("is_valid", False)
                and len(current_state.get("validation_errors", [])) == 0
            )
            status = "success" if is_valid else "failed"
            result = {
                "image_description": image_description,
                "annotation": current_state.get("current_annotation", ""),
                "is_valid": is_valid,
                "is_faithful": current_state.get("is_faithful", False),
                "is_complete": current_state.get("is_complete", False),
                "validation_attempts": current_state.get("validation_attempts", 0),
                "validation_errors": current_state.get("validation_errors", []),
                "validation_warnings": current_state.get("validation_warnings", []),
                "tag_suggestions": current_state.get("tag_suggestions", {}),
                "evaluation_feedback": current_state.get("evaluation_feedback", ""),
                "assessment_feedback": current_state.get("assessment_feedback", ""),
                "status": status,
                "image_metadata": image_metadata,
            }
            usage_summary = _usage_summary(usage)
            if usage_summary is not None:
                result["usage"] = usage_summary.model_dump()

            yield send_event("result", result)

            # Collect telemetry after sending result but before done event
            try:
                await _collect_stream_telemetry(
                    request=request,
                    req=req,
                    current_state=current_state,
                    start_time=start_time,
                    source="api-image-stream",
                    description=image_description,
                    usage=usage,
                )
            except Exception:
                logging.debug(
                    "Telemetry collection failed for image streaming request", exc_info=True
                )

            yield send_event("done", {"message": "Workflow completed"})

        except asyncio.CancelledError:
            raise
        except Exception as e:
            _status, error_type, message = _describe_llm_error(e)
            logging.exception("Streaming image workflow error (%s)", error_type)
            yield send_event("error", {"message": message, "error_type": error_type})
            # Collect telemetry on error
            try:
                await _collect_stream_telemetry(
                    request=request,
                    req=req,
                    current_state=current_state,
                    start_time=start_time,
                    source="api-image-stream",
                    description=image_description or "image-annotation-failed",
                    usage=usage,
                )
            except Exception:
                logging.warning("Telemetry collection failed on image error", exc_info=True)
            yield send_event("done", {"message": "Workflow ended with error"})

    async def streamed_events():
        """Hold one usage scope open for the whole stream.

        The scope lives outside the generator that runs the workflow so that
        every LLM call made while the response streams is attributed to this
        request.
        """
        with usage_scope() as usage:
            async for chunk in event_generator(usage):
                yield chunk

    return StreamingResponse(
        streamed_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "X-Content-Type-Options": "nosniff",  # Prevent MIME-type sniffing; helps Safari trust text/event-stream
        },
    )


@app.post("/validate", response_model=ValidationResponse)
async def validate(
    request: ValidationRequest, api_key: str = Depends(api_key_auth)
) -> ValidationResponse:
    """Validate a HED annotation string.

    Requires API key authentication via X-API-Key header.

    Args:
        request: Validation request with HED string
        api_key: API key for authentication (injected by dependency)

    Returns:
        Validation result with errors and warnings

    Raises:
        HTTPException: If validation fails or authentication fails
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


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Submit user feedback about an annotation.

    This endpoint is public (no authentication required) to allow feedback
    from frontend and CLI users without requiring API keys.

    The feedback is saved and optionally processed immediately if GITHUB_TOKEN
    is available in the environment. Otherwise, feedback is saved for later
    processing via CI workflow.

    Args:
        request: Feedback submission with annotation data and user comment

    Returns:
        FeedbackResponse with feedback ID and status
    """
    from uuid import uuid4

    try:
        # Generate unique feedback ID
        feedback_id = str(uuid4())[:8]
        timestamp = datetime.now().isoformat()

        # Create feedback record
        feedback_record = {
            "feedback_id": feedback_id,
            "timestamp": timestamp,
            "version": __version__,
            "type": request.type,
            "description": request.description,
            "image_description": request.image_description,
            "annotation": request.annotation,
            "is_valid": request.is_valid,
            "is_faithful": request.is_faithful,
            "is_complete": request.is_complete,
            "validation_errors": request.validation_errors,
            "validation_warnings": request.validation_warnings,
            "evaluation_feedback": request.evaluation_feedback,
            "assessment_feedback": request.assessment_feedback,
            "user_comment": request.user_comment,
        }

        # Save to feedback/unprocessed directory (always save for backup/audit)
        feedback_dir = Path("feedback/unprocessed")
        feedback_dir.mkdir(parents=True, exist_ok=True)

        filename = f"feedback-{timestamp.replace(':', '-').replace('.', '-')}.jsonl"
        filepath = feedback_dir / filename

        with open(filepath, "w") as f:
            f.write(json.dumps(feedback_record) + "\n")

        # Log the feedback submission
        audit_logger.log(
            event="feedback_submitted",
            data={"feedback_id": feedback_id, "type": request.type},
        )

        # Try to process immediately if GitHub token and Anthropic key are available
        processing_result = None
        github_token = os.getenv("GITHUB_TOKEN")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        if github_token and anthropic_key:
            try:
                from src.agents.feedback_triage_agent import (
                    FeedbackRecord as FeedbackRecordModel,
                )
                from src.agents.feedback_triage_agent import (
                    FeedbackTriageAgent,
                    save_processed_feedback,
                )
                from src.utils.github_client import GitHubClient

                # Create feedback record model
                record = FeedbackRecordModel.from_json(feedback_record)

                # Create GitHub client
                github_client = GitHubClient(
                    token=github_token,
                    owner=os.getenv("GITHUB_REPOSITORY_OWNER", "Annotation-Garden"),
                    repo=os.getenv("GITHUB_REPOSITORY", "hedit").split("/")[-1],
                )

                # Create LLM for triage (server credentials from the environment)
                model = os.getenv("ANNOTATION_MODEL", DEFAULT_MODEL)
                llm = create_anthropic_llm(
                    model=model,
                    temperature=0.1,
                    max_tokens=1000,
                    role="triage",
                )

                # Create and run triage agent
                agent = FeedbackTriageAgent(llm=llm, github_client=github_client)
                processing_result = await agent.process_and_execute(record, dry_run=False)

                # Save processed result
                save_processed_feedback(record, processing_result, Path("feedback/processed"))

                # Remove the original feedback file since it's been processed
                filepath.unlink(missing_ok=True)

                audit_logger.log(
                    event="feedback_processed",
                    data={
                        "feedback_id": feedback_id,
                        "action": processing_result.get("action"),
                        "issue_number": processing_result.get("issue_number"),
                    },
                )

            except Exception as e:
                # Log error but don't fail the request - feedback is still saved
                logging.exception("Feedback triage processing failed")
                audit_logger.log(
                    event="feedback_processing_error",
                    data={"feedback_id": feedback_id, "error": str(e)},
                )

        # Build response message
        if processing_result:
            action = processing_result.get("action", "unknown")
            if action == "create_issue":
                message = f"Thank you! Your feedback has been submitted as issue #{processing_result.get('issue_number')}."
            elif action == "comment":
                message = f"Thank you! Your feedback has been added to existing issue #{processing_result.get('issue_number')}."
            else:
                message = "Thank you for your feedback! It has been archived for review."
        else:
            message = "Thank you for your feedback! It will be reviewed and processed."

        return FeedbackResponse(
            success=True,
            feedback_id=feedback_id,
            message=message,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save feedback: {str(e)}",
        ) from e


@app.get("/metrics", response_model=MetricsResponse)
async def metrics(api_key: str = Depends(api_key_auth)) -> MetricsResponse:
    """Report LLM token usage, cost, and prompt-cache savings since startup.

    These are server-wide operator figures, so BYOK callers are refused: a
    BYOK request gets its own numbers in the ``usage`` field of its
    annotation response instead.

    Args:
        api_key: Authentication result (injected by dependency)

    Returns:
        Totals since startup, broken down by agent role and by model

    Raises:
        HTTPException: 403 when authenticated via BYOK
    """
    if api_key == "byok":
        raise HTTPException(
            status_code=403,
            detail="Server metrics require a server API key; "
            "per-request usage is returned in the annotation response.",
        )

    snapshot = process_ledger().snapshot()
    return MetricsResponse(
        since=_startup_time,
        total=UsageSummary(**snapshot["total"]),
        by_role={role: UsageSummary(**totals) for role, totals in snapshot["by_role"].items()},
        by_model={model: UsageSummary(**totals) for model, totals in snapshot["by_model"].items()},
    )


@app.get("/version")
async def get_version():
    """Get API version information.

    Returns:
        Version information including commit hash for deployment verification
    """
    return {
        "version": __version__,
        "commit": os.getenv("GIT_COMMIT", "unknown"),
    }


@app.get("/")
async def root():
    """Root endpoint with API information.

    Returns:
        API information
    """
    return {
        "name": "HEDit API",
        "version": __version__,
        "description": "Multi-agent system for HED annotation generation",
        "endpoints": {
            "POST /annotate": "Generate HED annotation from description",
            "POST /annotate/stream": "Generate HED annotation with streaming progress",
            "POST /annotate-from-image": "Generate HED annotation from image",
            "POST /annotate-from-image/stream": "Generate HED annotation from image with streaming",
            "POST /validate": "Validate HED annotation string",
            "POST /feedback": "Submit user feedback about annotation",
            "GET /health": "Health check",
            "GET /version": "Get version information",
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
