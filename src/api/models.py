"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field


class UsageSummary(BaseModel):
    """Token, cost, and prompt-cache figures for the LLM calls of one request.

    Every HEDit annotation re-sends a large static HED vocabulary guide, which
    prompt caching serves at a tenth of the input price after the first call.
    ``uncached_cost_usd`` is what the same calls would have cost without any
    caching, so ``savings_usd`` reports the difference rather than an estimate.

    Costs use Anthropic list prices and cover LLM calls only.

    Attributes:
        calls: Number of LLM calls made
        input_tokens: Total input tokens, including tokens served from cache
        uncached_input_tokens: Input tokens billed at full price
        cache_read_tokens: Input tokens served from the prompt cache
        cache_write_tokens: Input tokens written to the prompt cache
        output_tokens: Generated tokens
        total_tokens: Input plus output tokens
        cache_hit_rate: Share of input tokens served from cache (0.0-1.0)
        cost_usd: Cost with caching applied
        uncached_cost_usd: Cost the same calls would have had without caching
        savings_usd: Dollars not spent because of prompt caching
        savings_pct: Savings as a share of the uncached cost (0.0-1.0)
        models: Models that served the calls
        unpriced_calls: Calls on models with no published price in this build
    """

    calls: int = Field(..., description="Number of LLM calls")
    input_tokens: int = Field(..., description="Total input tokens (cached included)")
    uncached_input_tokens: int = Field(..., description="Input tokens billed at full price")
    cache_read_tokens: int = Field(..., description="Input tokens served from cache")
    cache_write_tokens: int = Field(..., description="Input tokens written to cache")
    output_tokens: int = Field(..., description="Generated tokens")
    total_tokens: int = Field(..., description="Input plus output tokens")
    cache_hit_rate: float = Field(..., description="Cached share of input tokens (0.0-1.0)")
    cost_usd: float = Field(..., description="Cost with caching applied")
    uncached_cost_usd: float = Field(..., description="Cost without any caching")
    savings_usd: float = Field(..., description="Dollars saved by prompt caching")
    savings_pct: float = Field(..., description="Share of cost saved (0.0-1.0)")
    models: list[str] = Field(default_factory=list, description="Models that served the calls")
    unpriced_calls: int = Field(default=0, description="Calls on models with no published price")


class MetricsResponse(BaseModel):
    """Server-wide LLM usage since startup.

    Attributes:
        since: Server startup time (ISO 8601)
        total: Combined totals across every agent role
        by_role: Totals per agent role (annotation, evaluation, vision, ...)
        by_model: Totals per model
    """

    since: str = Field(..., description="Server startup time (ISO 8601)")
    total: UsageSummary = Field(..., description="Combined totals")
    by_role: dict[str, UsageSummary] = Field(default_factory=dict, description="Totals per role")
    by_model: dict[str, UsageSummary] = Field(default_factory=dict, description="Totals per model")


class AnnotationRequest(BaseModel):
    """Request model for HED annotation generation.

    Attributes:
        description: Natural language event description to annotate
        schema_version: HED schema version to use
        max_validation_attempts: Maximum validation retry attempts
        run_assessment: Whether to run final assessment (adds extra time)
        model: Override model for annotation
        provider: Deprecated; ignored (all models are served by Anthropic)
        temperature: Override LLM temperature
    """

    description: str = Field(
        ...,
        description="Natural language event description",
        min_length=1,
        examples=["A red circle appears on the left side of the screen"],
    )
    schema_version: str = Field(
        default="8.4.0",
        description="HED schema version",
        examples=["8.3.0", "8.4.0"],
    )
    max_validation_attempts: int = Field(
        default=3,
        description="Maximum validation retry attempts (total iterations = this + 1)",
        ge=1,
        le=10,
    )
    run_assessment: bool = Field(
        default=False,
        description="Run final assessment for completeness (adds extra processing time)",
    )
    # Model configuration (optional)
    model: str | None = Field(
        default=None,
        description="Override annotation model (e.g., 'claude-sonnet-5')",
        examples=["claude-haiku-4-5", "claude-sonnet-5"],
    )
    provider: str | None = Field(
        default=None,
        description="Deprecated; ignored (all models are served by Anthropic)",
        examples=[None],
    )
    temperature: float | None = Field(
        default=None,
        description="Override LLM temperature (0.0-1.0)",
        ge=0.0,
        le=1.0,
        examples=[0.1, 0.3, 0.7],
    )
    no_extend: bool = Field(
        default=False,
        description="If True, prohibit tag extensions (use only existing HED vocabulary)",
    )
    telemetry_enabled: bool = Field(
        default=True,
        description="Allow telemetry collection for this request",
    )


class AnnotationResponse(BaseModel):
    """Response model for HED annotation generation.

    Attributes:
        annotation: Generated HED annotation string
        is_valid: Whether the annotation passed validation
        is_faithful: Whether the annotation is faithful to description
        is_complete: Whether the annotation is complete
        validation_attempts: Number of validation attempts made
        validation_errors: List of validation errors (if any)
        validation_warnings: List of validation warnings (if any)
        evaluation_feedback: Evaluation agent feedback
        assessment_feedback: Assessment agent feedback
        status: Overall workflow status
        usage: Token, cost, and prompt-cache figures for this request
    """

    annotation: str = Field(..., description="Generated HED annotation string")
    is_valid: bool = Field(..., description="Validation status")
    is_faithful: bool = Field(..., description="Faithfulness to original description")
    is_complete: bool = Field(..., description="Completeness status")
    validation_attempts: int = Field(..., description="Number of validation attempts")
    validation_errors: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    evaluation_feedback: str = Field(default="")
    assessment_feedback: str = Field(default="")
    status: str = Field(..., description="Workflow status", examples=["success", "failed"])
    usage: UsageSummary | None = Field(
        default=None, description="Token, cost, and prompt-cache figures for this request"
    )


class ValidationRequest(BaseModel):
    """Request model for HED validation only.

    Attributes:
        hed_string: HED annotation string to validate
        schema_version: HED schema version to use
    """

    hed_string: str = Field(
        ...,
        description="HED annotation string",
        min_length=1,
    )
    schema_version: str = Field(
        default="8.4.0",
        description="HED schema version",
    )


class ValidationResponse(BaseModel):
    """Response model for HED validation.

    Attributes:
        is_valid: Whether the HED string is valid
        errors: List of validation errors
        warnings: List of validation warnings
        parsed_string: Normalized HED string (if valid)
    """

    is_valid: bool = Field(..., description="Validation status")
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    parsed_string: str | None = Field(default=None)


class ImageAnnotationRequest(BaseModel):
    """Request model for image-based HED annotation generation.

    Attributes:
        image: Base64 encoded image or data URI
        prompt: Optional custom prompt for vision model (uses default if not provided)
        schema_version: HED schema version to use
        max_validation_attempts: Maximum validation retry attempts
        run_assessment: Whether to run final assessment (adds extra time)
        model: Override model for annotation
        vision_model: Override vision model for image description
        provider: Deprecated; ignored (all models are served by Anthropic)
        temperature: Override LLM temperature
    """

    image: str = Field(
        ...,
        description="Base64 encoded image or data URI (data:image/png;base64,...)",
        min_length=1,
    )
    prompt: str | None = Field(
        default=None,
        description="Optional custom prompt for vision model",
        examples=["Describe the visual elements in this image"],
    )
    schema_version: str = Field(
        default="8.4.0",
        description="HED schema version",
        examples=["8.3.0", "8.4.0"],
    )
    max_validation_attempts: int = Field(
        default=3,
        description="Maximum validation retry attempts (total iterations = this + 1)",
        ge=1,
        le=10,
    )
    run_assessment: bool = Field(
        default=False,
        description="Run final assessment for completeness (adds extra processing time)",
    )
    # Model configuration (optional)
    model: str | None = Field(
        default=None,
        description="Override annotation model (e.g., 'claude-sonnet-5')",
        examples=["claude-haiku-4-5", "claude-sonnet-5"],
    )
    vision_model: str | None = Field(
        default=None,
        description="Override vision model for image description",
        examples=["claude-haiku-4-5", "claude-sonnet-5"],
    )
    vision_provider: str | None = Field(
        default=None,
        description="Deprecated; ignored (all models are served by Anthropic)",
        examples=[None],
    )
    provider: str | None = Field(
        default=None,
        description="Deprecated; ignored (all models are served by Anthropic)",
        examples=[None],
    )
    temperature: float | None = Field(
        default=None,
        description="Override LLM temperature (0.0-1.0)",
        ge=0.0,
        le=1.0,
        examples=[0.1, 0.3, 0.7],
    )
    no_extend: bool = Field(
        default=False,
        description="If True, prohibit tag extensions (use only existing HED vocabulary)",
    )
    telemetry_enabled: bool = Field(
        default=True,
        description="Allow telemetry collection for this request",
    )


class ImageAnnotationResponse(BaseModel):
    """Response model for image-based HED annotation generation.

    Attributes:
        image_description: Generated description from vision model
        annotation: Generated HED annotation string
        is_valid: Whether the annotation passed validation
        is_faithful: Whether the annotation is faithful to description
        is_complete: Whether the annotation is complete
        validation_attempts: Number of validation attempts made
        validation_errors: List of validation errors (if any)
        validation_warnings: List of validation warnings (if any)
        evaluation_feedback: Evaluation agent feedback
        assessment_feedback: Assessment agent feedback
        status: Overall workflow status
        image_metadata: Metadata about the processed image
        usage: Token, cost, and prompt-cache figures for this request
    """

    image_description: str = Field(..., description="Generated image description")
    annotation: str = Field(..., description="Generated HED annotation string")
    is_valid: bool = Field(..., description="Validation status")
    is_faithful: bool = Field(..., description="Faithfulness to description")
    is_complete: bool = Field(..., description="Completeness status")
    validation_attempts: int = Field(..., description="Number of validation attempts")
    validation_errors: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    evaluation_feedback: str = Field(default="")
    assessment_feedback: str = Field(default="")
    status: str = Field(..., description="Workflow status", examples=["success", "failed"])
    image_metadata: dict = Field(default_factory=dict, description="Image metadata")
    usage: UsageSummary | None = Field(
        default=None, description="Token, cost, and prompt-cache figures for this request"
    )


class HealthResponse(BaseModel):
    """Response model for health check.

    Attributes:
        status: Service status
        version: API version
        llm_available: Whether LLM is available
        validator_available: Whether HED validator is available
    """

    status: str = Field(..., examples=["healthy", "degraded"])
    version: str = Field(..., examples=["0.1.0"])
    llm_available: bool
    validator_available: bool


class FeedbackRequest(BaseModel):
    """Request model for submitting user feedback.

    Attributes:
        type: Feedback type (text or image annotation)
        description: Original input description (for text mode)
        image_description: Image description (for image mode)
        annotation: Generated HED annotation
        is_valid: Whether the annotation was valid
        is_faithful: Whether the annotation was faithful
        is_complete: Whether the annotation was complete
        validation_errors: List of validation errors
        validation_warnings: List of validation warnings
        evaluation_feedback: Evaluation agent feedback
        assessment_feedback: Assessment agent feedback
        user_comment: Optional user comment about the annotation
    """

    type: str = Field(
        default="text",
        description="Feedback type",
        examples=["text", "image"],
    )
    version: str | None = Field(
        default=None,
        description="App version that generated the annotation",
    )
    description: str | None = Field(
        default=None,
        description="Original input description (for text mode)",
    )
    image_description: str | None = Field(
        default=None,
        description="Image description (for image mode)",
    )
    annotation: str = Field(
        ...,
        description="Generated HED annotation",
        min_length=1,
    )
    is_valid: bool = Field(
        default=False,
        description="Whether the annotation was valid",
    )
    is_faithful: bool | None = Field(
        default=None,
        description="Whether the annotation was faithful",
    )
    is_complete: bool | None = Field(
        default=None,
        description="Whether the annotation was complete",
    )
    validation_errors: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    evaluation_feedback: str = Field(default="")
    assessment_feedback: str = Field(default="")
    user_comment: str | None = Field(
        default=None,
        description="Optional user comment about the annotation",
    )


class FeedbackResponse(BaseModel):
    """Response model for feedback submission.

    Attributes:
        success: Whether feedback was saved successfully
        feedback_id: Unique identifier for the feedback
        message: Status message
    """

    success: bool = Field(..., description="Whether feedback was saved")
    feedback_id: str = Field(..., description="Unique identifier for the feedback")
    message: str = Field(..., description="Status message")
