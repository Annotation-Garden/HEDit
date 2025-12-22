"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field


class AnnotationRequest(BaseModel):
    """Request model for HED annotation generation.

    Attributes:
        description: Natural language event description to annotate
        schema_version: HED schema version to use
        max_validation_attempts: Maximum validation retry attempts
        run_assessment: Whether to run final assessment (adds extra time)
        model: Override model for annotation (BYOK mode only)
        provider: Override provider preference (BYOK mode only)
        temperature: Override LLM temperature (BYOK mode only)
    """

    description: str = Field(
        ...,
        description="Natural language event description",
        min_length=1,
        examples=["A red circle appears on the left side of the screen"],
    )
    schema_version: str = Field(
        default="8.3.0",
        description="HED schema version",
        examples=["8.3.0", "8.4.0"],
    )
    max_validation_attempts: int = Field(
        default=5,
        description="Maximum validation retry attempts",
        ge=1,
        le=10,
    )
    run_assessment: bool = Field(
        default=False,
        description="Run final assessment for completeness (adds extra processing time)",
    )
    # BYOK model configuration (optional, only used when X-OpenRouter-Key is provided)
    model: str | None = Field(
        default=None,
        description="Override model for annotation (BYOK mode only, e.g., 'openai/gpt-4o')",
        examples=["openai/gpt-4o", "anthropic/claude-3.5-sonnet", "qwen/qwen3-235b-a22b-2507"],
    )
    provider: str | None = Field(
        default=None,
        description="Override provider preference (BYOK mode only, e.g., 'Cerebras')",
        examples=["Cerebras", "deepinfra/fp8", None],
    )
    temperature: float | None = Field(
        default=None,
        description="Override LLM temperature (BYOK mode only, 0.0-1.0)",
        ge=0.0,
        le=1.0,
        examples=[0.1, 0.3, 0.7],
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
        default="8.3.0",
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
        model: Override model for annotation (BYOK mode only)
        vision_model: Override vision model for image description (BYOK mode only)
        provider: Override provider preference (BYOK mode only)
        temperature: Override LLM temperature (BYOK mode only)
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
        default=5,
        description="Maximum validation retry attempts",
        ge=1,
        le=10,
    )
    run_assessment: bool = Field(
        default=False,
        description="Run final assessment for completeness (adds extra processing time)",
    )
    # BYOK model configuration (optional, only used when X-OpenRouter-Key is provided)
    model: str | None = Field(
        default=None,
        description="Override model for annotation (BYOK mode only, e.g., 'openai/gpt-4o')",
        examples=["openai/gpt-4o", "anthropic/claude-3.5-sonnet"],
    )
    vision_model: str | None = Field(
        default=None,
        description="Override vision model for image description (BYOK mode only)",
        examples=["qwen/qwen3-vl-30b-a3b-instruct", "openai/gpt-4o"],
    )
    provider: str | None = Field(
        default=None,
        description="Override provider preference (BYOK mode only, e.g., 'Cerebras')",
        examples=["Cerebras", "deepinfra/fp8", None],
    )
    temperature: float | None = Field(
        default=None,
        description="Override LLM temperature (BYOK mode only, 0.0-1.0)",
        ge=0.0,
        le=1.0,
        examples=[0.1, 0.3, 0.7],
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
