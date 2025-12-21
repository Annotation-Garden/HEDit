"""Local execution backend for HEDit CLI (standalone mode).

Runs the LangGraph workflow locally without requiring the HEDit backend.
This mode requires additional dependencies: pip install hedit[standalone]

Dependencies: langgraph, langchain, langchain-openai, hedtools, etc.
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.cli.executor import ExecutionBackend, ExecutionError
from src.version import __version__

if TYPE_CHECKING:
    from src.agents.vision_agent import VisionAgent
    from src.agents.workflow import HedAnnotationWorkflow

# Flag to track if standalone dependencies are available
_STANDALONE_AVAILABLE: bool | None = None


def _check_standalone_deps() -> bool:
    """Check if standalone mode dependencies are installed."""
    global _STANDALONE_AVAILABLE
    if _STANDALONE_AVAILABLE is not None:
        return _STANDALONE_AVAILABLE

    try:
        import langchain  # noqa: F401
        import langchain_openai  # noqa: F401
        import langgraph  # noqa: F401

        # hedtools is optional for now (uses GitHub schema fetch)
        _STANDALONE_AVAILABLE = True
    except ImportError:
        _STANDALONE_AVAILABLE = False

    return _STANDALONE_AVAILABLE


class LocalExecutionBackend(ExecutionBackend):
    """Execution backend that runs the LangGraph workflow locally.

    This backend directly executes the multi-agent annotation workflow
    on the local machine. It requires the standalone dependencies to be
    installed via `pip install hedit[standalone]`.

    Benefits:
    - No dependency on external HEDit infrastructure
    - Better privacy (all processing local except LLM calls)
    - Faster response times (no extra network hop to backend)
    - Works offline except for OpenRouter LLM calls

    Requirements:
    - OpenRouter API key
    - Standalone dependencies installed
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        vision_model: str | None = None,
        provider: str | None = None,
        temperature: float = 0.1,
        schema_dir: Path | str | None = None,
    ):
        """Initialize local execution backend.

        Args:
            api_key: OpenRouter API key (required for LLM operations, optional for health/validate)
            model: Model for text annotation (default: openai/gpt-oss-120b)
            vision_model: Model for image annotation (default: qwen/qwen3-vl-30b-a3b-instruct)
            provider: Provider preference (cleared if custom model specified)
            temperature: LLM temperature (0.0-1.0)
            schema_dir: Optional directory with JSON schemas (None = fetch from GitHub)
        """
        # API key is optional at init time - only required for LLM operations

        self._api_key = api_key
        self._model = model or "openai/gpt-oss-120b"
        self._vision_model = vision_model or "qwen/qwen3-vl-30b-a3b-instruct"
        self._temperature = temperature
        self._schema_dir = Path(schema_dir) if schema_dir else None

        # Handle provider logic: clear if custom model specified
        # (Cerebras only works with default models)
        if provider is not None:
            self._provider = provider if provider else None
        elif model is not None and model != "openai/gpt-oss-120b":
            self._provider = None
        else:
            self._provider = "Cerebras"

        # Lazy initialization of workflow and vision agent
        self._workflow: HedAnnotationWorkflow | None = None
        self._vision_agent: VisionAgent | None = None

    @property
    def mode(self) -> str:
        """Get execution mode name."""
        return "standalone"

    def is_available(self) -> bool:
        """Check if standalone dependencies are installed."""
        return _check_standalone_deps()

    def _ensure_deps(self) -> None:
        """Ensure standalone dependencies are available."""
        if not self.is_available():
            raise ExecutionError(
                "Standalone mode requires additional dependencies",
                code="missing_dependencies",
                detail="Install with: pip install hedit[standalone]",
            )

    def _ensure_api_key(self) -> None:
        """Ensure API key is available for LLM operations."""
        if not self._api_key:
            raise ExecutionError(
                "OpenRouter API key required for standalone mode",
                code="missing_api_key",
                detail="Provide --api-key or run 'hedit init'",
            )

    def _get_workflow(self) -> HedAnnotationWorkflow:
        """Get or create the annotation workflow."""
        if self._workflow is None:
            self._ensure_deps()
            self._ensure_api_key()

            from src.agents.workflow import HedAnnotationWorkflow
            from src.cli.config import get_machine_id
            from src.utils.openrouter_llm import create_openrouter_llm

            # Get machine ID for cache optimization
            user_id = get_machine_id()

            # Create LLMs with user's key
            annotation_llm = create_openrouter_llm(
                model=self._model,
                api_key=self._api_key,
                temperature=self._temperature,
                provider=self._provider,
                user_id=user_id,
            )

            # Use same settings for all agents in standalone mode
            # (simplification for CLI usage)
            self._workflow = HedAnnotationWorkflow(
                llm=annotation_llm,
                evaluation_llm=annotation_llm,
                assessment_llm=annotation_llm,
                schema_dir=self._schema_dir,
                use_js_validator=False,  # Use Python validator in standalone
            )

        return self._workflow

    def _get_vision_agent(self) -> VisionAgent:
        """Get or create the vision agent."""
        if self._vision_agent is None:
            self._ensure_deps()
            self._ensure_api_key()

            from src.agents.vision_agent import VisionAgent
            from src.cli.config import get_machine_id
            from src.utils.openrouter_llm import create_openrouter_llm

            # Get machine ID for cache optimization
            user_id = get_machine_id()

            vision_llm = create_openrouter_llm(
                model=self._vision_model,
                api_key=self._api_key,
                temperature=0.3,  # Slightly higher for vision tasks
                provider=self._provider,
                user_id=user_id,
            )

            self._vision_agent = VisionAgent(llm=vision_llm)

        return self._vision_agent

    def _run_async(self, coro):
        """Run async coroutine synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # Already in async context - use nest_asyncio or create task
            import nest_asyncio

            nest_asyncio.apply()
            return asyncio.get_event_loop().run_until_complete(coro)
        else:
            return asyncio.run(coro)

    def annotate(
        self,
        description: str,
        schema_version: str = "8.4.0",
        max_validation_attempts: int = 5,
        run_assessment: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate HED annotation locally."""
        self._ensure_deps()

        workflow = self._get_workflow()

        async def _run():
            return await workflow.run(
                input_description=description,
                schema_version=schema_version,
                max_validation_attempts=max_validation_attempts,
                run_assessment=run_assessment,
            )

        try:
            final_state = self._run_async(_run())
        except Exception as e:
            raise ExecutionError(
                f"Annotation failed: {e}",
                code="workflow_error",
                detail=str(e),
            ) from e

        # Convert state to response format
        return {
            "status": "success" if final_state.get("is_valid") else "error",
            "hed_string": final_state.get("current_annotation", ""),
            "is_valid": final_state.get("is_valid", False),
            "validation_messages": final_state.get("validation_errors", []),
            "metadata": {
                "schema_version": schema_version,
                "validation_attempts": final_state.get("validation_attempts", 0),
                "total_iterations": final_state.get("total_iterations", 0),
                "is_faithful": final_state.get("is_faithful"),
                "is_complete": final_state.get("is_complete"),
                "evaluation_feedback": final_state.get("evaluation_feedback"),
                "assessment_feedback": final_state.get("assessment_feedback"),
                "mode": "standalone",
            },
        }

    def annotate_image(
        self,
        image_path: Path | str,
        prompt: str | None = None,
        schema_version: str = "8.4.0",
        max_validation_attempts: int = 5,
        run_assessment: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate HED annotation from image locally."""
        self._ensure_deps()

        image_path = Path(image_path)
        if not image_path.exists():
            raise ExecutionError(
                f"Image file not found: {image_path}",
                code="file_not_found",
            )

        # Read and encode image
        suffix = image_path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_types.get(suffix, "image/png")

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        image_uri = f"data:{mime_type};base64,{image_data}"

        vision_agent = self._get_vision_agent()
        workflow = self._get_workflow()

        async def _run():
            # Step 1: Generate image description
            vision_result = await vision_agent.describe_image(
                image_data=image_uri,
                custom_prompt=prompt,
            )
            description = vision_result["description"]

            # Step 2: Generate HED annotation from description
            final_state = await workflow.run(
                input_description=description,
                schema_version=schema_version,
                max_validation_attempts=max_validation_attempts,
                run_assessment=run_assessment,
            )

            return description, vision_result, final_state

        try:
            description, vision_result, final_state = self._run_async(_run())
        except Exception as e:
            raise ExecutionError(
                f"Image annotation failed: {e}",
                code="workflow_error",
                detail=str(e),
            ) from e

        return {
            "status": "success" if final_state.get("is_valid") else "error",
            "description": description,
            "hed_string": final_state.get("current_annotation", ""),
            "is_valid": final_state.get("is_valid", False),
            "validation_messages": final_state.get("validation_errors", []),
            "metadata": {
                "schema_version": schema_version,
                "vision_prompt": vision_result.get("prompt_used"),
                "image_metadata": vision_result.get("metadata"),
                "validation_attempts": final_state.get("validation_attempts", 0),
                "total_iterations": final_state.get("total_iterations", 0),
                "is_faithful": final_state.get("is_faithful"),
                "is_complete": final_state.get("is_complete"),
                "mode": "standalone",
            },
        }

    def validate(
        self,
        hed_string: str,
        schema_version: str = "8.4.0",
    ) -> dict[str, Any]:
        """Validate HED string locally.

        Uses JavaScript validator if available (Node.js + hed-javascript),
        otherwise falls back to Python validator (hedtools).
        """
        self._ensure_deps()

        try:
            from src.validation.hed_validator import (
                get_validator,
                is_js_validator_available,
            )

            # Get validator (prefers JS if available, falls back to Python)
            validator = get_validator(
                schema_version=schema_version,
                prefer_js=True,
                require_js=False,  # Allow fallback to Python
            )
            result = validator.validate(hed_string)

            # Convert ValidationResult to dict format
            messages = []
            for error in result.errors:
                messages.append(f"[ERROR] {error.message}")
            for warning in result.warnings:
                messages.append(f"[WARNING] {warning.message}")

            # Note which validator was used
            validator_type = "javascript" if is_js_validator_available() else "python"

            return {
                "is_valid": result.is_valid,
                "messages": messages,
                "validator": validator_type,
            }
        except ImportError:
            # hedtools not installed, return a warning
            return {
                "is_valid": None,
                "messages": [
                    "Local validation requires hedtools. Install with: pip install hedtools"
                ],
                "validator": None,
            }
        except Exception as e:
            raise ExecutionError(
                f"Validation failed: {e}",
                code="validation_error",
                detail=str(e),
            ) from e

    def health(self) -> dict[str, Any]:
        """Check local backend health."""
        deps_available = self.is_available()

        # Check hedtools availability
        hedtools_available = False
        try:
            import hed  # noqa: F401

            hedtools_available = True
        except ImportError:
            pass

        # Check JS validator availability
        js_validator_available = False
        try:
            from src.validation.hed_validator import is_js_validator_available

            js_validator_available = is_js_validator_available()
        except ImportError:
            pass

        # Determine which validator will be used
        if js_validator_available:
            validator_type = "javascript"
        elif hedtools_available:
            validator_type = "python"
        else:
            validator_type = None

        return {
            "status": "healthy" if deps_available else "unhealthy",
            "version": __version__,
            "mode": "standalone",
            "llm_available": deps_available,
            "validator_available": hedtools_available or js_validator_available,
            "validator_type": validator_type,
            "dependencies": {
                "langgraph": deps_available,
                "langchain": deps_available,
                "hedtools": hedtools_available,
                "hed_javascript": js_validator_available,
            },
        }
