"""Vision Agent for generating image descriptions using vision-language models.

This agent is responsible for analyzing images and generating detailed natural
language descriptions that can be used for HED annotation.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.utils.image_processing import prepare_image_for_vision_model

DEFAULT_VISION_PROMPT = """Describe what you see in this image. Include the setting, main elements, colors, lighting, and overall composition. Be specific and detailed. Form the response as a continuous paragraph. Maximum 200 words."""


class VisionAgent:
    """Agent that generates descriptions from images using vision-language models.

    This agent uses a vision-language model to analyze images and generate
    detailed descriptions that can be fed into the HED annotation pipeline.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        default_prompt: str = DEFAULT_VISION_PROMPT,
    ) -> None:
        """Initialize the vision agent.

        Args:
            llm: Vision-language model for image description
            default_prompt: Default prompt to use if none provided
        """
        self.llm = llm
        self.default_prompt = default_prompt

    async def describe_image(
        self,
        image_data: str,
        custom_prompt: str | None = None,
    ) -> dict:
        """Generate a description of an image.

        Args:
            image_data: Base64 encoded image or data URI
            custom_prompt: Optional custom prompt (uses default if not provided)

        Returns:
            Dictionary containing:
                - description: Generated image description
                - prompt_used: The prompt that was used
                - metadata: Image metadata from processing

        Raises:
            ImageProcessingError: If image validation fails
        """
        # Prepare image and validate
        data_uri, metadata = prepare_image_for_vision_model(image_data)

        # Use custom prompt or default
        prompt = custom_prompt or self.default_prompt

        # Create message with image content
        # Vision models support content with both text and images
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]
        )

        # Generate description
        response = await self.llm.ainvoke([message])
        description = response.content.strip()

        return {
            "description": description,
            "prompt_used": prompt,
            "metadata": metadata,
        }

    def describe_image_sync(
        self,
        image_data: str,
        custom_prompt: str | None = None,
    ) -> dict:
        """Synchronous version of describe_image.

        Args:
            image_data: Base64 encoded image or data URI
            custom_prompt: Optional custom prompt (uses default if not provided)

        Returns:
            Dictionary containing:
                - description: Generated image description
                - prompt_used: The prompt that was used
                - metadata: Image metadata from processing

        Raises:
            ImageProcessingError: If image validation fails
        """
        # Prepare image and validate
        data_uri, metadata = prepare_image_for_vision_model(image_data)

        # Use custom prompt or default
        prompt = custom_prompt or self.default_prompt

        # Create message with image content
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]
        )

        # Generate description
        response = self.llm.invoke([message])
        description = response.content.strip()

        return {
            "description": description,
            "prompt_used": prompt,
            "metadata": metadata,
        }


def create_vision_agent(
    llm: BaseChatModel,
    custom_prompt: str | None = None,
) -> VisionAgent:
    """Factory function to create a vision agent.

    Args:
        llm: Vision-language model
        custom_prompt: Optional custom default prompt

    Returns:
        Configured VisionAgent instance
    """
    return VisionAgent(
        llm=llm,
        default_prompt=custom_prompt or DEFAULT_VISION_PROMPT,
    )
