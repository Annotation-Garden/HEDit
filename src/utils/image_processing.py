"""Image processing utilities for vision model integration.

This module provides utilities for handling image data in base64 format,
validating images, and preparing them for vision model processing.
"""

import base64
import io

from PIL import Image

MAX_IMAGE_SIZE_MB = 10
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
SUPPORTED_FORMATS = {"PNG", "JPEG", "JPG", "WEBP"}


class ImageProcessingError(Exception):
    """Base exception for image processing errors."""

    pass


class InvalidImageFormatError(ImageProcessingError):
    """Raised when image format is not supported."""

    pass


class ImageSizeExceededError(ImageProcessingError):
    """Raised when image size exceeds maximum allowed size."""

    pass


class InvalidBase64Error(ImageProcessingError):
    """Raised when base64 encoding is invalid."""

    pass


def parse_data_uri(data_uri: str) -> tuple[str, str]:
    """Parse a data URI and extract mime type and base64 data.

    Args:
        data_uri: Data URI string (e.g., "data:image/png;base64,iVBORw0KG...")

    Returns:
        Tuple of (mime_type, base64_data)

    Raises:
        InvalidBase64Error: If data URI format is invalid
    """
    # Parse data URI using string operations (safer than regex for large inputs)
    if not data_uri.startswith("data:"):
        raise InvalidBase64Error(
            "Invalid data URI format. Expected: data:image/<type>;base64,<data>"
        )

    # Find the separator between metadata and data
    base64_marker = ";base64,"
    marker_pos = data_uri.find(base64_marker)
    if marker_pos == -1:
        raise InvalidBase64Error(
            "Invalid data URI format. Expected: data:image/<type>;base64,<data>"
        )

    # Extract mime type (between "data:" and ";base64,")
    mime_type = data_uri[5:marker_pos]  # 5 = len("data:")
    if not mime_type:
        raise InvalidBase64Error(
            "Invalid data URI format. Expected: data:image/<type>;base64,<data>"
        )

    # Extract base64 data (after ";base64,")
    base64_data = data_uri[marker_pos + len(base64_marker) :]
    if not base64_data:
        raise InvalidBase64Error(
            "Invalid data URI format. Expected: data:image/<type>;base64,<data>"
        )

    return mime_type, base64_data


def decode_base64_image(base64_str: str, validate: bool = True) -> tuple[Image.Image, dict]:
    """Decode a base64 string to a PIL Image and extract metadata.

    Args:
        base64_str: Base64 encoded image string or data URI
        validate: Whether to validate image format and size

    Returns:
        Tuple of (PIL Image object, metadata dict)

    Raises:
        InvalidBase64Error: If base64 decoding fails
        InvalidImageFormatError: If image format is not supported
        ImageSizeExceededError: If image size exceeds limit
    """
    # Handle data URI format
    mime_type = None
    if base64_str.startswith("data:"):
        mime_type, base64_str = parse_data_uri(base64_str)

    # Decode base64
    try:
        image_bytes = base64.b64decode(base64_str)
    except Exception as e:
        raise InvalidBase64Error(f"Failed to decode base64 string: {e}") from e

    # Check size
    size_bytes = len(image_bytes)
    if validate and size_bytes > MAX_IMAGE_SIZE_BYTES:
        size_mb = size_bytes / (1024 * 1024)
        raise ImageSizeExceededError(
            f"Image size ({size_mb:.2f}MB) exceeds maximum allowed size ({MAX_IMAGE_SIZE_MB}MB)"
        )

    # Open image
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise InvalidImageFormatError(f"Failed to open image: {e}") from e

    # Validate format
    if validate and image.format not in SUPPORTED_FORMATS:
        raise InvalidImageFormatError(
            f"Image format '{image.format}' is not supported. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )

    # Extract metadata
    metadata = {
        "format": image.format,
        "mode": image.mode,
        "size": image.size,
        "width": image.width,
        "height": image.height,
        "size_bytes": size_bytes,
        "size_mb": size_bytes / (1024 * 1024),
        "mime_type": mime_type or f"image/{image.format.lower()}",
    }

    return image, metadata


def encode_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Encode a PIL Image to base64 string.

    Args:
        image: PIL Image object
        format: Image format (PNG, JPEG, etc.)

    Returns:
        Base64 encoded string
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    base64_bytes = base64.b64encode(buffer.read())
    return base64_bytes.decode("utf-8")


def create_data_uri(image: Image.Image, format: str = "PNG") -> str:
    """Create a data URI from a PIL Image.

    Args:
        image: PIL Image object
        format: Image format (PNG, JPEG, etc.)

    Returns:
        Data URI string (e.g., "data:image/png;base64,iVBORw0KG...")
    """
    base64_str = encode_image_to_base64(image, format)
    mime_type = f"image/{format.lower()}"
    return f"data:{mime_type};base64,{base64_str}"


def validate_image_data(base64_str: str) -> dict:
    """Validate image data and return metadata without loading full image.

    Args:
        base64_str: Base64 encoded image string or data URI

    Returns:
        Dictionary with validation results and metadata

    Example:
        {
            "valid": True,
            "error": None,
            "metadata": {...}
        }
    """
    try:
        _, metadata = decode_base64_image(base64_str, validate=True)
        return {"valid": True, "error": None, "metadata": metadata}
    except ImageProcessingError as e:
        return {"valid": False, "error": str(e), "metadata": None}


def prepare_image_for_vision_model(base64_str: str) -> tuple[str, dict]:
    """Prepare image for vision model processing.

    This function validates the image and ensures it's in the correct format
    for the vision model API.

    Args:
        base64_str: Base64 encoded image string or data URI

    Returns:
        Tuple of (data_uri, metadata)

    Raises:
        ImageProcessingError: If image validation fails
    """
    # Decode and validate
    image, metadata = decode_base64_image(base64_str, validate=True)

    # Convert to data URI if not already
    if base64_str.startswith("data:"):
        data_uri = base64_str
    else:
        # Use original format if possible, otherwise PNG
        format = metadata.get("format", "PNG")
        data_uri = create_data_uri(image, format)

    return data_uri, metadata
