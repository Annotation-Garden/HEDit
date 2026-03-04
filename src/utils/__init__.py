"""Utility functions."""


def extract_text_content(content: object) -> str:
    """Extract text from an LLM response content field.

    Some models (e.g. gpt-oss on Groq) return structured blocks including
    thinking and text parts instead of a plain string.  This helper extracts
    only the text parts and joins them.

    Args:
        content: The ``response.content`` value, either a plain string or a
            list of dicts with ``type`` and ``text``/``thinking`` keys.

    Returns:
        The concatenated text content, stripped of whitespace.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        if parts:
            return "\n".join(parts).strip()
    return str(content).strip()
