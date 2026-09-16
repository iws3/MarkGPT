import logging

logger = logging.getLogger("multimodal_agent")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def extract_text(content) -> str:
    """Normalize LangChain message content into a plain display string.
    Handles both plain strings and Gemini's list-of-blocks format."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts).strip()

    return str(content)