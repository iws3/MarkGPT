import json
from utils.config import get_tools_needed
from gatherer import TOOLS

def route(question: str) -> list[str]:
    """
    Ask the model what tools are needed to answer this question.
    Returns a list of tool name strings.
    """
    response = get_tools_needed(question)
    try:
        return [t for t in response if t in TOOLS]
    except json.JSONDecodeError:
        return ["recent_commits", "directory_tree"]