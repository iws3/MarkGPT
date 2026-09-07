from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# [HumanMessage, AiMessage, ToolMessage, SystemMessage]

@tool
def get_module_deadline(module_name:str)->str:
    """Look up the submission deadline for a name bootcamp module."""
    deadlines={"ANN":"2026-09-01", "CNN":"2026-09-15", "RNN" :"2026-09-18"}
    return deadlines.get(module_name, "Module is unknown")


@tool

def count_students_in_module(module_name:str)->str:
    """Look up how many students are enrolled in a named bootcamp module"""
    counts={"ANN":25, "CNN":27, "RNN":78}
    return str(counts.get(module_name, 0))