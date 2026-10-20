"""I1. A wrap_tool_call hook that sometimes skips the tool.
Concept: a wrap hook decides whether handler() runs at all."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from fakes import ScriptedModel, tool_call, user

real_runs = 0


@tool
def get_reel_stats(reel_id: str) -> str:
    """Return stats for a reel."""
    global real_runs
    real_runs += 1
    return f"reel {reel_id}: 18200 views"


cache: dict[str, str] = {}


@wrap_tool_call
def cache_tool_results(request, handler):
    key = request.tool_call["name"] + json.dumps(request.tool_call["args"], sort_keys=True)
    if key in cache:
        print(f"  [cache] HIT  {key}")
        return ToolMessage(content=cache[key], tool_call_id=request.tool_call["id"])
    print(f"  [cache] MISS {key}")
    result = handler(request)                      # the real tool runs here
    if isinstance(result, ToolMessage):
        cache[key] = result.content
    return result


model = ScriptedModel(script=[
    tool_call("get_reel_stats", {"reel_id": "101"}, "c1"),
    tool_call("get_reel_stats", {"reel_id": "101"}, "c2"),   # the model asks again
    AIMessage("Reel 101 has 18200 views."),
])
agent = create_agent(model=model, tools=[get_reel_stats], middleware=[cache_tool_results])
agent.invoke(user("Check reel 101, then double check it"))
print(f"The model asked twice, the tool really ran {real_runs} time(s).")