# example 5 right here...

"""B4. An after_agent hook. Concept: after_agent runs ONCE per invoke(), even when the
loop went around several times. after_model, by contrast, runs on every trip."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, after_agent, after_model
from langchain_core.messages import AIMessage, ToolMessage

# from fakes import ScriptedModel, tool_call, user
from middlewares.tools import get_reel_stats, get_trending_hashtags
from core.models import create_model

model=create_model()
after_model_runs = 0


@after_model
def count_trips(state: AgentState, runtime):
    global after_model_runs
    after_model_runs += 1
    return None

@after_agent
def run_summary(state:AgentState, runtime):
    messages=state["messages"]
    model_calls=sum(isinstance(m, AIMessage) for m in messages)
    tool_runs=sum(isinstance(m, ToolMessage) for m in messages)
    print(f"[after_agent] run finished: {model_calls} model replies, {tool_runs} tool result")
    return None



agent=create_agent(model=model, tools=[get_trending_hashtags], middleware=[count_trips, run_summary])
# print("models run alrready")
agent.invoke({"messages":[("user", "check the reel 101 and suggest hastags")]})
print(f"after_model ran {after_model_runs} time, after_agent ran once")