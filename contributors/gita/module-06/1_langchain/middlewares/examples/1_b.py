"""A before_model hook that shows exactly what the model is about to receive

concept: before_model runs on EVERY trip aroun the loop  and can read and state

"""
import os, sys
from core.models import create_model
from .tools import get_reel_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# why the line above? because this file is in a subdir of the repo, and we want to import agent.py from the parent dir  



from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, before_model
from langchain_core.messages import AIMessage
model=create_model()
@before_model
def inspect_messages(state:AgentState, runtime):
    print("f\n the model is about to see {len(state['messages])} messages(s)")
    for message in state["messages"]:
        text=str(message.content) or f"(tool call: {message.tool_calls[0]['name']})"
        print(f"  {type(message).__name__:13} {text[:60]}")
    return None


agent=create_agent(model=model, tools=[get_reel_stats], middleware=[inspect_messages])

question=input("Ask your question to get the middleware going...")

result=agent.invoke(question)
print("\nFinal answer: ", result["messages"][-1].content)

