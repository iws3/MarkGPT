
"""B3. A before_agent hook that rejects bad input. Concept: before_agent runs once, and
a hook can end the run early with jump_to (declared with can_jump_to)."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, before_agent
from langchain_core.messages import AIMessage
from core.models import create_model

model=create_model()

MAX_LENGTH=200

@before_agent(can_jump_to=["end"])
def validate_input(state:AgentState, runtime):
    text=str(state["messages"][-1].content).strip()
    if not text:
        return {"messages":[AIMessage("Please type a message first")], "jump_to":"end"}
    if len(text) > MAX_LENGTH:
        return {"messages":[AIMessage(f"That is {len(text)} character. PLease keep it under {MAX_LENGTH}")], "jump_to":"end"}
    return None


for text in ["", "Write a caption for my dance reel", "x" *300]:
    agent=create_agent(model=model, tools=[], middleware=[validate_input])
    result=agent.invoke({"messages": [("user", text)]})
    print(f"INput lenght {len(text):3}->Model calls={model}->{result["messages"][-1].content}")

    
