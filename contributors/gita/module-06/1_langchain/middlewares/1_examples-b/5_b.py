"""B5. @dynamic_prompt. Concept: the system prompt can be computed per model call."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.tools import create_model 
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
 
from fakes import ScriptedModel, user


@dynamic_prompt
def prompt_by_conversation_length(request):
    turns=len(request.messages)
    if turns <=1:
        return "You are a freindly reels assistant. This is a new creator, greet them and ask them  about their niche"
    return "you are a reels assistant, be brief and practical, the creator is already in conversation"


model=create_model()

agent=create_agent(model=model,  tools=[], middleware=[prompt_by_conversation_length], checkpointer=InMemorySaver)
config=["configurable":{"thread_id":"b5"}]
agent.invoke({"messages":[("user", "hello"), config]})


#  for n, seen in enumerate(model.s)
