""" B2: THe same email, four PII strategies.
Concept: built in middleware is configuration"""
from core.models import create_model
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain_core.messages import AIMessage

message="MY email is maya@example.com, please write a caption."
model=create_model()

for strategy in ["redact", "mask", "hash", "block"]:
    agent=create_agent(model=model, tools=[], middleware=[PIIMiddleware("email", strategy=strategy)])
    try:
        aresult = agent.invoke({"messages": [("user", message)]})
        print(f"{strategy:7} -> model saw: {model.seen[0][-1].content}")
    except Exception as error:
       print(f"{strategy:7} -> {type(error).__name__}: the model was never called (calls={model.i})")
       

