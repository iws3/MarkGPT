"""I2. A wrap_model_call hook that retries. Concept: handler() can be called more than once."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import AIMessage

from fakes import ScriptedModel, user

MAX_ATTEMPTS = 3


@wrap_model_call
def retry_model(request, handler):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return handler(request)
        except Exception as error:
            print(f"  [retry] attempt {attempt} failed: {error}")
            if attempt == MAX_ATTEMPTS:
                raise                              # out of attempts: let the error through


print("Model fails twice, then works:")
model = ScriptedModel(script=[AIMessage("Caption ready!")], fail_times=2)
agent = create_agent(model=model, tools=[], middleware=[retry_model])
print("  ->", agent.invoke(user("Write a caption"))["messages"][-1].content)

print("\nModel fails four times (more than the 3 attempts allowed):")
model = ScriptedModel(script=[AIMessage("never reached")], fail_times=4)
agent = create_agent(model=model, tools=[], middleware=[retry_model])
try:
    agent.invoke(user("Write a caption"))
except RuntimeError as error:
    print(f"  -> gave up and raised: {error}")