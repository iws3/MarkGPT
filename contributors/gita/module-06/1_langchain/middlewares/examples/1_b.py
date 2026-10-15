"""A before_model hook that shows exactly what the model is about to receive

concept: before_model runs on EVERY trip aroun the loop  and can read and state

"""
import os, sys
from core.models import create_model
from middlewares.tools import get_reel_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# why the line above? because this file is in a subdir of the repo, and we want to import agent.py from the parent dir  



from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, before_model
from langchain_core.messages import AIMessage
model=create_model()
@before_model
def inspect_messages(state:AgentState, runtime):
    print(f"\n  The model is about to see {len(state['messages'])} message(s):")
    for message in state["messages"]:
        text=str(message.content) or f"(tool call: {message.tool_calls[0]['name']})"
        print(f"  {type(message).__name__:13} {text[:60]}")
    return None


agent=create_agent(model=model, tools=[get_reel_stats], middleware=[inspect_messages])

question=input("Ask your question to get the middleware going...")
result = agent.invoke({"messages": [("user", question)]})
print("\nFinal answer: ", result["messages"][-1].content)


# The model is about to see 1 message(s):
#   HumanMessage  who is the GOAT of football

# Final answer:  [{'type': 'text', 'text': 'I cannot answer that question. My purpose is to provide information about Instagram Reels.', 'extras': {'signature': 'CqcDAWkUfRP+UMbWY+U1so90VJqXQ9+HhgJ5GmgLAHsxYC+m/uPUvU+p/jaA2m6pfCcXb6FhwwOioq5uIUzwnf9aZrEvXYZi/8U5eSzNG08RHtitmkYIfxHSZT28TEzTy5S1cUjalspMTCPOwI76Z51aQgAjcp/qoPHg4jghrXiy3FjUXO6aOMN7HHbDQx1aJUJPRPKr/xOJ1AEyJjEeCpe6crlW61E4Vnf0nKJAZMJYfSnuIogit7yIBsBeBC4YBV62Byslghk32P+9/HXU+eFq4vEsiYbDEK2rv2B+1SwtVhiHxIkk30fJ6fg9N05pY4WPmPW6oGrJE+wnKhYNmBj+6ZSZm4iDvfhaQZhqelHl0Zi3DMjDtspPjCiVnk/TOWEM12q88LzaMe/sVzKrufpOSIOibsY7RNA2FV1xJT+zvKncKUQlhJ+7f7+3ISB5itHfpENkUACSO187nEKvRHYsrg5cPKk0sDYN+ae3VFumobSD/f/PXme2NXh951nsH0fmAZrgKYurtvlUYs1xe2sgSJvACwC4/LfrDxxQGBHW7nl+La8219Z7'}}]


# WHY IS THE MODEL NOT ANSWERING ABOVE



