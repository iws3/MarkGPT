# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.memory import InMemorySaver


from langchain.agents import create_agent
from utils.tools import get_module_deadline
from .models import create_model

checkpointer=InMemorySaver()
print(checkpointer)

model=create_model()

agent=create_agent(
    model=model,
    tools=[get_module_deadline],
    system_prompt="You are a helpful assistant for assisting in my assignmnets",
    checkpointer=checkpointer,
)

config={"configurable":{"thread_id":"student-42-session"}}
result1=agent.invoke({"messages":[{"role":"user", "content":"When is the CNN module due?"}]}, config)
print(result1)

result2=agent.invoke({"messages":[{"role":"user", "content":"And what about ANN?"}]}, config)

print(result2["messages"][-1].content)