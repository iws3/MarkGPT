from langchain.agents import create_agent
from .tools import get_module_deadline, count_students_in_module
from .models import create_model

model=create_model()
agent=create_agent(
    model=model,
    tools=[get_module_deadline, count_students_in_module],
    system_prompt="You are an operational assistant for coding bootcamp, use tools to answer the factual question accurately"
)
print(agent)

messages={"messages":[{"role":"user", "content":"When is the CNN module due?"}]}
result=agent.invoke(messages)


print(result["messages"][-1].content)

# streaming agents output to the screen
for step in agent.stream({"messages":[{"role":"user", "content":"How many students are in the ANN"}]}):
    print(step)
    