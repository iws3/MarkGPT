from utils.tools import get_module_deadline, count_students_in_module

from .models import create_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

model=create_model()
tools=[get_module_deadline, count_students_in_module]

llm=model.bind_tools(tools)


# ____________________

tools_by_name=[t.name for t in tools]

# print(tools_by_name)
messages=[HumanMessage("How many studnts are in the CNN module and when is it due?")]

ai_response=llm.invoke(messages)
print(f"The Ai response is: {ai_response}")
messages.append(ai_response)
print(f"The messages array is: {messages}")

for tool in ai_response.tool_calls:
    print("hello there")