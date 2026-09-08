from utils.tools import get_module_deadline, count_students_in_module

from .models import create_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

model=create_model()
tools=[get_module_deadline, count_students_in_module]

llm=model.bind_tools(tools)


# ____________________

tools_by_name={t.name: t for t in tools}
# see=tools_by_name["count_students_in_module"]

# StructuredTool(name='get_module_deadline', description='Look up the submission deadline for a name bootcamp module.', args_schema=<class 'langchain_core.utils.pydantic.get_module_deadline'>, func=<function get_module_deadline at 0x000001E48ACAE200>), 'count_students_in_module': is the value of t.name


print(tools_by_name)
messages=[HumanMessage("How many studnts are in the CNN module and when is it due?")]

ai_response=llm.invoke(messages)
print(f"The Ai response is: {ai_response}")
messages.append(ai_response)
print(f"The messages array is: {messages}")


for tool in ai_response.tool_calls:
    print(f"TOOL IS: {tool}")
    tool_fn=tools_by_name[tool["name"]]
    result=tool_fn.invoke(tool["args"])
    print(f"Result is: {result}")
    messages.append(ToolMessage(content=result, tool_call_id=tool["id"]))
    
print(f"The messages list is : {messages}")
messages.append(SystemMessage(content="Please make sure you extend the result with moe content of up to 70 words .. be explicit"))
final=model.invoke(messages)
print(final.content)