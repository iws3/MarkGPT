# we are going to be doing our tool call here:
# we need to import tools from utils

from utils.tools import get_module_deadline, count_students_in_module
from .models import create_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage



tools=[get_module_deadline, count_students_in_module]

llm=create_model().bind_tools(tools)
# print(llm)
tools_by_name={t.name: t for t in tools}
# 'get_module_deadline': StructuredTool(name='get_module_deadline', description='Look up the submission deadline for a name bootcamp module.', args_schema=<class 'langchain_core.utils.pydantic.get_module_deadline'>, func=<function get_module_deadline at 0x0000022E8DADE200>), 'count_students_in_module': StructuredTool(name='count_students_in_module', description='Look up how many students are enrolled in a named bootcamp module', args_schema=<class 'langchain_core.utils.pydantic.count_students_in_module'>, func=<function count_students_in_module at 0x0000022E91E50B80>)}

# we are going to be constructing a message list [HumanMessage, AiMessage, SystemMessage, ToolMessage]

messages=[HumanMessage("How many students are in the CNN module and when is it due")]
ai_response=llm.invoke(messages)
# print(f"Ai response is: {ai_response}")

#  tool_calls=[{'name': 'count_students_in_module', 'args': {'module_name': 'CNN module'}, 'id': '4f3b376f-6dcc-4e6c-a608-4ee91cbab3bd', 'type': 'tool_call'}, {'name': 'get_module_deadline', 'args': {'module_name': 'CNN module'}, 'id': 'f3c8d007-a1ef-487e-ab21-a55614008f74', 'type': 'tool_call'}] 

messages.append(ai_response)

# for us to be able to pass this tool info our llm, we need to run a loop:
for tool in ai_response.tool_calls:
    tool_fn=tools_by_name[tool["name"]]
    result=tool_fn.invoke(tool["args"])
    print(f"tool result is: {result}")
    messages.append(ToolMessage(content=result, tool_call_id=tool["id"]))

final=llm.invoke(messages)
print(final.content)



