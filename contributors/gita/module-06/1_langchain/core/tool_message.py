from langchain_core.messages import HumanMessage, ToolMessage

from .models import create_model
from .tools import get_module_deadline

model=create_model().bind_tools([get_module_deadline])

messages=[HumanMessage("PLease with information and eplaination of what CNNS, ARE , give me the deadline to submit my work on CNNS")]

ai_response=model.invoke(messages)

print(ai_response)

messages.append(ai_response)


for call in ai_response.tool_calls:
    result=get_module_deadline.invoke(call["args"])
    messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

final=model.invoke(messages)
print(final.content)