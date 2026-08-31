# from core.models import get_model
# from config import DEFAULT_MODEL

from .models import get_model
from config import DEFAULT_MODEL
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
model=get_model(DEFAULT_MODEL)

# A message is a structured object with three parts: role (system/user/assistant/tool), content (text, images,
# files, etc.), and metadata (token usage, IDs).


# result=model.invoke("Who is the GOAT of football")

# print(result.content)


# messages=[
#     SystemMessage("you are helpfull assistant that translate English to French"),
#     HumanMessage("Who is the GOAT of football?")
    
# ]

# respnse=model.invoke(messages)
# print(respnse)

# model.invoke("Why do parrots talk?")

message=model.invoke([
    {"role":"system", "content":"you are a poetry expert"},
    {"role":"user", "content":"Write about Messi losing 2014 world cup"}
])

print(message)


