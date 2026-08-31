from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from .models import get_model
from config import DEFAULT_MODEL
prompt=ChatPromptTemplate.from_messages([
    ("system", "you are a coincise teacher for an ai bootcamp. Answer in {max_sentence} sentences or fewer"),
    
    ("human", "{question}")
])
model=get_model(DEFAULT_MODEL)
# A prompt template itself is runnable
# we need to use a chain
class ActionItem(BaseModel):
    task:str=Field(description="What needs to be done")
    owner:str=Field(description="Who is responsible"),
    due_date:str=Field(description="Due date in formate: yyyy-mm-dd")
    

structured_model=model.with_structured_output(ActionItem)
result=structured_model.invoke("Gita will finish the model training this week saturday, the  team lead will  review on sunday")

print(result.model_dump())
# chain=prompt | model

# formmatted=chain.invoke({"max_sentence":4, "question":"What is L2 regularization in ml"})

# print(formmatted)