# get the model in and make sure it works
from .models import create_model
from config import QWEN
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model=create_model(QWEN)
parser=StrOutputParser()

prompt_template=ChatPromptTemplate.from_messages(
    [
        ("system", "please  in one sentence describe this {football_player} and give a brief history of his career, and also provide a list of his achievements in football."),
        ("human", "{question}")
    ]
)

football_player=input("Enter the name of the football player: ")
question=input("Enter the question you want to ask about the football player: ")


chain= prompt_template | model | parser
result=chain.invoke({
    "football_player": football_player,
    "question": question
})

print(result)

# prompt_template.invoke({
#     "football_player":"Lionel Messi",
#     "question":"What are the achievements of Lionel Messi in football?"
# })