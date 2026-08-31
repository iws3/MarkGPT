import os

from dotenv import load_dotenv

from langchain.chat_models import init_chat_model


load_dotenv()

llm=init_chat_model("groq:qwen/qwen3.8-27b", temperature=0.3)

print(llm.invoke("what is the capital of cameroon"))

