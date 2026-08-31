from config import LLAMA, GEMINI, DEFAULT_TEMPERATURE
from langchain.chat_models import init_chat_model

def llm(model:str, temperature:str=DEFAULT_TEMPERATURE):
    return init_chat_model(model=model, temperature=temperature)
