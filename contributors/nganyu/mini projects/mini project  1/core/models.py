from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import os
from config import DEFAULT_MODEL, DEFAULT_TEMPERATURE

load_dotenv() 
def create_model(model:str=DEFAULT_MODEL):
    return init_chat_model(model, temperature=DEFAULT_TEMPERATURE)
    pass