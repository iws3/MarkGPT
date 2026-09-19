from langchain.chat_models import init_chat_model
import os
from config import DEFAULT_MODEL, DEFAULT_TEMPERATURE
from dotenv import load_dotenv

load_dotenv()
def create_model(model_name:str=DEFAULT_MODEL):
 return init_chat_model(model_name, temperature=DEFAULT_TEMPERATURE)