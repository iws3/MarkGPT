from config import DEFAULT_MODEL, DEFAULT_TEMPERATURE
from langchain.chat_models import init_chat_model
import os

from  dotenv import load_dotenv
load_dotenv()


def get_model(model_name:str):
    return init_chat_model(model_name, temperature=DEFAULT_TEMPERATURE)