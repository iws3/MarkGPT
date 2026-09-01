from langchain.chat_models import init_chat_model
import os
from config import DEFAULT_MODEL, DEFAULT_TEMPERATURE


from  dotenv import load_dotenv
load_dotenv()


# a reusable function [its going to be used from any file our porject, without having to create it from scratch again]

def create_model(model_name:str=DEFAULT_MODEL):
    # this function will just return the model as output
    return init_chat_model(model_name, temperature=DEFAULT_TEMPERATURE)
    
 
    
