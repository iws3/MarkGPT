from config import DEFAULT_MODEL, LLAMA
from core.models import get_model

import os

from  dotenv import load_dotenv
load_dotenv()

def ping(model, question:str)->str:
    try:
        return model.invoke(question).content
    except Exception as e:
        print(f"the error that occured is: {e}")


question="What is the capital of cameroon?"

for name in [DEFAULT_MODEL, LLAMA]:
    print(name)
    model=get_model(name)
    result=ping(model, question)
    print(f"{name}: {result}")