from .models import create_model
from config import QWEN
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser

from .prompts import prompt


model = create_model(QWEN)
parser = StrOutputParser()


class Feedback(BaseModel): 
         sentiment:str=Field(description="overall sentiment of the submission ") 
         key_points:list[str]=Field(description="points in the feedback ")  
         sugested_Action:str=Field(description="actions and instructor should take base on the feedback ")
         original_text:str=Field(description="original feedback text ")  

