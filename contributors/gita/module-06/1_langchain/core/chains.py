import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from .models import get_model
from .prompt import chat_prompt
from config import QWEN

@st.cache_resource
def get_chat_chain():
    model=get_model(QWEN)
    return chat_prompt | model | StrOutputParser()