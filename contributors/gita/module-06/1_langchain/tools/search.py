#SEARCH

import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()

web_search = TavilySearch(
    max_results=5,
    topic="general"
)


