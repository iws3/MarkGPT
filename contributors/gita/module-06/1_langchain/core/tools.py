from dotenv import load_dotenv
import os
load_dotenv()
from langchain_tavily import TavilySearch

web_search=TavilySearch(
    max_result=5,
    topic="general"
 )

result=web_search.invoke({"query":"LAtest news about europe top five leagues"})
print(result)