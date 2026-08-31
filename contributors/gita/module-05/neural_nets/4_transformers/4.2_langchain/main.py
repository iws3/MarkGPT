import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
load_dotenv() # reads ANTHROPIC_API_KEY from a local .env fil
claude = init_chat_model("google_genai:gemini-2.5-flash", temperature=0.7)
response=claude.invoke("who won 2026 Fifa world cup??")
# Swapping providers later only changes this one line:
# gpt = init_chat_model("openai:gpt-5", temperature=0.7)

print(response.content)