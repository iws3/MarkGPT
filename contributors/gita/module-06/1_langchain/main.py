# getting started with langchain [way to talk to LLMS as a developer and create your own custom apps]


# 1. how connect to LLMs using langchain
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv() # reads GOOGLE_API_KEY automatically from our .env file


model=init_chat_model("groq:qwen/qwen3.8-27b", temperature=0.7)

# what is temperature ? : a model parameter that controls the randomness of the model's output. A higher temperature (e.g., 0.8) makes the output more random, while a lower temperature (e.g., 0.2) makes it more focused and deterministic.

# .invoke() : BASICALLY HELPS PASS INPUTS TO A MODEL AND GET OUTPUTS BACK

# runnable()

response=model.invoke("who is JESUS CHRIST")
print(response.content)

