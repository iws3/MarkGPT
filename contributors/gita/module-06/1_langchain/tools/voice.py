from dotenv import load_dotenv
import requests
import os
from langchain_core.tools import tool

load_dotenv()

API_KEY=os.environ["YARNGPT_API_KEY"]
API_URL = "https://yarngpt.ai/api/v1/tts"
# WHY? BECUASE YARNGPT IS AN EXTERNAL SERVICE NOT GIVEN TO US BY LANGCHAIN

@tool
def text_to_speech(text:str, voice:str="Emma", response_format:str="mp3")->str:
    """Generate Nigerian-accented speech via the YarnGPT hosted API. Returns path to audio file."""
    headers={"Authorization":f"Bearer {API_KEY}"}
    payload={
        "text":text,
        "voice":voice,
        "response_format":response_format
    }
    response=requests.post(API_URL, headers=headers, json=payload, stream=True, timeout=60)
    if response.status_code!=200:
        raise RuntimeError(f"YarnGPT API error {response.text}")

    # if we have somewthing back
    os.makedirs("generated_audio", exist_ok=True)
    path=f"generated_audio/output.{response.format}"
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return path




