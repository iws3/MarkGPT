# core/voice.py
import os
import time
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

API_URL = "https://yarngpt.ai/api/v1/tts"
API_KEY = os.environ["YARNGPT_API_KEY"]

MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 3


@tool
def text_to_speech(text: str, voice: str = "Idera", response_format: str = "mp3") -> str:
    """Generate Nigerian-accented speech via the YarnGPT hosted API. Returns path to audio file."""
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "text": text,
        "voice": voice,
        "response_format": response_format,
    }

    last_error = None

    for attempt in range(1, MAX_RETRIES + 2):
        response = requests.post(API_URL, headers=headers, json=payload, stream=True, timeout=60)

        if response.status_code == 200:
            os.makedirs("generated_audio", exist_ok=True)
            path = f"generated_audio/output.{response_format}"
            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return path

        last_error = f"YarnGPT API error {response.status_code}: {response.text}"

        # 500s can be transient, worth a retry. 400s are your payload, retrying won't help.
        if response.status_code >= 500 and attempt <= MAX_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS)
            continue

        break

    raise RuntimeError(last_error)