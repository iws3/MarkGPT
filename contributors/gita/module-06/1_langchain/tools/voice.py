# core/voice.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://yarngpt.ai/api/v1/tts"
API_KEY = os.environ["YARNGPT_API_KEY"]

def text_to_speech(text: str, voice: str = "Idera", response_format: str = "mp3") -> str:
    """Generate Nigerian-accented speech via the YarnGPT hosted API. Returns path to audio file."""
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "text": text,
        "voice": voice,
        "response_format": response_format,
    }
    response = requests.post(API_URL, headers=headers, json=payload, stream=True, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(f"YarnGPT API error {response.status_code}: {response.text}")

    os.makedirs("generated_audio", exist_ok=True)
    path = f"generated_audio/output.{response_format}"
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return path