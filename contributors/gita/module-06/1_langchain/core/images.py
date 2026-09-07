# core/images.py
import os
import requests
import urllib.parse

def generate_image(prompt: str, width: int = 1024, height: int = 1024) -> str:
    """Generate an image from a text prompt and save it locally.
    Returns the file path to the saved image.
    """
    encoded = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&nologo=true"

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    os.makedirs("generated_images", exist_ok=True)
    path = f"generated_images/{abs(hash(prompt))}.png"
    with open(path, "wb") as f:
        f.write(response.content)

    return path