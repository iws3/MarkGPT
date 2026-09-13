import os
import requests
import urllib.parse

def generate_image(prompt: str) -> str:
    encoded = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    os.makedirs("generated_images", exist_ok=True)
    path = f"generated_images/{abs(hash(prompt))}.png"
    with open(path, "wb") as f:
        f.write(response.content)
    return path