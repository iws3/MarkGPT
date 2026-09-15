import os
import requests
import urllib.parse
from langchain_core.tools import tool

@tool
def generate_image(prompt: str) -> str:
    """
    Generate an image from a text description and return the file path.
     Use this when the user asks you to create, draw, or visualize something.
     Args:
     prompt: A clear, descriptive prompt of the image to generate.
    """
    encoded = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    os.makedirs("generated_images", exist_ok=True)
    path = f"generated_images/{abs(hash(prompt))}.png"
    with open(path, "wb") as f:
        f.write(response.content)
    return path