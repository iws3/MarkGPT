import base64
from langchain_core.messages import HumanMessage
from core.models import create_model

from langchain_core.tools import tool

vision_model = create_model()
@tool
def analyze_image(image_path: str, question: str) -> str:

    """
    Analyze an image and answer a question about it.
   Use this when the user uploads an image and asks what's in it,
    or asks you to describe, identify, or extract information from an image.
   Args:
   image_path: Local file path to the image.
    question: What to determine about the image.
    
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    b64_image = base64.b64encode(image_bytes).decode('utf-8')

    message = HumanMessage(content=[
        {"type": "text", "text": question},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
    ])

    response = vision_model.invoke([message])
    return response.content