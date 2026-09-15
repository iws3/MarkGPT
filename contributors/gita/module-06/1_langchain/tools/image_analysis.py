import base64
from langchain_core.messages import HumanMessage
from core.models import create_model

vision_model=create_model()

def analyze_image(image_path:str, question:str)->str:
    with open(image_path, "rb") as f:
        image_bytes=f.read()
    b64_image=base64.b64encode(image_bytes).decode('utf-8')


    message = HumanMessage(content=[
        {"type": "text", "text": question},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
])
    response = vision_model.invoke([message])