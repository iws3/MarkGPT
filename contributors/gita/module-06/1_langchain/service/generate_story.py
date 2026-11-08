import base64
from fastapi import FastAPI, File, Form, UploadFile
from langchain_core.messages import HumanMessage
from core.models import create_model
from config import QWEN

app = FastAPI()

llm1 = create_model()
llm2 = create_model()

@app.post("/generate-story")
async def generate_story(file: UploadFile = File(...), prompt: str = Form(...)):
    image_bytes = await file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    media_type = file.content_type or "image/jpeg"
    image_url = f"data:{media_type};base64,{image_base64}"

    image_message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    )
    description = llm1.invoke([image_message]).content

    story_prompt = (
        f"Create a beautiful story based on this image description: {description} "
        f"and this prompt: {prompt}"
    )
    story = llm2.invoke(story_prompt).content

    return {
        "success": True,
        "description": description,
        "story": story,
        "filename": file.filename,
    }
