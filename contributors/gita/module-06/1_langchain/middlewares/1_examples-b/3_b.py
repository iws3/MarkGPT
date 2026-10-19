
"""B3. A before_agent hook that rejects bad input. Concept: before_agent runs once, and
a hook can end the run early with jump_to (declared with can_jump_to)."""
# missing

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, before_agent
from langchain_core.messages import AIMessage
from core.models import create_model

model=create_model()

MAX_LENGTH=200

@before_agent(can_jump_to=["end"])
def validate_input(state:AgentState, runtime):
    text=str(state["messages"][-1].content).strip()
    if not text:
        return {"messages":[AIMessage("Please type a message first")], "jump_to":"end"}
    if len(text) > MAX_LENGTH:
        return {"messages":[AIMessage(f"That is {len(text)} character. PLease keep it under {MAX_LENGTH}")], "jump_to":"end"}
    return None


for text in ["", "Write a caption for my dance reel", "x" *300]:
    agent=create_agent(model=model, tools=[], middleware=[validate_input])
    result=agent.invoke({"messages": [("user", text)]})
    print(f"input length {len(text):3} -> model calls={model} -> {result['messages'][-1].content}")




# input length  0 -> model calls=metadata={'lc_versions': {'langchain-core': '1.6.3', 'langchain': '1.4.0', 'langchain-google-genai': '4.4.0'}} profile={'name': 'Gemini 2.5 Flash', 'release_date': '2025-06-17', 'last_updated': '2025-06-17', 'open_weights': False, 'max_input_tokens': 1048576, 'max_output_tokens': 65536, 'text_inputs': True, 'image_inputs': True, 'audio_inputs': True, 'pdf_inputs': True, 'video_inputs': True, 'text_outputs': True, 'image_outputs': False, 'audio_outputs': False, 'video_outputs': False, 'reasoning_output': True, 'tool_calling': True, 'structured_output': True, 'attachment': True, 'temperature': True, 'image_url_inputs': True, 'image_tool_message': True, 'tool_choice': True} google_api_key=SecretStr('**********') model='gemini-2.5-flash' temperature=0.5 max_retries=1 timeout=30.0 client=<google.genai.client.Client object at 0x72866f741510> default_metadata=() model_kwargs={} -> Please type a message first
# Direct use of automatic function calling (AFC) in Models.generate_content is not recommended. Instead, we recommend to use AFC in Chat.send_message. Similarly, direct use of AFC in Models.generate_content_stream is not recommended. Instead, we recommend to use AFC in Chat.send_message_stream.
# input length  33 -> model calls=metadata={'lc_versions': {'langchain-core': '1.6.3', 'langchain': '1.4.0', 'langchain-google-genai': '4.4.0'}} profile={'name': 'Gemini 2.5 Flash', 'release_date': '2025-06-17', 'last_updated': '2025-06-17', 'open_weights': False, 'max_input_tokens': 1048576, 'max_output_tokens': 65536, 'text_inputs': True, 'image_inputs': True, 'audio_inputs': True, 'pdf_inputs': True, 'video_inputs': True, 'text_outputs': True, 'image_outputs': False, 'audio_outputs': False, 'video_outputs': False, 'reasoning_output': True, 'tool_calling': True, 'structured_output': True, 'attachment': True, 'temperature': True, 'image_url_inputs': True, 'image_tool_message': True, 'tool_choice': True} google_api_key=SecretStr('**********') model='gemini-2.5-flash' temperature=0.5 max_retries=1 timeout=30.0 client=<google.genai.client.Client object at 0x72866f741510> default_metadata=() model_kwargs={} -> Here are a few options for your dance reel caption, ranging from short and punchy to more descriptive, along with tips to customize them!

# **Short & Sweet:**

# 1.  "It's reel time! 💃✨"
# 2.  "My heart in motion. Enjoy!"
# 3.  "Just moving. Feeling the flow. 🎶"
# 4.  "Passion in every step. 🔥"
# 5.  "A little bit of what I love. ❤️"

# **Enthusiastic & Energetic:**

# 6.  "Finally dropping my latest dance reel! So much love for every moment captured here. Let the rhythm take over! 🕺💫"
# 7.  "Get ready to groove! My new reel is here, packed with passion, power, and pure joy. Which part is your favorite?"
# 8.  "Turning passion into motion! So excited to share this compilation of my recent work. Hope you feel the energy! ✨"
# 9.  "Hit play and let the good vibes roll! So thrilled to share my updated dance reel with you all. What are you waiting for?"
# 10. "Beyond excited to share my latest dance journey with you all! Every step, every beat, pure dedication. Enjoy the ride! 🚀"

# **Reflective & Artistic:**

# 11. "A glimpse into my world, where every step tells a story. So grateful for the journey and the art of movement. 🙏"
# 12. "Finding my voice through movement. This reel is a reflection of my growth, dedication, and endless love for dance. Hope it resonates with you."
# 13. "More than just steps, it's emotion, dedication, and pure joy. Hope you feel the vibe and the story within each frame. 💫"
# 14. "This reel represents countless hours, sweat, and pure heart. It's a privilege to share my passion with you. Thank you for watching."

**Professional / Audition Focused:**

# 15. "Presenting my updated dance reel! A compilation of recent work showcasing versatility, commitment, and artistry. Available for bookings/collaborations. 📧"
# 16. "For consideration: My latest dance reel. Excited for new opportunities and to bring my unique movement to your next project. #DancerForHire"
# 17. "Showcasing my range and passion in this new reel. Looking forward to connecting with fellow artists and industry professionals. Let's create! 🤝"



    
