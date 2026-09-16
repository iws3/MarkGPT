import os
import streamlit as st
from core.agents import build_multimodal_agent

st.set_page_config(page_title="Multimodal Assistant", page_icon="🧩")

os.makedirs("uploads", exist_ok=True)

agent = build_multimodal_agent()

uploaded_image = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"])
image_path = None

if uploaded_image:
    image_path = f"uploads/{uploaded_image.name}"
    with open(image_path, "wb") as f:
        f.write(uploaded_image.read())
    st.image(uploaded_image, width=300)

prompt = st.chat_input("Ask, search, generate, or analyze")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            full_prompt = prompt
            if image_path:
                full_prompt += f" (An image is available at: {image_path})"

            with st.spinner("Thinking..."):
                result = agent.invoke({"messages": [{"role": "user", "content": full_prompt}]})

            st.markdown(result["messages"][-1].content)

        except Exception as e:
            st.error(f"Something went wrong: {e}")