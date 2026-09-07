# pages/7_Image_Generator.py
import streamlit as st
from core.images import generate_image

st.set_page_config(page_title="Image Generator", page_icon="🎨")
st.title("Image Generator")

prompt = st.text_input("Describe the image you want")

if st.button("Generate") and prompt:
    with st.spinner("Generating..."):
        try:
            path = generate_image(prompt)
            st.image(path, caption=prompt, use_container_width=True)
        except Exception as e:
            st.error("Image generation failed. Try again in a moment.")