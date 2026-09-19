import os
import streamlit as st

st.set_page_config(page_title="Multimodal Assistant", page_icon="🧩")

os.makedirs("uploads", exist_ok=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("image_path"):
            st.image(msg["image_path"], width=300)
        if msg.get("audio_path"):
            st.audio(msg["audio_path"])
        if msg.get("content"):
            st.markdown(msg["content"])

user_input = st.chat_input(
    "Ask, search, generate, or analyze",
    accept_file=True,
    file_type=["png", "jpg", "jpeg"],
)

if user_input:
    from core.agents import build_multimodal_agent
    from core.utils import extract_text, logger

    if "agent" not in st.session_state:
        st.session_state.agent = build_multimodal_agent()
    agent = st.session_state.agent

    prompt_text = user_input.text
    uploaded_file = user_input.files[0] if user_input.files else None

    image_path = None
    if uploaded_file:
        image_path = f"uploads/{uploaded_file.name}"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.read())

    st.session_state.messages.append({
        "role": "user",
        "content": prompt_text,
        "image_path": image_path,
    })
    with st.chat_message("user"):
        if image_path:
            st.image(image_path, width=300)
        st.markdown(prompt_text)

    full_prompt = prompt_text
    if image_path:
        full_prompt += f" (An image is available at: {image_path})"

    with st.chat_message("assistant"):
        try:
            logger.info(f"Invoking agent with prompt: {full_prompt!r}")

            with st.spinner("Thinking..."):
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": full_prompt}]},
                    config={"recursion_limit": 15},
                )

            logger.info(f"Agent finished. {len(result['messages'])} messages in trace.")

            gen_image_path = None
            gen_audio_path = None

            for m in result["messages"]:
                tool_name = getattr(m, "name", None)
                content = getattr(m, "content", None)
                if tool_name:
                    logger.info(f"Tool called: {tool_name} -> {str(content)[:120]}")
                if tool_name == "generate_image" and isinstance(content, str):
                    gen_image_path = content
                elif tool_name == "text_to_speech" and isinstance(content, str):
                    gen_audio_path = content

            final_text = extract_text(result["messages"][-1].content)
            logger.info(f"Final response: {final_text[:200]}")

            if gen_image_path:
                st.image(gen_image_path, width=300)
            if gen_audio_path:
                st.audio(gen_audio_path)
            st.markdown(final_text)

            st.session_state.messages.append({
                "role": "assistant",
                "content": final_text,
                "image_path": gen_image_path,
                "audio_path": gen_audio_path,
            })

        except Exception as e:
            logger.exception("Agent invocation failed")
            st.error(f"Something went wrong: {e}")