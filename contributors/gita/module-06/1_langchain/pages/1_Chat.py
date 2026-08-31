import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from core.chains import get_chat_chain

st.set_page_config(page_title="SEEDCHAT", page_icon="🧑‍⚕️")

if "history" not in st.session_state:
    st.session_state.history = []

chain = get_chat_chain()

for msg in st.session_state.history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

user_input = st.chat_input("Ask something")

if user_input:
    st.session_state.history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        think_box = st.expander("💭 Thinking", expanded=True)
        think_placeholder = think_box.empty()

        def answer_only_stream():
            """
            Consumes the raw model stream, routes <think>...</think>
            content into the expander live, and yields only the
            post-</think> text for st.write_stream to render.
            """
            buffer = ""
            in_think = False
            seen_think = False
            think_text = ""

            raw_stream = chain.stream({
                "input": user_input,
                "history": st.session_state.history[:-1],
            })

            for chunk in raw_stream:
                buffer += chunk

                while True:
                    if not seen_think and not in_think:
                        idx = buffer.find("<think>")
                        if idx == -1:
                            break  # wait for more chunks — tag may be split across them
                        buffer = buffer[idx + len("<think>"):]
                        in_think = True
                        continue

                    if in_think:
                        idx = buffer.find("</think>")
                        if idx == -1:
                            think_text += buffer
                            think_placeholder.markdown(think_text)
                            buffer = ""
                            break
                        think_text += buffer[:idx]
                        think_placeholder.markdown(think_text)
                        buffer = buffer[idx + len("</think>"):]
                        in_think = False
                        seen_think = True
                        continue

                    # seen_think == True -> this is real answer text
                    if buffer:
                        yield buffer
                        buffer = ""
                    break

            # flush leftovers: covers (a) trailing answer text after loop ends,
            # and (b) models that never emit a <think> block at all
            if buffer and (seen_think or not in_think):
                yield buffer

        full_reply = st.write_stream(answer_only_stream())

    st.session_state.history.append(AIMessage(content=full_reply))