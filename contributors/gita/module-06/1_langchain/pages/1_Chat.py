import streamlit as st

st.set_page_config(page_title="Chat", page_icon="📈")

# streamlit run app.py --> 

if "history" not in st.session_state:
    # empty list is going to store the ai message and the human message
    st.session_state.history=[]
    

user_input=st.chat_input("Ask your question....")

if st.session_state.history or user_input:
    from langchain_core.messages import AIMessage, HumanMessage
    from core.chains import get_chat_chain

    chain=get_chat_chain()

    for msg in st.session_state.history:
        role="user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

if user_input:
    st.session_state.history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        full_reply=st.write_stream(
            chain.stream({"input":user_input, "history":st.session_state.history[:-1]})
        )
        st.session_state.history.append(AIMessage(content=full_reply))
        
