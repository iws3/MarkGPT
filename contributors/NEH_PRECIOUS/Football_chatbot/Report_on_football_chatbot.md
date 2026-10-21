# Football Chatbot

## Abstract

This project builds a football chatbot with LangChain and Streamlit. Users chat with the bot in a web page and can pick a persona, such as an encouraging tactician, a football medical doctor, or a football player. Replies stream in as they are generated, and the conversation history is kept so the bot remembers earlier messages.

## 1. Introduction

Chatbots feel more useful when they have a clear role and remember the conversation. This project combines three ideas: a chat interface built with Streamlit, a LangChain chain that takes the user's question, the history, and a persona, and streaming output so the answer appears word by word.

## 2. Components

The app is a single Streamlit page, 1_Chat.py, which uses a chat chain from core.chains.

- Page setup: sets the page title and icon and shows the header "Football Chat Bot".
- Persona selector: a sidebar dropdown with three choices: Encouraging tactician, Football medical doctor, and Football player.
- Chat chain: `get_chat_chain()` from core.chains builds the chain that answers the questions.
- History: the conversation is stored in `st.session_state` as HumanMessage and AIMessage objects, so it survives Streamlit reruns.
- Chat display: past messages are shown with `st.chat_message`, using a user or assistant role.
- Input: `st.chat_input` collects the user's question.
- Streaming reply: `chain.stream()` is passed the question, the earlier history, and the chosen persona, and `st.write_stream` shows the answer as it arrives, with a "thinking..." spinner while the bot works.

## 3. Workflow

The user picks a persona in the sidebar and types a football question. The app saves the question to the history and displays it. It then sends the question, the earlier messages, and the persona to the chain, which streams the answer back to the page. When the answer is complete, the app saves it to the history so the next question has the full context.

## 4. How to Create

1. Set up a Python project and virtual environment.
2. Install streamlit, langchain, and the packages for your model provider.
3. Create the chat chain in core.chains with `get_chat_chain()`, using a prompt that includes the persona, the history, and the user's input.
4. Create the Streamlit page and set the page title, icon, and header.
5. Add the persona dropdown to the sidebar.
6. Set up the history in `st.session_state`.
7. Display past messages with `st.chat_message`.
8. Add `st.chat_input` and save each question to the history.
9. Stream the chain's reply with `st.write_stream` and save it as an AIMessage.
10. Run the app with `streamlit run`.

## 5. Problems Faced

- Keeping the conversation history between Streamlit reruns.
- Passing the history, input, and persona to the chain with the right variable names.
- Showing the answer as it streams while still saving the full reply.
- Making the chosen persona affect the bot's answers.
- Understanding how HumanMessage and AIMessage represent the conversation.

## 6. Conclusion

This project shows how to build a streaming, persona-based chatbot with LangChain and Streamlit. It can be extended with more personas, football data tools, or document retrieval for match reports and statistics.

## 7. GitHub Repository

Project Repository:https://github.com/negprecious69-commits/Generative_Ai/blob/main/1.8_langchain/1_langchain/pages/1_Chat.py