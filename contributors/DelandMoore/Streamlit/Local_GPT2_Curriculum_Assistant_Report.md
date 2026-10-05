# Local GPT-2 Curriculum Assistant Report

## Abstract
This project is a Streamlit chat interface for a locally stored GPT-2 text-generation model. It accepts a student's question, generates a response, and displays the conversation during the current session. The source code does not include curriculum documents, retrieval, or evaluation results, so it does not establish factual accuracy or educational effectiveness.

## 1. Introduction
The app is presented as a curriculum assistant for questions about deadlines, notebooks, datasets, assignments, and course information. It provides a simple chat workflow backed by a locally loaded language model.

## 2. Methodology

### 2.1 Input and Conversation State
The interface collects free-text questions with Streamlit's chat input. User and assistant messages are stored in `st.session_state.messages`, rendered in the chat, and counted in the sidebar. The clear button empties this session state.

### 2.2 Text-Generation Model
The app loads a Hugging Face `text-generation` pipeline from `C:\Users\ATZ COMPUTERS\Desktop\my_local_gpt2`, using that directory for both the model and tokenizer. The pipeline is cached with `st.cache_resource`. Each submitted question is formatted as `Student: ...\nAssistant:` and generated text is shown as the assistant response.

## 3. Implementation
`CHBOT.py` builds the Streamlit interface, loads the model once per cached resource lifecycle, and manages the visible conversation in session state. The model-generation call receives only the current question; earlier turns are not included in its prompt.

## 4. Observations
The script is a compact demonstration of local text generation, but it does not connect to course materials or verify answers against a source. Despite the conversation display, follow-up questions are generated without prior chat context. The absolute Windows model path also ties deployment to a machine with that directory and compatible model files. No evaluation results, error handling for model-load or generation failures, or generation-configuration controls are included in the source.

## 5. Conclusion
The project demonstrates a basic Streamlit chat application using a local language model. To function as a dependable curriculum assistant, it would need access to authoritative course information, context-aware follow-up handling, and evaluation of answer quality. Its responses should be treated as generated suggestions, not verified course guidance.

## 6. Source Code
- Streamlit app: [CHBOT.py](CHBOT.py)
- GitHub repository: https://github.com/DelandMoore/Generative-AI---2026
- GitHub source file: https://github.com/DelandMoore/Generative-AI---2026/blob/master/Streamlit%20Bases/CHBOT.py
- Local model directory configured in the app: `C:\Users\ATZ COMPUTERS\Desktop\my_local_gpt2`
