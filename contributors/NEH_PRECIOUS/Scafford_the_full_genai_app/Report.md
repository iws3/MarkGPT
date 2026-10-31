# Scaffold the Full GenAI App

## Abstract

This project builds the foundation for a GenAI application using LangChain and Streamlit. It sets up the full folder skeleton, a central configuration file, a reusable function that creates a language model from any provider, a minimal Streamlit app, and a small test script that confirms the model responds. Every later mini project in the guide builds on this structure.

## 1. Introduction

GenAI apps grow quickly: new models, new prompts, and new features keep getting added. Starting with a clean structure keeps settings in one place, makes the model code reusable, and keeps secrets out of version control.

This project focuses on three ideas:

- Keeping all settings in one config file.
- Creating models through one reusable function, so switching providers only means changing a model name string.
- Testing the setup early with a simple script.

## 2. Project Structure

The project root holds app.py, config.py, requirements.txt, .gitignore, and the .env file that stores the API keys. Inside it, the core folder contains models.py, where the model creation function lives, and the tests folder contains test_model.py, which checks that the model responds. Both core and tests include an empty __init__.py so they can be imported as Python packages.

## 3. Components

### 3.1 Configuration (config.py)

All shared settings live in this file:

- `DEFAULT_MODEL`: the default model, Google Gemini (`google_genai:gemini-3.6-flash`).
- `QWEN`: a Qwen model served through Groq (`groq:qwen/qwen3.6-27b`).
- `LLAMA`: a Llama Prompt Guard model served through Groq (`groq:meta-llama/llama-prompt-guard-2-22m`).
- `DEFAULT_TEMPERATURE`: 0.5, which controls how varied the model's answers are.
- `CHUNK_SIZE`: 800, the size of text chunks for later document splitting.
- `CHUNK_OVERLAP`: 100, the overlap between neighbouring chunks.

Model names use the format `provider:model-name`, which LangChain reads to pick the right provider.

### 3.2 Model Creation (core/models.py)

This file holds `create_model()`, a reusable function that any file in the project can import.

- It loads API keys from the `.env` file using `python-dotenv`.
- It takes an optional `model_name` and falls back to `DEFAULT_MODEL`.
- It calls LangChain's `init_chat_model` with the model name and `DEFAULT_TEMPERATURE`, and returns the model.

Because `init_chat_model` understands the `provider:model` format, the same function works for Gemini, Groq, and other providers.

### 3.3 Model Test (test_model.py)

A small script that checks the setup end to end:

- It imports `create_model` and the `QWEN` model name.
- `test_model(question, model_name)` creates the model and invokes it with a question.
- It asks a sample question ("what is the difference between a car and a bike") and prints the response.

### 3.4 Streamlit App (app.py)

A minimal app that imports Streamlit, sets the page configuration with `st.set_page_config()`, and shows a placeholder message: "genai_app scaffold ready".

### 3.5 Project Files

- `.gitignore`: excludes `.env`, `.streamlit/secrets.toml`, `__pycache__/`, and `venv/`, so secrets and generated files stay out of Git.
- `requirements.txt`: pins the packages the project needs.
- `.env`: stores the API keys. It is not committed.

## 4. Workflow

The settings in config.py are read by create_model(), which builds the language model. The test script then sends a question to that model with invoke() and prints the response it gets back.

## 5. How to Create

1. Create the project folder and a virtual environment.
2. Create every folder and empty file from the project structure.
3. Add an empty `__init__.py` to `core/` and `tests/`.
4. Write `config.py` with the model names and settings.
5. Write `core/models.py` with `create_model()`.
6. Create a `.env` file and add your API keys, for example the Google and Groq keys.
7. Write `.gitignore` so `.env` is never committed.
8. Write `requirements.txt` and install it with `pip install -r requirements.txt`. Since the project uses Gemini and Groq models, make sure `langchain-google-genai` and `langchain-groq` are installed too.
9. Write `app.py` with the placeholder message.
10. Run `test_model.py` from the project root to confirm the model responds.
11. Start the app with `streamlit run app.py`.

## 6. Problems Faced

- Keeping function and constant names consistent across files, such as `create_model` and `QWEN`.
- Importing `config` and `core.models` correctly, which only works when scripts are run from the project root.
- Understanding the `provider:model-name` format used by `init_chat_model`.
- Making sure the right API key is in `.env` for each provider.
- Keeping `.env` and other secrets out of version control.

## 7. Conclusion

This project sets up the base of the GenAI app: a clean folder structure, central settings, a reusable model function, and a test that confirms the model works. Later mini projects such as prompts, chains, document splitting, and retrieval can now be added without changing the setup.

## 8. GitHub Repository

Project Repository: https://github.com/negprecious69-commits/Generative_Ai/blob/main/1.8_langchain/1_langchain/core/models.py
Config : https://github.com/negprecious69-commits/Generative_Ai/blob/main/1.8_langchain/1_langchain/config.py