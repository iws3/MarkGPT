# DeepAgent Research Workflow and Streamlit Application Report

## Abstract

This project develops a DeepAgent research workflow that uses Tavily web search, a Groq-hosted language model, and Deep Agents middleware to investigate current topics. The workflow is exposed through a Streamlit chat application, allowing a user to submit a research question and observe the agent's progress as it searches for information and creates a concise summary. The project also examines practical issues in agent development, including token limits, tool-call validation, prompt design, repeated execution, and Streamlit import behavior.

## 1. Introduction

Large language models can produce useful explanations, but their internal knowledge may be incomplete or out of date. A research agent improves this process by connecting the model to external search tools and giving it a structured workflow for gathering and summarizing information.

The purpose of this task was to build a DeepAgent that can:

- receive a research question from a user,
- search the web for current information,
- use the returned evidence to prepare a cited summary,
- write or expose the generated result as a file,
- stream progress and responses through a user interface.

The implementation is contained in `core/DeepAgentTask.py` and `core/Deep_app.py`.

## 2. Methodology

### 2.1 Research Tool

`DeepAgentTask.py` defines the `research_search` tool with LangChain's `@tool` decorator. The tool accepts a query, a result limit, and a topic category. It uses `TavilySearch` to retrieve current web results and returns a bounded text payload to the agent.

The search output is intentionally limited to one result and 1,500 characters. This keeps the agent's context small and reduces both response time and token consumption.

### 2.2 DeepAgent Configuration

The agent is created with `create_deep_agent`. Its model can be selected using the `GROQ_MODEL` environment variable, which makes the implementation easier to adapt to different available models and service tiers.

The system prompt gives the agent a focused workflow: perform one search, write a short cited summary, and stop. The prompt avoids unnecessary planning or repeated research because the task is designed to be fast and predictable.

Unused built-in tools and the automatic general-purpose sub-agent are disabled. This reduces the tool schema sent to the model and helps prevent requests from exceeding the available input-token limit.

### 2.3 Streamed Execution

The agent is executed with `agent.stream(..., stream_mode="values")`. Each streamed state is inspected for todo items and messages. The application can therefore display progress while the research task is running instead of waiting silently for the complete workflow to finish.

The standalone execution is protected by an `if __name__ == "__main__"` guard. This allows other modules to import the configured agent without automatically starting a research task.

## 3. System Design and Implementation

The project separates agent configuration from the user interface. This keeps model initialization and tool definitions in `DeepAgentTask.py`, while Streamlit rendering and session handling remain in `Deep_app.py`.

### 3.1 DeepAgentTask.py

This module loads environment variables, configures the Tavily search tool, initializes the selected chat model, and creates the DeepAgent. It also contains a small `open_file` compatibility tool for model responses that request that tool name. The agent is available as a module-level `agent` object for the Streamlit application.

### 3.2 Deep_app.py

The Streamlit application imports the configured agent and caches it with `st.cache_resource`. Users enter questions through `st.chat_input`. The app stores previous messages in `st.session_state` and renders them as chat messages.

During execution, todo items are shown in a placeholder and the latest agent message is displayed as it arrives. Generated files are shown in expandable sections, allowing users to inspect the research output directly in the browser.

## 4. Discussion

The implementation demonstrates the value of combining tool use, an agent orchestration layer, and a graphical interface. Tavily provides current web information, while the DeepAgent decides when to use the research tool and how to produce a final summary. Streaming makes the process more transparent and gives the user feedback during longer operations.

The design also shows that agent reliability depends on more than model quality. Tool schemas, prompt length, output limits, middleware configuration, and import behavior all affect whether the workflow completes successfully. Careful control of these factors makes the application faster and more stable.

## 5. Challenges and Observations

Several challenges were encountered during development:

- **Groq input-token limits:** The original request exceeded the organization's 7,000-token input limit. The prompt, search output, and unused tool schemas were reduced to lower the request size.
- **Duplicate execution:** The initial script used both `agent.invoke()` and `agent.stream()`, causing the research task to run twice. The final workflow uses one streaming execution.
- **Invalid tool calls:** Some model responses attempted to call unavailable tools or supplied incorrect arguments. The tool list and prompt were made more explicit, and a compatibility `open_file` tool was added.
- **Truncated tool-call JSON:** A long generated summary was cut off during a `write_file` call. The summary is now limited to 150 words and the output budget was increased from 512 to 768 tokens.
- **Streamlit import behavior:** The application initially called `research_search()` without a query and also triggered standalone execution during import. Importing `agent` and adding the `__main__` guard resolved this problem.

These observations show that efficient agent development requires controlling both the model workflow and the surrounding application code.

## 6. Conclusion

This project successfully implements a DeepAgent research assistant with Tavily web search and a Streamlit chat interface. The agent can receive a question, gather external information, stream its progress, and present a concise research result.

The final implementation is designed for efficiency. It limits search output, uses a short task prompt, performs one focused search, removes unnecessary tools, and avoids duplicate execution. These changes make the workflow more suitable for environments with strict token limits and demonstrate important practical principles for building reliable tool-using AI systems.

## 7. GitHub Repository

GitHub links for the files reviewed in this report:

- Theory notes: https://github.com/DelandMoore/Generative-AI---2026/blob/master/1.8_langchain/core/theory.md
- DeepAgent workflow: https://github.com/DelandMoore/Generative-AI---2026/blob/master/1.8_langchain/core/DeepAgentTask.py
- Streamlit application: https://github.com/DelandMoore/Generative-AI---2026/blob/master/1.8_langchain/core/Deep_app.py

Project repository: https://github.com/DelandMoore/Generative-AI---2026

