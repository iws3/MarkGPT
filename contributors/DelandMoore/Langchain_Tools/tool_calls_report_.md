# Tools Report: LangChain Tool-Calling Workflow

## Abstract
This project builds a LangChain workflow that allows a language model to answer bootcamp questions using four custom tools. The tools provide module deadlines, student counts, prerequisites, and room/session schedules.

## 1. Introduction
Language models may not know local course information. LangChain tools solve this problem by allowing the model to request structured data before producing an answer.

The workflow receives a question, selects the required tools, executes them, and sends the results back to the model for a final response.

## 2. Methodology

### 2.1 Tools
The tools are defined in `utils/tools3.py`:
- `get_module_deadline`
- `count_students_in_module`
- `prerequisite_counter`
- `session_module_lookup`

Each tool uses LangChain's `@tool` decorator and includes a description for the model. Unknown modules return `"Module not found"`.

### 2.2 Tool-Calling Workflow
`core/tool_calls3.py` binds the tools to the model with `bind_tools(tools)`. When the model requests a tool, the script invokes it and adds the result as a `ToolMessage`. The updated message history is then sent back to the model to generate the final answer.

The script tests three questions that use different tool combinations and one general question that should not require a tool. Empty `tool_calls` are handled without an error.

## 3. Implementation
The project separates tool definitions from model execution. A tool-name dictionary maps each model request to the correct function. This makes the workflow easy to read, test, and extend.

## 4. Observations
Clear tool names, docstrings, and type annotations improve tool selection. The prerequisite tool must accept a module name string, and the session tool must provide both a room and a schedule. A no-tool question also needs a separate execution path.

The files passed Python compilation, and the tools returned the expected lookup values in the `langchain` environment. Full model execution requires valid model-provider credentials.

## 5. Conclusion
The project demonstrates how LangChain connects a language model to structured application data. The completed workflow supports four tools, multiple tool calls, final answer generation, and graceful handling of questions that do not require tools.

## 6. GitHub Repository
- Direct tool-calling implementation: https://github.com/DelandMoore/Generative-AI---2026/blob/master/core/tool_calls3.py
- Tool definitions: https://github.com/DelandMoore/Generative-AI---2026/blob/master/utils/tools3.py
- Project repository: https://github.com/DelandMoore/Generative-AI---2026
