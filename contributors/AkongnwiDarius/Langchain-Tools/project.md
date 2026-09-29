# LangChain Tool Calling Mini Project

## 1. Overview

This mini project demonstrates how an LLM can use custom tools to retrieve specific information about university courses.

The model receives a user's question, determines which tool is needed, calls the appropriate tool, and then generates a final response using the tool result.

## 2. Features

The project includes tools for:

* Checking a course submission deadline
* Checking the number of students in a course
* Checking course prerequisites
* Checking the room and schedule for a course

## 2. Technologies Used
### 2.1 Tools

The tools are defined in `utils/tool_ex.py` 

This file contains the custom tools used by the project:

* `module_deadline_lookup`
* `student_count_lookup`
* `prerequisite`
* `session_module_lookup`

The tools are created using LangChain's `@tool` decorator.

`tool_call_ex.py`.This file connects the tools to the LLM. The available tools are bound to the model, and the model can decide which tool to call based on the user's question.

The tool result is then added to the conversation before the LLM generates the final response.

## 3. How It Works

When a user asks a question, the LLM looks at the question and decides which tool can provide the information needed. The selected tool then looks up the information and sends the result back to the LLM. The LLM uses that information to give the user a final answer.

The tool-calling process is handled in `tool_call_ex.py`

- For example, a user can ask:

> "How many students are in the Statistics course and when is the deadline?"

The LLM can call the student-count and deadline tools, receive their results, and use those results to produce the final answer.

## 4. Purpose

The purpose of this project is to demonstrate the basic concept of **LLM tool calling with LangChain** and how an LLM can interact with external functions to obtain information before generating a response.

## 5. Conclusion

This project provided a practical understanding of how LLMs can work with custom tools using LangChain. By creating tools for retrieving course information and connecting them to an LLM, the project demonstrated how a model can identify the information needed, use the appropriate tool, and provide a useful response to the user. Overall, it helped build a basic understanding of tool calling and how it can be used to make LLM applications more useful and interactive.

## 6. Github Repositories

Tool Creation: https://github.com/AkongnwiDarius/Generative_Ai/blob/master/1.7_langchain/utils/tool_ex.py
Tool Calling and Execution: https://github.com/AkongnwiDarius/Generative_Ai/blob/master/1.7_langchain/core/tool_call_ex.py
Project Repository: https://github.com/AkongnwiDarius/Generative_Ai.git