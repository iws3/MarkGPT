# Project
# Deep Agents Exercise
## Abstract

This exercise studies "deep agents", a type of AI agent built for complex, multi-step tasks. A deep agent can plan its work as a to-do list, store information in a file workspace, and hand parts of the job to sub-agents. The notebook first explains these ideas and compares them with simpler "shallow" agents. It then builds a working deep agent that acts as a research assistant, giving it a web-search tool and a Google Gemini language model, and tests it with a question. These notes explain what was done at each stage, the problems found, and what was learned.

## Introduction

Many AI agents work in a simple loop: the model thinks, chooses a tool, calls it, looks at the result and repeats until it can answer. This is the pattern the notebook calls a shallow agent. It works well for short tasks, but it struggles with long jobs such as planning a trip or writing a research report, because every message is replayed into the model's memory on each turn and the context fills up with detail.

A deep agent adds extra abilities for long jobs. According to the notebook, a deep agent has four main parts: a planning tool that creates a to-do list, a file system that stores research results, sub-agents that take over delegated work, and a system prompt that tells the agent how to approach tasks. The library used is called deepagents, and the notebook describes it as built on top of LangGraph.

The tools and technologies used were:
- The deepagents library, which provides the deep agent.
- LangChain, which provides the model connection and tool support.
- Google's Gemini language model, accessed through LangChain.
- Tavily, a web-search service, used as the agent's tool.
- python-dotenv, which loads the API keys from a hidden environment file so they are not typed into the notebook.
- uv, the project manager that holds the project settings and the list of required packages.
- Jupyter notebook, where all the work was written and run.

## Methodology

The work was done in the order of the notebook, in a project folder with its own settings file, package list and notebook.

**Step 1: Set up the project.** The project was created with uv. Its settings file lists the packages needed: deepagents, langchain, langchain-groq, langchain-openai, python-dotenv, tavily-python and ipykernel. A separate requirements file lists the same tools in a simpler form.

**Step 2: Write the concept notes.** Five sections explain the ideas before any code is run.

1. Shallow agents versus deep agents. A shallow agent follows a think, act, observe loop: the user gives input, the model decides on an action, calls a tool, observes the result and returns an answer. A deep agent adds planning, a file system, sub-agents and a guiding system prompt.
2. How a deep agent breaks a task into a to-do list. The worked example is organising a ten-day research trip. The agent would create a to-do list in its state, work through it and update each item's status. The steps listed are: understand the research objectives, research possible destinations, identify research institutions, develop a ten-day itinerary, estimate the budget, and prepare the final research trip plan. Large results can be saved to files and used later for the final summary.
3. The file system. This is a workspace where the agent saves information while it works and reads it back later. Large results go into named files and the agent reads back only the file it needs, whereas in an ordinary chat every message is replayed into the context each turn.
4. Sub-agent delegation. A sub-agent works in its own context window instead of sharing the parent agent's full context. The notebook gives four reasons this matters: it reduces context overload, supports specialisation, limits irrelevant information and improves organisation.
5. State management. A simple chain follows a fixed sequence, step one, step two, step three, then output. This suits predictable processes. Complex agents must make decisions, call tools, delegate, return to earlier steps and remember what is finished, so a graph structure (a LangGraph state graph) with different nodes and connections suits them better.

**Step 3: Load the API keys.** The notebook reads the environment file with python-dotenv and sets the keys for Google, Groq and Tavily so the services can be used.

**Step 4: Build the web-search tool.** A small function wraps the Tavily search service. It takes a query, a number of results, a topic (general or news) and an option to include raw context, and it returns the search results. This function is what the agent will call when it needs information from the internet.

**Step 5: Create the language model.** The notebook creates a Gemini model through LangChain's model-creation function. The printed output shows the model as Gemini 3.6 Flash, with a release date of 21 July 2026, a context window of about one million input tokens, up to 65,536 output tokens, and support for tool calling, structured output and reasoning. Because the model object printed successfully, the model name was accepted by the service.

**Step 6: Create the deep agent.** The agent was created with the deep-agent function, given the model, the web-search tool and a short system prompt: "you are a research assistant". The notebook output shows the result as a compiled state graph, and a picture of the agent's graph is displayed under the cell.

**Step 7: Test the agent.** The agent was asked a simple question: "what is the fastest car on earth". The notebook printed the last message in the reply. The answer was long and organised into sections, comparing a jet-powered land-speed record vehicle, the fastest wheel-driven vehicle and several fast road-legal supercars, with a table of speeds. The text arrived in two blocks, with the table split across them.

**Step 8: Try streaming.** A final cell attempts to print the agent's reply word by word as it is produced. No output was saved for this cell.

## Discussion

The notebook shows the difference between knowing a concept and using it. The first half explains why long, complex tasks need planning, file storage and delegation. The second half builds the simplest possible deep agent: one model, one tool and one sentence of instructions. The fact that this agent took only a few lines to create shows how much the library does automatically, including setting up the planning and file-system tools that the notes describe.

The notes on context are the most useful part of the exercise. Storing big results in files and delegating work to sub-agents keeps the main agent's memory clear, so it can continue a long task without being overloaded. This is the main reason to prefer a deep agent over a shallow one for jobs such as research.

The test question was simple, so it does not show the deep-agent features. A question like the ten-day trip example would be a better test, because it would force the agent to create a to-do list, save research to files and possibly use sub-agents. The system prompt was also very short. A longer prompt that explains the agent's role, how to plan, when to search and how to present its answer would be expected to give more reliable results.

## Challenges

1. The planning, file-system and sub-agent features are explained in writing but are not demonstrated in the code. Only a simple question was asked, and the notebook output does not show whether the agent made a to-do list or called the search tool.
2. The streaming cell has no recorded output. As written, it sends a plain text question to the agent and reads a content value from each piece of the stream. Streams from agents normally return updates of the agent's state rather than message objects, so this cell may not work as written. This was not checked by running it.
3. The search function passes a setting named with a singular word (max result) while the parameter described in its own definition is called max results. The setting may be ignored by the search service. This was not verified.
4. The final answer is long and contains many facts, such as speeds and dates, that were not checked against reliable sources. The notebook does not show whether they came from the search tool or from the model's own knowledge.
5. Some small mistakes appear in the notes: spelling errors such as "fatest", "tackel" and "resaerch", and heading tags that are not closed properly.
6. The project settings file asks for Python 3.14 or later, but the notebook's saved information shows it was run with Python 3.13.9, so the environment may not match the project settings.
7. The readme file and the package folder inside the project are empty, so the notebook is the only real content.
8. The project folder contains an environment file with API keys. It is good practice to make sure it is never pushed to GitHub. This was not checked for this folder.

## Observations

What is confirmed and what is not:
- The concept notes, the project settings, the code structure and the saved outputs were verified by reading the notebook and project files directly. This includes the model details printed in the notebook and the fact that the agent returned an answer to the test question.
- The notebook was not re-run for this report, so everything about its behaviour comes from the outputs saved in the file.
- The facts in the agent's answer about cars were not verified.
- Whether the agent used the search tool, made a to-do list or used sub-agents is not shown by the saved output and is therefore not known.

Observations about the work:
- The notes are clear and well organised, and they correctly separate chains from graphs.
- Creating a deep agent needs very little code, which makes it easy to start.
- The exercise demonstrates the foundation only. The deeper features still need to be tested.
- A harder test question, such as the ten-day research trip, would show whether the planning and file features really work.

## Conclusion

The Deep Agents exercise explains what makes a deep agent different from a shallow one and then builds a working research assistant with a Gemini model and a web-search tool. It shows that a deep agent can be created with very little code, and the concept notes give a clear understanding of planning, file storage, sub-agents and state graphs. The work stops at a simple test, so the main strengths of deep agents were explained but not yet demonstrated. The next step is to test the agent on a long, multi-step task, record how it plans and uses files, fix the streaming cell and check the facts in its answers.

## GitHub Links

- Project folder: https://github.com/AkongnwiDarius/Generative_Ai/tree/master/DeepagentExercise
- Notebook: https://github.com/AkongnwiDarius/Generative_Ai/blob/master/DeepagentExercise/deep/deep%20agents.ipynb
