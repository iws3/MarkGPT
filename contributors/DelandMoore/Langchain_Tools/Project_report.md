# Tools Report: Development of a LangChain Tool-Driven Agent Workflow

## Abstract
This project explores the integration of LangChain tools with a large language model to enable dynamic, context-aware responses grounded in structured data. The goal was to implement tool-based reasoning so that the model could retrieve relevant information such as module deadlines, student counts, prerequisite requirements, and session mappings before generating a final answer. The study compares a direct tool-calling approach with a higher-level agent-based implementation, demonstrating how language models can be extended beyond static text generation to perform task-oriented and data-driven query resolution.



## 1. Introduction
Large language models are highly capable of generating natural language responses; however, they often lack access to real-time or domain-specific information unless they are connected to external tools or data sources. LangChain provides a framework for addressing this limitation by enabling models to invoke structured functions as tools. This allows the model to interact with predefined APIs or utility functions and incorporate the returned results into its final reasoning.

The purpose of this project was to design and test a LangChain workflow in which a model:
- interprets a user query,
- selects the appropriate tool,
- executes the tool with the relevant arguments,
- incorporates the tool output into a second model call,
- and produces a final response based on the retrieved results.

This project demonstrates both a manual tool-calling pattern and an agent-driven approach.



## 2. Methodology

### 2.1 Tool Creation
The project defines custom LangChain tools designed to retrieve module-related information. These tools provide structured access to data points such as:
- module deadlines,
- student counts,
- prerequisite requirements,
- and mapping between sessions and modules.

Each tool is decorated using LangChain’s tool mechanism, making the function callable by the model. The tool description is crucial because it informs the model about the purpose, domain, and expected input of the function.

### 2.2 Direct Tool-Calling Workflow
The file `core/tool_calls3.py` implements a direct interaction pattern. The workflow involves the following sequence:
1. A human query is passed to the model.
2. The model identifies the relevant tool(s).
3. The tool call is executed with specific arguments.
4. The returned data is wrapped as a `ToolMessage`.
5. The message history is passed back into the model.
6. The model generates the final answer using the tool output.

This approach illustrates how tool results can be incorporated into a conversational context to improve answer accuracy and relevance.

### 2.3 Agent-Based Workflow
The file `core/agents_exo.py` develops a higher-level agent implementation. In this version, a model is bound to a set of tools and configured with a system prompt. The agent then receives a question and determines whether tool invocation is necessary. This represents a more autonomous pattern in which the model acts as an orchestrator for task execution.

This method is especially useful when the system needs to handle multiple information sources or dynamic decision-making based on user intent.



## 3. System Design and Implementation
The project used a modular design, separating the LLM setup from the tool definitions. Tools were created in the utilities layer and then imported into the core workflow files. This separation promotes maintainability, readability, and extensibility.

The key components of the implementation include:
- tool definitions for data retrieval,
- model initialization,
- tool binding to the selected LLM,
- execution of tool calls,
- message chaining across model invocations,
- and final answer generation.

This design provides a structure that can be extended for more complex applications, including database queries, document retrieval, or external API calls.



## 4. Discussion
The project demonstrates the practical advantages of tool-augmented LLM systems. Traditional language models may provide plausible but unverified responses. By attaching tools, the system becomes more reliable because the final answer is grounded in real data. This is especially important in educational and operational contexts where factual accuracy is essential.

The direct tool-calling method is useful for controlled workflows and explicit reasoning steps, while the agent-based method is better suited to more autonomous interactions where the model chooses the required actions. Both approaches contribute to the broader objective of building dependable AI systems that can operate beyond pure text generation.

Furthermore, the use of tool descriptions is critical. Since the model relies on these descriptions to understand what each tool does, the quality of the explanation embedded in the tool function improves tool selection accuracy and final answer quality.



## 5. Challenges and Observations
Several practical considerations emerged during the implementation:
- Tool selection depends on good function naming and descriptions.
- The model must be guided with clear prompts to ensure the correct tool is called.
- Tool output must be formatted consistently to be useful in subsequent model reasoning.
- The workflow requires careful message construction to preserve the context of the previous tool calls and results.

These observations highlight the importance of designing prompt and tool interfaces thoughtfully when building agentic systems.



## 6. Conclusion
This project successfully implemented and evaluated a LangChain-based tool-driven workflow for educational domain queries. It illustrates how a language model can use external functions to retrieve structured information and produce accurate, grounded responses. The comparison between direct tool invocation and the agent abstraction shows that LangChain provides a flexible framework for building intelligent systems capable of external interaction and decision-making.

The work demonstrates foundational principles relevant to modern AI systems, especially in the areas of retrieval-augmented generation, agent orchestration, and tool-using language models.



## 7. GitHub Repository
GitHub Links for the exact files reviewed in this report:
- Direct tool-calling implementation: https://github.com/DelandMoore/Generative-AI---2026/blob/master/core/tool_calls3.py
- Agent-based implementation: https://github.com/DelandMoore/Generative-AI---2026/blob/master/core/agents_exo.py

Project repository: https://github.com/DelandMoore/Generative-AI---2026

