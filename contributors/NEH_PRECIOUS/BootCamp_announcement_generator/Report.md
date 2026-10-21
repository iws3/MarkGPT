# Bootcamp Announcement Generator with LangChain

## Abstract

This project uses LangChain to generate bootcamp announcements. A prompt template, a language model, and an output parser are combined into a chain that turns announcement details into a short message for a chosen audience. A Pydantic schema is also used to return the result as structured data.

## 1. Introduction

Bootcamps regularly need to share announcements such as schedule changes, deadlines, and new sessions. This project automates that by letting a language model write concise announcements from a few inputs: the audience, the tone, and the announcement details.

The project focuses on prompt templates, chains built with the pipe operator, and structured output.

## 2. Components

- Prompt : a ChatPromptTemplate with a system message that sets the assistant's role and a human message that carries the announcement details. Variables: audience, tone, announcement_details.
- Chain : combines the prompt, the model, and a StrOutputParser into announcement_chain.
- Structured output : invokes the chain with the bootcamp details and defines a Pydantic model so the announcement can be returned as structured data.

## 3. Workflow

Announcement Details → Prompt Template → Language Model → Output Parser → Final Announcement

For structured output:

Announcement Details → Language Model with Pydantic Schema → Structured Announcement

## 4. How to Create

1. Set up a Python project and virtual environment.
2. Install the required LangChain packages and model dependencies.
3. Create the model in a separate models file.
4. Write the announcement prompt template with its variables.
5. Build the chain with prompt | model | parser.
6. Invoke the chain with the audience, tone, and announcement details.
7. Define a Pydantic schema and use with_structured_output.
8. Test with different bootcamp announcements and audiences.

## 5. Problems Faced

- Understanding how the pipe operator connects prompt, model, and parser.
- Keeping variable names consistent between the prompt and the invoke call.
- Correct imports and file names across the core package.
- Understanding the difference between plain chain output and structured output.
- Writing a Pydantic schema the model can fill correctly.

## 6. Conclusion

This project demonstrates how LangChain can automate bootcamp announcements using a reusable prompt chain and structured output. It can be extended with more announcement types, different tones, or a simple interface for staff to enter details.

## 7. GitHub Repository

Project Repository: https://github.com/negprecious69-commits/Generative_Ai/blob/main/1.8_langchain/1_langchain/core/announcement_chain.py