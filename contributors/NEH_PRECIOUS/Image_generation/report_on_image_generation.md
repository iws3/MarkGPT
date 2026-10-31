# Image Generation Tool with LangChain

## Abstract

This project builds a LangChain tool that generates an image from a text description. The tool sends the description to the Pollinations image service, saves the returned image to a local folder, and returns the file path. It is designed so a language model can call it whenever a user asks to create, draw, or visualise something.

## 1. Introduction

Language models only produce text, so they need tools to do other things, such as creating images. LangChain tools are ordinary Python functions that a model can decide to call, based on the function's name, arguments, and docstring.

This project focuses on writing one such tool: a function that takes a prompt, requests an image, and gives back the saved file's path.

## 2. Components

The project has one file, image_gen.py, with a single function, `generate_image(prompt)`.

- The function is meant to be marked with LangChain's `@tool` decorator, so a model can discover and call it.
- Its docstring tells the model what the tool does and when to use it, which is how the model decides to call it.
- It takes one argument, `prompt`, a clear description of the image to generate.
- It requests the image from the Pollinations service with the `requests` library, using a 60-second timeout.
- It raises an error if the request fails, using `raise_for_status()`.
- It creates a folder called generate_image if it does not exist, then writes the image to a .png file named from the prompt.
- It returns the file path as a string.

## 3. Workflow

The user asks for an image, and the model decides to call the tool with a descriptive prompt. The tool sends the prompt to the image service, saves the image it receives into the generate_image folder, and returns the file path. The path can then be used to display the image or passed back to the model.

## 4. How to Create

1. Set up a Python project and virtual environment.
2. Install `langchain-core` and `requests`.
3. Create image_gen.py and import `tool`, `urllib.parse`, `requests`, and `os`.
4. Write `generate_image(prompt)` with a clear docstring.
5. URL-encode the prompt and build the Pollinations request URL.
6. Send the request and check the response for errors.
7. Create the output folder and save the image bytes to a .png file.
8. Return the file path.
9. Add the `@tool` decorator and test the function with a sample prompt.
10. Bind the tool to a model so it can call it.

## 5. Problems Faced

- Understanding how a plain Python function becomes a tool a model can call.
- Writing a docstring that clearly tells the model when to use the tool.
- Encoding the prompt correctly so spaces and special characters work in the URL.
- Handling slow or failed requests with a timeout and error check.
- Saving the response as an image file and returning its path.

## 6. Conclusion

This project demonstrates how to give a language model the ability to generate images by wrapping an image service in a LangChain tool. The same pattern can be extended with options such as image size, other image services, or a Streamlit interface that displays the result.

## 7. GitHub Repository

Project Repository: github.com/negprecious69-commits/Generative_Ai/blob/main/1.8_langchain/1_langchain/tools/image_gen.py