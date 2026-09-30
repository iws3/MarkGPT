## 1. Project Overview

This project focuses on the development of a tool-calling system using LangChain and Python to automate access to bootcamp information. The system integrates a Large Language Model (LLM) with Python tools that allow users to retrieve module deadlines, count enrolled students, check prerequisites, and access classroom schedules through natural-language questions.

The system uses LangChain to connect the language model to predefined Python functions. When a user asks a question, the model identifies the appropriate tool, requests its execution, and uses the returned information to generate a response.

This project demonstrates how Large Language Models can interact with external functions to retrieve structured information and perform specific tasks.

## 2. Project Objectives

The main objective of this project is to develop an intelligent assistant that simplifies access to bootcamp information through automated tool calling.
The specific objectives are:
To retrieve submission deadlines for different bootcamp modules.
To determine the number of students enrolled in each module.
To verify whether a student has completed the prerequisite required for a module.
To retrieve classroom schedules based on room identification.
To integrate Python functions with a language model using LangChain.
To automate the execution of tools based on user questions.

## 3. Technologies Used
The project uses Python as its primary programming language and LangChain as the framework for integrating tools with a language model. A language model is responsible for interpreting user questions and generating responses, while Python functions retrieve the information stored in the application.

The system also uses structured dictionaries to store the sample bootcamp data and LangChain message objects to manage communication between the user, model, and tools.

## 4. Description of the Tools
The project consists of four main tools, each designed to perform a specific task.
4.1 Module Deadline Tool
The module deadline tool retrieves the submission deadline for a specified bootcamp module. It supports the ANN, CNN, and LLM modules, with their respective deadlines stored in the application.
The tool enables users to obtain deadline information by asking questions in natural language, without having to search manually through records.

4.2 Student Enrolment Counting Tool
The student enrolment tool retrieves the number of students enrolled in a particular module.
The current project data indicates that ANN has 24 students, CNN has 21 students, and LLM has 34 students. The tool provides access to these figures when users request enrolment information.

4.3 Prerequisite Checking Tool
The prerequisite checking tool determines whether a completed module matches the prerequisite required for another module.
In the current implementation, CNN requires ANN, while LLM requires CNN. The tool compares the requested module with the completed module and returns a message indicating whether the prerequisite matches.
This functionality can help students understand the learning sequence defined for the bootcamp.

4.4 Classroom Schedule Tool
The classroom schedule tool retrieves information about sessions scheduled in a particular classroom.
The current project includes schedules for two rooms. Room A101 has ANN on Monday at 10:00 and CNN on Wednesday at 14:00. Room B202 has LLM on Tuesday at 09:00 and Python on Thursday at 13:00.
The tool allows users to retrieve classroom timetable information by providing a room identifier.

## 5. System Architecture and Workflow

The system is designed to allow communication between the user, the language model, and the Python tools.
User Request
The user submits a question in natural language about a bootcamp module, student enrolment, prerequisites, or classroom schedules.
Language Model Processing
The language model interprets the question and identifies the appropriate tool or tools needed to retrieve the requested information.
Tool Selection and Execution

The application identifies the requested Python function and executes it using the arguments provided by the model.
Tool Result
The selected tool retrieves the relevant information and returns the result to the application.
Final Response
The tool result is sent back to the language model, which uses the information to generate a natural-language response for the user.
This workflow enables the language model to access predefined information through Python functions rather than relying entirely on information generated from its own training.

## 6. Implementation Approach

The implementation is divided into two main parts: tool definition and tool orchestration.
The tool definition component contains the four functions responsible for retrieving bootcamp information. Each function is defined as a LangChain tool, allowing the language model to identify its purpose and the arguments it requires.
The orchestration component imports the tools, connects them to the language model, and manages the execution process. It maintains the conversation history, processes tool calls, stores tool results, and sends the updated conversation back to the model.
This separation makes the system easier to understand, maintain, and extend with additional tools in the future.

## 7. Applications of the System

The tool-calling system can be used to answer different types of bootcamp-related questions, including:
Retrieving the submission deadline for a specific module.
Checking the number of students enrolled in a module.
Verifying prerequisite relationships between modules.
Finding the sessions scheduled in a particular classroom.
Providing users with quick access to predefined bootcamp information.
For example, a user can ask about the deadline for CNN or the number of students enrolled in ANN. The system can select the relevant tool, retrieve the configured information, and provide a response.
For classroom schedules, the current implementation requires a room identifier.

## 8. Testing and Expected Results

Testing is necessary to verify that each tool returns the correct information and that the language model can invoke the tools appropriately.
The expected results based on the current project data include:
Test
Expected Result
Retrieve the CNN deadline
2026-09-17
Count ANN students
24 students
Check CNN prerequisite against ANN
Prerequisite matches
Check LLM prerequisite against CNN
Prerequisite matches
Retrieve the A101 schedule
Monday ANN and Wednesday CNN
Retrieve an unknown room

A message indicating no schedule was found
These are expected outcomes from the supplied implementation. Actual testing should be carried out in the project environment to confirm that the tools and model work together correctly.

## 9. Limitations

Although the system demonstrates the main principles of LangChain tool calling, it has some limitations.
Static data: The information is stored in Python dictionaries and must be manually updated.
Input consistency: Some tools require module names to match the stored keys exactly.
Limited prerequisite information: The current implementation defines prerequisites for only CNN and LLM.
Room-based timetable retrieval: The classroom schedule tool searches by room rather than directly by module.
Error handling: Additional handling is needed for invalid arguments, failed tool execution, and model errors.
No database integration: The current implementation does not retrieve information from a live database.
These limitations define areas that can be addressed as the project develops.

## 10. Future Improvements

Future development can extend the system by introducing a database to store bootcamp information and allow records to be updated without modifying the source code.
Additional improvements may include input validation, improved error handling, expanded prerequisite relationships, and the ability to retrieve timetables by module as well as by classroom.
The system could also be extended with a graphical user interface to make it more accessible to students and administrators. Automated testing and authentication could further support its reliability and use in a larger bootcamp environment.

## 11. Conclusion

The LangChain tool-calling system demonstrates the integration of Python functions with a Large Language Model to automate access to bootcamp information. By implementing tools for module deadlines, student enrolment, prerequisite checking, and classroom schedules, the project provides a foundation for a natural-language information assistant.
The system illustrates how a language model can identify relevant functions, request their execution, receive structured results, and generate understandable responses. Although the current implementation relies on static data, it provides a basis for future development into a more comprehensive bootcamp management assistant.

## 12. References           

Project source file: tools.py — Definitions of the bootcamp information tools.
Project source file: tool_call.py — Tool integration, model invocation, and execution workflow.
LangChain documentation — Official LangChain Documentation
.
## Github links
Tool creation link : https://github.com/ihimbrukumsemuedi-jpg/GenAI-2035/blob/master/1.8_langchain/utils/tools.py
Tool call link : https://github.com/ihimbrukumsemuedi-jpg/GenAI-2035/blob/master/1.8_langchain/core/tool_call.py