# Project
# Bootcamp Announcement Generator
## Abstract

This project builds a program that writes announcements automatically. The user supplies who the announcement is for, the tone it should have, and the details to include. LangChain then joins a prompt template, a language model and an output parser into one pipeline (a chain) that returns the finished announcement. In the example used, the program writes a formal announcement telling SEED AI Bootcamp students that the bootcamp starts on Friday. A second script then tries to convert the announcement into a structured record. These notes explain how each part was built, what it does, the problems found and what was learned.

## Introduction

Organisations send announcements all the time: start dates, deadlines, changes and reminders. Each one has to suit its audience and use the right tone. A message to students may be friendly, while a message about a deadline may need to be urgent. Writing every message from scratch is repetitive, and it is easy to forget details.

Instead of writing a new request to the model each time, a prompt template is written once. It contains the fixed instructions plus blank spaces (placeholders) for the parts that change: the audience, the tone and the details. Whenever a new announcement is needed, only the blanks are filled in.

LangChain lets separate steps be connected like pipes. The chain used here has three steps:
- The prompt template fills in the placeholders and builds the full message.
- The model reads the message and writes the announcement.
- The string parser takes the model's reply and returns only the plain text.

The tools used were:
- LangChain, which builds the prompt, the chain and the model connection.
- Pydantic, which defines the structure used in the structured-output attempt.
- A reusable model-creation function, which sets up the model with the default name and a temperature of 0.5 from the configuration file.
- An environment file, which holds the API key.

## Methodology

The project was built in three stages across four files.

### Stage 1: The first working version

1. The model and the string parser were created.
2. A prompt template was written with two messages. The system message gives the model its instructions: write an announcement for a given audience, in a given tone, with given details. The human message repeats the announcement details. The audience, tone and details are placeholders filled in later.
3. The prompt, the model and the parser were connected into one chain.
4. The chain was run with real values: the audience was "SEED AI Bootcamp students", the tone was "Formal", and the details were "Bootcamp starts this Friday". The program fills the template with these values, the model writes the announcement, the parser returns the text, and the text is printed.
5. The script also contains lines, currently switched off, that would ask the user to type the audience and details, and a note listing three tones: formal, friendly and urgent. This shows the intention that one template should serve many tones.

### Stage 2: Making the prompt and chain reusable

The template was moved into its own file so other scripts could use it. This version states the tone more directly, in the form "in a (tone) tone, give an announcement for (audience) with details (details)". A separate chain file then builds the full chain once. Any other file can now import the finished chain and use it without rebuilding it.

### Stage 3: The structured-output attempt

This script imports the reusable chain and runs it. It then defines a structure called Announcement with three fields (tone, audience and announcement details), wraps the model so that it answers in that structure, runs it, and prints the answer as a dictionary. The aim was to turn the plain announcement into data that could later be placed into an email, a notice or a website.

## Discussion

The central lesson is how little code is needed to build a flexible tool once the pieces are separated. The template holds the instructions, the model does the writing and the parser cleans up the answer. Changing the tone from formal to friendly or urgent only means changing one value, not rewriting the prompt.

Stage 1 puts everything in one file, which is fine for learning. Stage 2 separates the prompt from the chain. This is a common good practice: if the wording of the prompt needs to change, it is changed in one place and every script that uses it receives the update.

A plain announcement is only text. A structured announcement could be split into parts, such as a subject line, a body and a call to action, and then inserted into other documents. This is why Stage 3 was attempted, even though it needs corrections.

The template also sends the announcement details twice, once in the system message and once as the human message. This helps make sure the model sees the details, but it repeats information and could be simplified.

## Challenges

1. The tone and audience values are swapped in the structured-output script. The tone is given the value "SEED AI Bootcamp students" and the audience is given the value "Urgent", so the model is told to write in a tone called "SEED AI Bootcamp students" for an audience called "Urgent".
2. The literal text "(result)" is sent to the structured model instead of the announcement. The script passes the characters themselves, not the variable that holds the generated announcement, so the structured model never sees the real announcement.
3. Every field description in the structure is also the literal text "(result)". The descriptions are supposed to tell the model what each field contains, so this gives it no instructions.
4. The structure repeats the inputs. It has fields for tone, audience and details, which the user already supplied, and no field for the announcement itself. A more useful structure would hold a subject, a body and a call to action.
5. Two versions of the prompt exist, one using the word "tones" and the other using "tone". Using the wrong key causes a missing-variable error, so keeping a single version would avoid this.
6. The details, "Bootcamp starts this Friday", give no date, time or place. The model may invent these to complete the announcement. The prompt should tell the model to use only the facts it is given.
7. Some imports are unused or immediately replaced, and both scripts call the model as soon as they are imported.
8. There is no error handling, no tests and no saved output, because the announcements are only printed.

## Observations

What is confirmed and what is not:
- The explanation of the four files, the two prompt versions, the chain and the structure was verified by reading the source files directly.
- The scripts were not run for this report, so the announcements the model actually generated, and whether the structured version works, are not known. No outputs are claimed.
- The model name in the configuration file was not checked against the provider's list of models, so it is not confirmed to exist.

Observations about the design:
- The text-generation part (Stages 1 and 2) is complete in design and simple to understand.
- The structured part (Stage 3) needs correcting. The two main faults, the swapped values and the literal "(result)" text, would stop it from working as intended.
- Good names and descriptions matter. The "(result)" descriptions show how a small oversight can leave the model with no instructions.
- Moving shared code into separate prompt, chain and model files made the later scripts shorter.

Improvements suggested (not tested): redesign the structure so it holds the subject, the body and a call to action; pass the real generated announcement to the structured model; and run the three tones on the same details, comparing and recording the outputs.

## Conclusion

The Bootcamp Announcement Generator was built in three stages: a first chain that connects a prompt template, a model and a parser; a reusable version with the prompt and chain in their own files; and an attempt to produce a structured announcement. It achieves the main goal of producing an announcement for a chosen audience and tone with very little code, and it teaches the core LangChain pattern of building small parts and connecting them. The structured script has faults that were identified: the swapped values, the literal "(result)" text sent in place of the real announcement, and a structure that does not contain the announcement itself. After these are corrected and the outputs of a test run are recorded, the project will be complete and ready for assessment.

## GitHub Links
- Text generation: https://github.com/AkongnwiDarius/Generative_Ai/blob/master/1.7_langchain/core/bootcamp_generator.py
- Prompt template: https://github.com/AkongnwiDarius/Generative_Ai/blob/master/1.7_langchain/core/prompt_boot.py
- Reusable chain: https://github.com/AkongnwiDarius/Generative_Ai/blob/master/1.7_langchain/core/chain.py
- Structured version: https://github.com/AkongnwiDarius/Generative_Ai/blob/master/1.7_langchain/core/announcement.py