# Student Learning Feedback Report

## Abstract
This project is a small Streamlit form for collecting student learning feedback. It asks for a student's name, topics they understand, a module rating, and a short description of their learning journey, then displays a summary after submission. The source code does not save responses or provide aggregate analysis.

## 1. Introduction
Student feedback can help instructors understand which topics are clear and how students experience a module. This app provides a simple interface for gathering an individual student's self-reported understanding and feedback.

## 2. Methodology

### 2.1 Feedback Inputs
The form collects a name, a multiselect choice from Deep Learning, Machine Learning, and LLMs, a rating from 1 to 5 (defaulting to 2), and free-text feedback about the student's journey.

### 2.2 Submission Workflow
The fields are grouped in a Streamlit form. When the student selects the submit button, the app renders a sentence summarizing the submitted name, selected topics, rating, and feedback.

## 3. Implementation
`mini_project.py` uses Streamlit's `st.form`, `st.text_input`, `st.multiselect`, `st.slider`, and `st.text_area` widgets. The submitted values are interpolated into a Markdown string for display.

## 4. Observations
The app is a straightforward feedback-form prototype. Responses are displayed in the current app interaction but are not written to a file, database, or external service, so the code does not provide durable collection or reporting across students. The form also does not mark any fields as required or validate the feedback. The topic prompt contains a spelling error (`understnding`), and the generated summary could be presented more clearly as a formatted result.

## 5. Conclusion
The project demonstrates a basic Streamlit workflow for collecting and displaying student feedback. To support ongoing course evaluation, it would need a persistent, appropriately protected response store and a way to review responses across submissions. As written, it is best suited to a simple interactive demonstration.

## 6. Source Code
- Streamlit app: [mini_project.py](mini_project.py)
- GitHub repository: https://github.com/DelandMoore/Generative-AI---2026
- GitHub source file: https://github.com/DelandMoore/Generative-AI---2026/blob/master/Streamlit%20Bases/mini_project.py
