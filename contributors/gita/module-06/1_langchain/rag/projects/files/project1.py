from core.models import create_model

model=create_model()


questions=[
"What is the late-submission penalty in our course syllabus?",
"Who is the instructor for the CNN module?",
"What is the passing grade threshold for the final project?",
"How many total contact hours does the bootcamp run for?",
"What happens if a student misses the midterm assessment?",
]


for q in questions:
    response=model.invoke(q)
    print(f"Q:{q}")
    print(f"A: {response.content}\n")
