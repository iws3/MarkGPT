from core.models import llm

def test_lmm(question:str, model:str):
    model.invoke(question)
    
question="How many world cup did pele won?"

answer=test_lmm(question, llm)

print(answer)