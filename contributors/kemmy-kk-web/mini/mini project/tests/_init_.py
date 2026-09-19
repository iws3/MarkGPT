from config import QWEN 
from core.model import create_model

def test_model(question,model_name):
    model=create_model()
    return model.invoke(question)



question="what is the difference between deeplearning and machinelearning?"

test_result=test_model(question, QWEN)
print(test_result.content[0]["text"])