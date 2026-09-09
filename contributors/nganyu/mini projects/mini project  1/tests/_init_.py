from core.models import create_model
from config import QWEN

def test_model(question, model):
    model = create_model(model)
    return model.invoke(question)

question = "what is the difference between a car and a bike"
test_result = test_model(question, QWEN)
print(test_result.content)