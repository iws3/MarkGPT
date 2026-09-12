from .models import create_model

model=create_model()

response=model.invoke("Who won the Fifa world cup in 2026?")
print(response.content)