from .models import create_model
from config import QWEN

model=create_model(QWEN)
# response=model.invoke("Who is the first lady of the united state of America")

# print(response.content)

# for chunk in model.stream("who is the GOAT OF FOOTBALL"):
#     print(chunk.text, flush=True)


message_batch=[
    "Why do they call black Americans Nigros",
    "Who was the first President Caemeroon",
    "Who started the first World war 2?"
]

responses=model.batch(message_batch)

for r in responses:
    print(r.text)
    