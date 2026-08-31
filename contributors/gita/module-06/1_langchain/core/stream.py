from .models import get_model
from config import DEFAULT_MODEL

model=get_model(DEFAULT_MODEL)


# for chunk in model.stream("what is the difference between machine learning and deeplearnng"):
#     print(chunk.text, end="|", flush=True)
    

full=None

for chunk in model.stream("who is JESUS CHRIST"):
    full= chunk if full is None else chunk + full
    print(full.text)
print(full.content_blocks) 
