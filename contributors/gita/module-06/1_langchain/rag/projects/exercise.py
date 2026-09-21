import numpy as np

def cosine_similarity(a:list[float], b:list[float])->float:
    a, b=np.array(a), np.array(b)
    dot_product=np.dot(a, b)
    magnitude_a=np.linalg.norm(a)
    magnitude_b=np.linalg.norm(b)
    return dot_product/(magnitude_a * magnitude_b)
    print(f"vector A: {a} , vector B: {b}")


result= cosine_similarity([1,0,1], [0, 1, 1])
print(result)
