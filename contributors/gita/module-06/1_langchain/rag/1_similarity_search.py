import numpy as np

def cosine_similarity(a, b):
    a, b=np.array(a), np.array(b)
    return np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))


a=[1, 0, 2]
b=[2, 1, 0]

result=cosine_similarity(a, b)
print(result)