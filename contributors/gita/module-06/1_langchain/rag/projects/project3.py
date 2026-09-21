import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
sentences = [
"The invoice payment is overdue.",
"Payment for the bill is past due.",
"The cat is sleeping on the couch.",
"A feline is resting on the sofa.",
"Stock prices fell sharply today.",
"The weather forecast predicts rain.",
]


embeddings_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectors = embeddings_model.embed_documents(sentences)

# print(vectors)


def cosine_similarity(a, b):
    a, b=np.array(a), np.array(b)
    return np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))

print("Cosine Similarity Matrix:\n00")

header="      " + "".join(f"S{i}  "  for i in range(len(sentences)))

print(header)

for i, vec_i in enumerate(vectors):
    row=f"S{i}   " + "".join(f"{cosine_similarity(vec_i, vectors[j]):.2f} " for j in range(len(sentences)) )
    print(row)