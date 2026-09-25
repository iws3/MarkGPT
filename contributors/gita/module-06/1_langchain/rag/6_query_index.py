# scripts/query_index.py
import time

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# Load persisted vector store
start = time.perf_counter()
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
reload_time = time.perf_counter() - start

# Perform similarity search
query_start = time.perf_counter()
results = vector_store.similarity_search("how much is the seed web dev bootcamp?", k=3)
query_time = time.perf_counter() - query_start

# Display results
print(f"Reloaded persisted index in {reload_time:.3f} seconds (no re-embedding of documents)")
print(f"Query took {query_time:.3f} seconds")
for doc in results:
    print(f"- {doc.page_content[:100]}...")