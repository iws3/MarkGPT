import time

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from core.ingestion import load_and_split_folder


# Load and split documents
start = time.perf_counter()
chunks = load_and_split_folder("docs")

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create and persist vector store
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

elapsed = time.perf_counter() - start
print(f"Built and persisted index with {len(chunks)} chunks in {elapsed:.2f} seconds")