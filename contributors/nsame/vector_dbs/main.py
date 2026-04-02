from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


texts = [
    "The sky appears blue because of a phenomenon called Rayleigh scattering.",
    "Rayleigh scattering causes shorter wavelengths of light (blue) to scatter more than longer ones.",
    "Apples are fruits that grow on trees. Granny Smith is a popular green variety.",
    "Investment banks help companies raise capital through stocks and bonds.",
    "The financial bank of the river was eroding due to strong currents.",
    "Large Language Models like GPT can sometimes hallucinate facts if not given proper context.",
    "Vector databases store embeddings and enable semantic search using cosine similarity.",
    "RAG stands for Retrieval Augmented Generation. It improves LLM answers by adding relevant context."
]

# Convert raw text into LangChain Document objects
documents = [Document(page_content=text) for text in texts]

print("Step 1: Created", len(documents), "documents\n")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

print(f"Step 2: Split into {len(chunks)} chunks\n")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"   
)

print("Step 3: Vector database created and documents indexed!\n")

query = "Why does the sky look blue?"

results = vectorstore.similarity_search(query, k=3)  # Retrieve top 3 most relevant chunks

print(f"Query: '{query}'\n")
print("Top relevant chunks found:\n")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content}\n")

print("Similarity Search with Scores:")
docs_with_score = vectorstore.similarity_search_with_score(query, k=3)
for doc, score in docs_with_score:
    print(f"Score: {score:.4f} → {doc.page_content[:150]}...\n")


context = "\n\n".join([doc.page_content for doc in results])

print("Ready for RAG!")
print("You can now send this context + your question to an LLM like:")
print(f"""
Question: {query}

Context:
{context}

Answer based only on the context above. If you don't know, say "I don't know".
""")