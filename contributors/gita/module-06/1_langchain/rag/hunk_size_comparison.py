from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


docs=TextLoader("docs/my_document.txt").load()

# query = "What is the refund policy?"
query="How much does the Web Development Bootcamp cost?"

for chunk_size in [300, 800, 1500]:
    overlap = int(chunk_size * 0.15)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_documents(docs)
    avg_length = sum(len(c.page_content) for c in chunks) / len(chunks)
    store = InMemoryVectorStore(embeddings)
    store.add_documents(chunks)
    top_result = store.similarity_search(query, k=1)[0]
    print(f"\n=== Chunk size {chunk_size} (overlap {overlap}) ===")
    print(f"Total chunks: {len(chunks)}")
    print(f"Average chunk length: {avg_length:.0f} characters")
    print(f"Top match for '{query}':\n{top_result.page_content[:200]}...")