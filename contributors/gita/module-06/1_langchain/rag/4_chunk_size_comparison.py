# chunk_size_comparison.py 

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

docs=TextLoader("docs/my_document.txt").load()
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
query="How much is the FEE for the web dev Bootcamp? "

for chunk_size in [300, 800, 1500]:
    # calculate the overlap size
    overlap=int(chunk_size * 0.15)
    splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks=splitter.split_documents(docs)
    # avg_length=sum(len(c.page_content)  for c in chunks)
    len_content=[len(c.page_content) for c in chunks]
    print(f"Lengths for {chunk_size} are : {len_content}")
    average_length=sum(len_content)/len(len_content)
    print(f"Average length for {chunk_size} is : {average_length:.2f}")
    store=InMemoryVectorStore(embeddings)
    store.add_documents(chunks)

    top_result=store.similarity_search(query, k=1)[0]
    print(f"\n===Chunk size {chunk_size} (overlap {overlap})===")
    print(f"Total chunks: {len(chunks)}")
    print(f"Average chunk length: {average_length:.0f} characters")
    print(f"Top match for: '{query}:\n{top_result.page_content}...")



    