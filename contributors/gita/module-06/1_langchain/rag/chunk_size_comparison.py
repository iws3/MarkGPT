# chunk_size_comparison.py

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

docs=TextLoader("docs/my_document.txt").load()


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