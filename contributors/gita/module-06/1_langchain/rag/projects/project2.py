# import os
# from langchain_community.document_loaders import TextLoader, PyPDFLoader
# def load_folder(folder_path:str):
#     documents=[]
#     for filename in os.listdir(folder_path):
#         path=os.path.join(folder_path, filename)
#         if filename.endswith(".pdf"):
#             documents.extend(PyPDFLoader(path).load())
#         elif filename.endswith(".txt"):
#             documents.extend(TextLoader(path).load())
#     return documents


# load_folder("files")


import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"BASE_DIR: {BASE_DIR}")

def load_folder(folder_path: str):
    documents = []
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            documents.extend(PyPDFLoader(path).load())
        elif filename.endswith(".txt"):
            documents.extend(TextLoader(path).load())
    return documents

docs = load_folder(os.path.join(BASE_DIR, "files"))
print(f"Step 1 - Loaded {len(docs)} raw documents")


print("PRINTING DOCUMENTS.................")

# for i, doc in enumerate(docs):
#     if i>=86:
#         print(f"{i+1}- {doc}")
        

    
# for i, doc in enumerate(docs[86:], start=87):
#     print(f"{i}- {doc.metadata} | {doc.page_content[:200]}")

splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
chunks=splitter.split_documents(docs)
print(f"STage 2- SPlit: {len(chunks)} chunks")
print(f"Preview of chunk 0: {chunks[0].page_content}...")
print(f"Preview of chunk 1: {chunks[1].page_content}...")
