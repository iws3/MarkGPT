import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_documents(file_path:str):
    if file_path.endswith(".pdf"):
        return PyPDFLoader(file_path).load()
    elif file_path.endswith(".txt"):
        return TextLoader(file_path).load()
    raise ValueError(f"UNsupported file type: {file_path}")


def load_and_split_folder(folder_path:str)->list:
    splitter=RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_chunks=[]

    for filename in os.listdir(folder_path):
        docs=load_documents(os.path.join(folder_path, filename))
        all_chunks.extend(splitter.split_documents(docs))
        return all_chunks