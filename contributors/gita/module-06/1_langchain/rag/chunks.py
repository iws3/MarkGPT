from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
pdf_loader = PyPDFLoader("doc.pdf")
pdf_docs = pdf_loader.load()
# # one Document per page, with page number in metadata
# text_loader = TextLoader("module_notes.txt")
# text_docs = text_loader.load()


splitter=RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ",  " ", ""]
)

chunks=splitter.split_documents(pdf_docs)
print(f"Loaded {len(pdf_docs)} pages from the PDF")

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk.page_content[:500]}...")