from langchain_community.document_loaders import PyPDFLoader, TextLoader
pdf_loader = PyPDFLoader("docs/docs.pdf")
pdf_docs = pdf_loader.load()
# one Document per page, with page number in metadata
text_loader = TextLoader("module_notes.txt")
text_docs = text_loader.load()
print(f"Loaded {len(pdf_docs)} pages from the PDF")
print(pdf_docs[0].metadata)
print(f"Printitng the pdf docs: {pdf_docs}")


# langchain_community
# pip install pypdf

