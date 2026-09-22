from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import streamlit as st
import os
import tempfile

st.set_page_config(page_title="RAG Ingestion", page_icon="📄")
st.title("RAG Ingestion...")

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# loading the uploaded file:

def load_uploaded_file(uploaded):
    suffix=os.path.splitext(uploaded.name)[1].lower()
    if suffix not in (".pdf", ".txt", ".md"):
        raise ValueError(f"Unsupportted file type: {uploaded.name}")
    # create a temporary file on disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        path=tmp.name

    try:
        if suffix==".pdf":
            docs=PyPDFLoader(path).load()
        else:
            docs=TextLoader(path).load()
    finally:
        os.remove(path)
    for doc in docs:
        doc.metadata["source"]=uploaded.name
        
    return docs

# THe sidebar sliders

chunk_size=st.sidebar.slider("CHunk size", 200, 2000, 1000, step=100)
chunk_overlap=st.sidebar.slider("Chunk overlap", 0, 500, 200, step=50)


# The file uploader

files=files=st.file_uploader(
    "Upload documents", type=["pdf", "txt"],
    accept_multiple_files=True
)

# part 6: ingest button

if st.button("Ingest", type="primary", disabled=not files):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    all_chunks=[]
    with st.spinner("splitting odcuments......"):
        for f in files:
            try:
                docs=load_uploaded_file(f)
                all_chunks.extend(splitter.split_documents(docs))
            except Exception as e:
                st.warning(f"Skipped {f.name}: {e}")
    if all_chunks:
        with st.spinner(F"Embedding {len(all_chunks)} chunks..."):
            store=InMemoryVectorStore(get_embeddings())
            store.add_documents(all_chunks)
            
        st.session_state.store=store
        st.session_state.chunks=all_chunks
        st.success(...)

if "chunks" in st.session_state:
    chunks = st.session_state.chunks
    col1, col2=st.colums(2)
    col1.metric("Chunks", len(chunks))
    col2.metric("Avg chunk length", sum(len(c.page_content) for c in chunks))

    with st.expander("Prevview the first 5 chunks"):
        for i, c in enumerate(chunks[:5], 1):
            st.caption(f"Chunk {i} | {c.metadata.get('source')} | page {c.metadata.get('page', '-')}")
            st.text(c.page_content)

