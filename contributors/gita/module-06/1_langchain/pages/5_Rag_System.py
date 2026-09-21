import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="RAG Ingestion", page_icon="📄")
st.title("RAG Ingestion")


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_uploaded_file(uploaded):
    """Streamlit uploads live in memory, so write to a temp file for the loaders."""
    suffix = os.path.splitext(uploaded.name)[1].lower()
    if suffix not in (".pdf", ".txt"):
        raise ValueError(f"Unsupported file type: {uploaded.name}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        path = tmp.name

    try:
        if suffix == ".pdf":
            docs = PyPDFLoader(path).load()
        else:
            docs = TextLoader(path, encoding="utf-8").load()
    finally:
        os.remove(path)

    for doc in docs:
        doc.metadata["source"] = uploaded.name
    return docs


# Sidebar settings
st.sidebar.header("Chunking")
chunk_size = st.sidebar.slider("Chunk size", 200, 2000, 1000, step=100)
chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 500, 200, step=50)

# Upload
files = st.file_uploader(
    "Upload documents", type=["pdf", "txt"], accept_multiple_files=True
)

if st.button("Ingest", type="primary", disabled=not files):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    all_chunks = []

    with st.spinner("Loading and splitting..."):
        for f in files:
            try:
                docs = load_uploaded_file(f)
                all_chunks.extend(splitter.split_documents(docs))
            except Exception as e:
                st.warning(f"Skipped {f.name}: {e}")

    if all_chunks:
        with st.spinner(f"Embedding {len(all_chunks)} chunks..."):
            store = InMemoryVectorStore(get_embeddings())
            store.add_documents(all_chunks)
        st.session_state.store = store
        st.session_state.chunks = all_chunks
        st.success(f"Ingested {len(files)} file(s) into {len(all_chunks)} chunks.")

# Results
if "chunks" in st.session_state:
    chunks = st.session_state.chunks
    col1, col2 = st.columns(2)
    col1.metric("Chunks", len(chunks))
    col2.metric("Avg chunk length", sum(len(c.page_content) for c in chunks) // len(chunks))

    with st.expander("Preview first 5 chunks"):
        for i, c in enumerate(chunks[:5], 1):
            st.caption(f"Chunk {i} | {c.metadata.get('source')} | page {c.metadata.get('page', '-')}")
            st.text(c.page_content)

    st.subheader("Test retrieval")
    query = st.text_input("Ask something about your documents")
    if query:
        results = st.session_state.store.similarity_search(query, k=3)
        for r in results:
            st.caption(f"{r.metadata.get('source')} | page {r.metadata.get('page', '-')}")
            st.write(r.page_content)
            st.divider()