import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import tempfile
import PyPDF2

st.set_page_config(page_title="Knowledge Base AI Agent", layout="wide")
st.title("📘 Company Knowledge Base – AI Assistant")

env_key = os.environ.get("OPENAI_API_KEY", "")
api_key = st.text_input("OpenAI API Key (or set OPENAI_API_KEY in Streamlit secrets):", type="password", value=env_key)

uploaded_files = st.file_uploader("Upload PDF or TXT files", accept_multiple_files=True)

if uploaded_files and api_key:
    temp_dir = tempfile.mkdtemp()
    raw_texts = []

    for uploaded in uploaded_files:
        file_path = temp_dir + "/" + uploaded.name
        with open(file_path, "wb") as f:
            f.write(uploaded.getbuffer())

        if uploaded.name.lower().endswith(".pdf"):
            try:
                with open(file_path, "rb") as fh:
                    reader = PyPDF2.PdfReader(fh)
                    text = []
                    for p in reader.pages:
                        text.append(p.extract_text() or "")
                    doc_text = "\n".join(text)
            except:
                doc_text = ""
        else:
            try:
                with open(file_path, "r", errors="ignore") as fh:
                    doc_text = fh.read()
            except:
                doc_text = ""

        if doc_text.strip():
            raw_texts.append(doc_text)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = []
    for t in raw_texts:
        chunks.extend(splitter.split_text(t))

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = Chroma.from_texts(chunks, embedding=embeddings)
    retriever = vectordb.as_retriever()
    llm = OpenAI(openai_api_key=api_key)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=True
    )

    st.success("Indexing complete! Ask your question below.")

    query = st.text_input("Ask a question about the uploaded documents:")
    if query:
        result = qa_chain({"query": query})
        st.write("### Answer")
        st.write(result["result"])
