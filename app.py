import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

st.set_page_config(page_title="🧠 Study Coach Bot", layout="centered")

# Title and Description
st.title("🧠 Study Coach Bot")
st.write("Upload a PDF, ask any question, and get answers powered by LLMs and RAG!")

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload your PDF", type="pdf")

if uploaded_file:
    # Extract text
    reader = PdfReader(uploaded_file)
    raw_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    st.success("✅ PDF text extracted successfully.")

    # Chunk and clean
    def split_text(text, chunk_size=500):
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    chunks = split_text(raw_text)
    clean_chunks = [chunk.replace("\n", " ").strip() for chunk in chunks]

    # Embed
    st.info("🔍 Embedding text...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(clean_chunks)

    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    st.success("📚 FAISS index built!")

    # LLM
    llm = pipeline("text2text-generation", model="google/flan-t5-large")

    # Ask a question
    query = st.text_input("💬 Ask a question from the document:")
    if query:
        query_vec = embedder.encode([query])
        D, I = index.search(np.array(query_vec), k=3)
        context = " ".join([clean_chunks[i] for i in I[0]])

        prompt = f"""You are a helpful AI tutor. Based on the context below, answer the question.

        Context: {context}

        Question: {query}

        Answer:"""

        st.info("🤖 Generating answer...")
        response = llm(prompt, max_length=256, do_sample=False)[0]['generated_text']
        st.markdown("### ✅ Answer")
        st.write(response)
