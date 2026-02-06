import streamlit as st
from agent import agent
from rag import create_vector_store
import os
from datetime import datetime

st.set_page_config(page_title="Rag and web search agent", layout="centered")
st.title("Rag and web search agent")

os.makedirs("data", exist_ok=True)


uploaded_file = st.file_uploader(
    "Upload a document (txt, pdf, docx only for now)",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=False,
)

if uploaded_file is not None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join("data", f"{timestamp}_{uploaded_file.name}")

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"File uploaded successfully: {uploaded_file.name}")

    # Rebuild the FAISS index immediately using all .txt files in data/
    with st.spinner("Building / updating FAISS index from uploaded files..."):
        try:
            create_vector_store()
            st.success("FAISS index rebuilt. You can now ask questions about your documents.")
        except Exception as e:
            st.error(f"Failed to build FAISS index: {e}")

# Question input
query = st.text_input("Ask your question")

if st.button("Submit"):
    if query.strip():
        with st.spinner("Thinking..."):
            answer = agent(query)
        st.write(answer)
