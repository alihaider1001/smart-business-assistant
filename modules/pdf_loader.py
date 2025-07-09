from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os

def load_pdf(uploaded_file):
    # Save uploaded Streamlit file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        # Load and split PDF
        loader = PyPDFLoader(tmp_path)
        raw_pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,     # ~200–300 tokens
            chunk_overlap=200    # Keeps some context
        )
        return splitter.split_documents(raw_pages)

    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
