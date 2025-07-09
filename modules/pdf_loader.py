from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile

def load_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    raw_pages = loader.load()

    # 🔪 Split into smaller chunks (max 512 tokens ≈ 2000 chars safely)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # ~200–300 tokens
        chunk_overlap=200
    )
    return splitter.split_documents(raw_pages)
