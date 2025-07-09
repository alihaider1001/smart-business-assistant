from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

def create_vectorstore(pages):
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-xl"  # You can also try instructor-large or instructor-base
    )
    return FAISS.from_documents(pages, embeddings)
