import os
import streamlit as st
from modules.pdf_loader import load_pdf
from modules.vectorstore import create_vectorstore
from modules.llm_model import load_llm_pipeline
from modules.qa_chain import create_qa_chain

# Set Hugging Face Token (if using)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets.get("HF_TOKEN", "")

st.set_page_config(page_title="Smart Business Report Assistant", layout="centered")
st.title("📊 Smart Business Report Assistant")

uploaded_files = st.file_uploader(
    "📎 Upload one or more PDF reports/invoices",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("🔄 Processing PDFs..."):
        all_docs = []
        for file in uploaded_files:
            docs = load_pdf(file)
            all_docs.extend(docs)

        vectorstore = create_vectorstore(all_docs)
        llm = load_llm_pipeline()
        qa_chain = create_qa_chain(llm, vectorstore)

    st.success("✅ Ready! Ask your questions below.")
    query = st.text_input("❓ Ask a question about the uploaded PDF(s)")

    if query:
        with st.spinner("💬 Thinking..."):
            try:
                result = qa_chain.invoke({"query": query})
                answer = result.get("result", "❌ No answer found. Try a different question.")
            except Exception as e:
                answer = f"⚠️ Error: {str(e)}"

        st.markdown("### 💡 Answer")
        st.markdown(
            f"""
            <div style='background-color: #1e1e1e; padding: 12px; border-radius: 8px; color: white; font-size: 16px;'>
            {answer}
            </div>
            """,
            unsafe_allow_html=True,
        )
