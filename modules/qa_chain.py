from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def create_qa_chain(llm, vectorstore):
    retriever = vectorstore.as_retriever()

    template = """
You are an AI assistant helping users analyze multiple PDFs (such as resumes, reports, invoices).
When answering questions, always speak from the user's perspective — say "your resume", not "my resume".

Be concise, polite, and answer in bullet points or short structured text.

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
