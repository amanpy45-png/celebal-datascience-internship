import os
import streamlit as st

from utils.loader import load_pdf
from utils.splitter import split_documents
from utils.embeddings import load_embedding_model
from utils.vectorstore import create_vectorstore
from utils.rag import create_rag_chain

st.set_page_config(
    page_title="Document Question Answering System",
    layout="wide"
)
st.title("Document Question Answering System (RAG)")
st.write("Upload any PDF and ask questions based on its content.")


uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)
if uploaded_file is not None:
    os.makedirs("data", exist_ok=True)
    pdf_path = os.path.join("data", uploaded_file.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("PDF Uploaded Successfully")
    

    with st.spinner("Processing PDF..."):
        documents = load_pdf(pdf_path)
        chunks = split_documents(documents)
        embeddings = load_embedding_model()
        vectorstore = create_vectorstore(
            chunks,
            embeddings
        )
        retriever = vectorstore.as_retriever(
            search_kwargs={"k":3}
        )
        rag_chain = create_rag_chain(
            retriever
        )
    st.success("RAG System Ready!")

    question = st.text_input(
        "Ask a Question"
    )
    if st.button("Get Answer"):
        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating Answer..."):
                result = rag_chain.invoke(
                    {
                        "input": question
                    }
                )
            st.subheader("Answer")
            st.write(result["answer"])
            st.subheader("Retrieved Sources")

            for i, doc in enumerate(result["context"], start=1):
                page = doc.metadata.get("page", "Unknown")
                with st.expander(
                    f"Source {i} | Page {page + 1 if page != 'Unknown' else page}"
                ):
                    st.write(doc.page_content)
