from langchain_community.vectorstores import FAISS

def create_vectorstore(chunks, embeddings):
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )
    print("Vector Store Created Successfully!")
    return vectorstore
