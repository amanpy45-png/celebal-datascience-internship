import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def create_rag_chain(retriever):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_template("""
You are an AI assistant for document question answering.

Answer ONLY from the provided context.

If the answer cannot be found in the context, reply exactly:

"I couldn't find this information in the provided document."

Keep the answer concise and factual.

Context:
{context}

Question:
{input}

Answer:
""")

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    rag_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return rag_chain
