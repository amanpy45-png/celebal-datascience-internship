# 📚 Document Question Answering System using Retrieval-Augmented Generation (RAG)

## 📌 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system that answers user questions based on the content of uploaded PDF documents.

Instead of relying solely on a Large Language Model's pre-trained knowledge, the system retrieves the most relevant information from the uploaded document and uses it as context to generate accurate, grounded, and context-aware answers.

The application is built using **LangChain, FAISS, HuggingFace Embeddings, ChatGroq (Llama 3.1), and Streamlit**, allowing users to upload **any PDF** and interact with it through natural language questions.

---

## 🎯 Objectives

- Build a Document Question Answering System using RAG.
- Enable question answering over custom PDF documents.
- Understand document retrieval using vector embeddings.
- Learn how retrieval and generation work together in modern AI applications.
- Develop a user-friendly interface using Streamlit.

---

## 🚀 Features

- 📄 Upload any PDF document
- ✂️ Automatic text chunking
- 🧠 Semantic embeddings using HuggingFace
- 📦 FAISS vector database for efficient similarity search
- 🔍 Retrieve the most relevant document chunks
- 🤖 Generate context-aware answers using Llama 3.1 (Groq)
- 📖 Display retrieved source passages with page numbers
- 🌐 Interactive Streamlit web interface

---

## 🛠️ Technologies Used

- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Sentence Transformers
- ChatGroq (Llama 3.1)
- PyPDF
- Python Dotenv

---

## 🏗️ Project Architecture

```
                User Uploads PDF
                        │
                        ▼
                 PyPDFLoader
                        │
                        ▼
        RecursiveCharacterTextSplitter
                        │
                        ▼
         HuggingFace Embeddings
                        │
                        ▼
                 FAISS Vector Store
                        │
                        ▼
                 User Question
                        │
                        ▼
          Similarity Search (Top-K)
                        │
                        ▼
         Retrieved Relevant Chunks
                        │
                        ▼
         ChatGroq (Llama 3.1-8B)
                        │
                        ▼
              Context-Aware Answer
                        │
                        ▼
       Display Answer + Source Chunks
```

---

## ⚙️ Workflow

1. Upload a PDF document.
2. Extract text from the document.
3. Split the text into overlapping chunks.
4. Generate embeddings for each chunk.
5. Store embeddings in a FAISS vector database.
6. Accept a user question.
7. Retrieve the most relevant chunks using semantic similarity.
8. Pass the retrieved context and question to the LLM.
9. Display the generated answer along with the supporting document sources.

---

## 📂 Project Structure

```
week7/
│
├── app.py
├── .env
├── requirements.txt
│
├── temp/
│
└── utils/
    ├── loader.py
    ├── splitter.py
    ├── embeddings.py
    ├── vectorstore.py
    └── rag.py
```

---

## 📊 Components

### Document Loader
Loads PDF documents using `PyPDFLoader`.

### Text Splitter
Splits large documents into overlapping chunks for efficient retrieval.

### Embedding Model
Uses **sentence-transformers/all-MiniLM-L6-v2** to generate semantic embeddings.

### Vector Database
Stores document embeddings in **FAISS** for fast similarity search.

### Retriever
Finds the top relevant chunks corresponding to the user's query.

### Large Language Model
Uses **Llama 3.1 8B Instant** through **Groq** to generate grounded answers.

---

## 💡 Example

### User Question

```
What is this document about?
```

### System Process

- Retrieves the most relevant document chunks.
- Uses the retrieved context.
- Generates a concise and accurate answer.
- Displays the supporting source passages.

---

## 📚 Key Learnings

Through this project, I learned:

- Retrieval-Augmented Generation (RAG)
- Document preprocessing
- PDF text extraction
- Text chunking strategies
- Semantic embeddings
- Vector databases using FAISS
- Similarity search
- Prompt engineering
- Context-aware question answering
- Streamlit application development
- Building modular AI applications using LangChain

---

## 🎯 Applications

- Enterprise Knowledge Assistants
- AI-powered Chatbots
- Document Search Systems
- Research Paper Q&A
- Resume Question Answering
- Legal Document Analysis
- Healthcare Documentation Assistants
- Educational Learning Assistants

---

## 📈 Future Improvements

- Support multiple PDFs simultaneously
- Conversation history
- Persistent FAISS index
- Hybrid search (Keyword + Vector Search)
- Citation highlighting
- Multi-file knowledge base
- Cloud deployment

---

## 📌 Conclusion

This project demonstrates how Retrieval-Augmented Generation (RAG) combines document retrieval and large language models to answer questions from custom documents.

By integrating document loading, semantic embeddings, vector search, retrieval, and language generation into a single pipeline, the system produces accurate and context-aware responses while reducing hallucinations.

This project provides practical experience in building modern AI-powered document question answering systems and highlights the importance of retrieval in enhancing the reliability of Large Language Models.