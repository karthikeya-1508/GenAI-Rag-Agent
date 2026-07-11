# 📄 GenAI RAG Agent - Document Question Answering System

A Retrieval-Augmented Generation (RAG) based Question Answering system that allows users to upload documents and ask natural language questions. The application retrieves the most relevant document chunks using semantic search and generates context-aware answers using Google's Gemini Large Language Model.

The project demonstrates the complete RAG pipeline from data ingestion to deployment using Streamlit.

---

# 🚀 Project Overview

Traditional LLMs generate responses based only on their pre-trained knowledge and cannot access custom documents.

This project solves that problem by implementing a **Retrieval-Augmented Generation (RAG)** pipeline, where:

- Documents are ingested and processed
- Text is split into meaningful chunks
- Chunks are converted into embeddings
- Embeddings are stored in Pinecone Vector Database
- User queries are embedded
- Relevant document chunks are retrieved
- Retrieved context is passed to Gemini LLM
- Gemini generates accurate, context-aware answers

---

# 🎯 Problem Statement

Large Language Models cannot directly answer questions from private or custom documents.

The objective of this project is to build an intelligent Question Answering system capable of:

- Understanding uploaded documents
- Retrieving only relevant information
- Reducing hallucinations
- Producing accurate answers using retrieved context

---

# ⚙️ RAG Pipeline

```
                PDF / DOCX Documents
                        │
                        ▼
              Data Ingestion
                        │
                        ▼
              Text Extraction
                        │
                        ▼
              Text Chunking
                        │
                        ▼
        HuggingFace Embeddings
                        │
                        ▼
          Pinecone Vector Database
                        │
                        ▼
              User Question
                        │
                        ▼
          Query Embedding
                        │
                        ▼
        Similarity Search (Top-K)
                        │
                        ▼
          Retrieved Context
                        │
                        ▼
             Gemini LLM
                        │
                        ▼
              Final Answer
```

---

# 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Frontend | Streamlit |
| LLM | Google Gemini |
| Embedding Model | HuggingFace Sentence Transformers |
| Vector Database | Pinecone |
| Framework | LangChain |
| Document Processing | PyPDF, Docx2txt |
| Environment Variables | python-dotenv |

---

# 📁 Repository Structure

```
GenAI-Rag-Agent/
│
├── .gitignore
│
├── PDF_QA_Bot_POC.docx
│      Project documentation
│
├── RAG_Overview.md
│      Theoretical explanation of the RAG pipeline
│
├── RAG_based_Q&A_Bot.ipynb
│      Development notebook containing complete implementation,
│      experimentation, debugging, and testing of the RAG pipeline
│
├── rag_pipeline.py
│      Production-ready implementation of the complete RAG pipeline
│
├── streamlit_app.py
│      Streamlit frontend application
│
├── requirements.txt
│      Project dependencies
│
└── README.md
```

---

# 📌 Project Workflow

### Step 1 — Data Ingestion

- Load PDF and DOCX documents
- Extract document text

---

### Step 2 — Text Chunking

- Split documents into manageable chunks
- Preserve contextual information using overlap

---

### Step 3 — Embedding Generation

Generate dense vector embeddings using:

- HuggingFace Sentence Transformers

---

### Step 4 — Vector Storage

Store document embeddings inside:

- Pinecone Vector Database

---

### Step 5 — User Query Processing

- Convert user question into embedding
- Perform semantic similarity search

---

### Step 6 — Context Retrieval

Retrieve the Top-K most relevant document chunks.

---

### Step 7 — Response Generation

Pass retrieved context and user query to Google's Gemini LLM to generate the final answer.

---

# 📂 About the Files

### 📒 RAG_based_Q&A_Bot.ipynb

This notebook contains the complete development workflow including:

- Data ingestion
- Document preprocessing
- Chunk creation
- Embedding generation
- Pinecone integration
- Retrieval logic
- Prompt engineering
- Response generation
- Debugging and experimentation

---

### 🐍 rag_pipeline.py

A production-ready implementation of the RAG pipeline with modular and reusable code for deployment.

---

### 🌐 streamlit_app.py

Provides an interactive web interface where users can:

- Upload documents
- Ask questions
- Receive AI-generated answers

---

### 📘 RAG_Overview.md

Contains theoretical concepts including:

- What is RAG?
- RAG architecture
- Pipeline explanation
- Retrieval process
- Benefits of RAG

---



# 💡 Features

- Document Question Answering
- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Pinecone Vector Database
- HuggingFace Embeddings
- Google Gemini Integration
- Prompt Engineering
- Modular Production Code
- Streamlit User Interface

---

# 📈 Future Improvements

- Support multiple document uploads
- Conversation memory
- Citation-aware responses
- Hybrid search (Dense + Sparse Retrieval)
- Reranking models
- Evaluation metrics for RAG
- Docker deployment
- Cloud deployment (AWS/Azure/GCP)

---

# 📚 Learning Outcomes

Through this project, I gained hands-on experience with:

- Retrieval-Augmented Generation (RAG)
- LangChain
- Vector Databases
- Semantic Search
- Prompt Engineering
- Google Gemini API
- HuggingFace Embeddings
- Pinecone Integration
- Streamlit Deployment
- End-to-End GenAI Application Development

---
