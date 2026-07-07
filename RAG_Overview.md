# Retrieval-Augmented Generation (RAG)

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that enhances a Large Language Model (LLM) by retrieving relevant information from external documents before generating an answer.

Instead of relying only on its pre-trained knowledge, the LLM uses the retrieved documents to provide more accurate and context-aware responses.

---

## Why do we use RAG?

We use RAG because:

- LLMs have a limited context window and cannot process millions of tokens at once.
- Companies have private documents (HR policies, product manuals, healthcare records, etc.) that are not part of the LLM's training data.
- RAG retrieves only the relevant information, making responses:
  - More accurate
  - Faster
  - Cost-effective
  - Less prone to hallucinations

---

# RAG Pipeline

1. **Load Documents**
   - Load PDFs, Word files, Text files, etc.

2. **Extract Text**
   - Extract text and split it into chunks.

3. **Create Chunks**
   - Divide large documents into smaller chunks (e.g., 2000 tokens) with optional overlap.

4. **Generate Embeddings**
   - Convert each chunk into vector embeddings using models like OpenAI, Gemini, Hugging Face, or Ollama.

5. **Store in Vector Database**
   - Save embeddings in a Vector DB such as FAISS, ChromaDB, Pinecone, or AstraDB.

6. **Retrieve Relevant Chunks**
   - Convert the user query into an embedding and retrieve the most similar chunks using similarity search (Cosine Similarity, Euclidean Distance, BM25).

7. **Prompt Engineering**
   - Combine the retrieved chunks with the user's question to create a prompt.

8. **Pass Context to LLM**
   - Send the retrieved chunks to the LLM (GPT-4, Gemini, Llama, etc.).

9. **Generate Answer**
   - The LLM generates an answer based on the retrieved context.

10. **Display Response**
    - Return the final answer to the user.

---
# RAG Flowchart

```text
        Documents
(PDF, DOCX, TXT, etc.)
          │
          ▼
   Load Documents
          │
          ▼
     Extract Text
          │
          ▼
 Create Chunks (+ Overlap)
          │
          ▼
 Generate Embeddings
          │
          ▼
 Store in Vector DB
          │
          ▼
      User Query
          │
          ▼
Convert Query to Embedding
          │
          ▼
 Retrieve Top-K Chunks
          │
          ▼
  Prompt Engineering
          │
          ▼
 Pass Context to LLM
          │
          ▼
  Generate Answer
          │
          ▼
  Display Response
```