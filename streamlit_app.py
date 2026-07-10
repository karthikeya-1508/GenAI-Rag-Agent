import os

import streamlit as st
from dotenv import load_dotenv
from google import genai

from rag_pipeline import (
    PDF_FOLDER,
    INDEX_NAME,
    create_embedding_model,
    create_vector_store,
    connect_to_pinecone,
    load_documents,
    run_rag_pipeline,
    split_documents,
)


@st.cache_resource
def initialize_rag_system():
    load_dotenv()

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    if not pinecone_api_key or not gemini_api_key:
        st.error("Missing API keys. Please set PINECONE_API_KEY and GEMINI_API_KEY in your environment or .env file.")
        st.stop()

    documents, cleaned_documents = load_documents(PDF_FOLDER)
    _, rec_chunks = split_documents(cleaned_documents)

    embedding_model = create_embedding_model()
    _, index = connect_to_pinecone(pinecone_api_key, INDEX_NAME)
    create_vector_store(index, rec_chunks, embedding_model)
    client = genai.Client(api_key=gemini_api_key)

    return embedding_model, index, client, gemini_model


def main():
    st.set_page_config(page_title="RAG Q&A Bot", page_icon="🤖", layout="centered")
    st.title("RAG Q&A Bot")
    st.write("Ask a question based on the documents in the PDF folder.")

    embedding_model, index, client, gemini_model = initialize_rag_system()

    question = st.text_area(
        "Question",
        placeholder="Example: Explain the Kalman Filter and its role in Sensor Fusion.",
        height=120,
    )

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question first.")
            return

        with st.spinner("Generating answer..."):
            answer, _, _ = run_rag_pipeline(
                question,
                embedding_model,
                index,
                client,
                k=5,
                model_name=gemini_model,
            )

        st.subheader("Answer")
        st.write(answer)


if __name__ == "__main__":
    main()
