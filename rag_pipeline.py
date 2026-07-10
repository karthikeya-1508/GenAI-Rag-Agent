import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_community.docstore.document import Document
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from google import genai
from google.genai.errors import ClientError


WORKSPACE_ROOT = Path(__file__).resolve().parent
PDF_FOLDER = WORKSPACE_ROOT / "pdf_data"
INDEX_NAME = "rag-index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    return text.strip()


def load_documents(pdf_folder: str | os.PathLike | None = None):
    folder = Path(pdf_folder or PDF_FOLDER)
    loader = DirectoryLoader(
        str(folder),
        glob=["*.pdf", "*.docx"],
        loader_cls=lambda path: PyPDFLoader(path) if path.lower().endswith(".pdf") else Docx2txtLoader(path),
    )
    documents = loader.load()
    if not documents:
        raise ValueError(f"No PDF or DOCX files found in {folder}")

    cleaned_documents = [clean_text(doc.page_content) for doc in documents]
    return documents, cleaned_documents


def split_documents(cleaned_documents):
    char_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    rec_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)

    char_chunks = []
    rec_chunks = []

    for doc in cleaned_documents:
        char_chunks.extend(char_splitter.split_text(doc))
        rec_chunks.extend(rec_splitter.split_text(doc))

    return char_chunks, rec_chunks


def create_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    return HuggingFaceEmbeddings(model_name=model_name)


def connect_to_pinecone(api_key: str | None = None, index_name: str = INDEX_NAME):
    if not api_key:
        raise ValueError("PINECONE_API_KEY is missing")

    pc = Pinecone(api_key=api_key)
    if index_name not in pc.list_indexes().names():
        dimension = 384
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc, pc.Index(index_name)


def create_vector_store(index, rec_chunks, embedding_model):
    rec_documents = [Document(page_content=chunk) for chunk in rec_chunks]
    rec_embeddings = embedding_model.embed_documents([doc.page_content for doc in rec_documents])
    rec_embeddings_np = np.array(rec_embeddings).astype("float32")

    vectors = []
    for i, (embedding, chunk) in enumerate(zip(rec_embeddings_np, rec_chunks)):
        vectors.append({
            "id": str(i),
            "values": embedding.tolist(),
            "metadata": {"text": chunk},
        })

    index.upsert(vectors=vectors)
    return rec_embeddings_np


def build_context_passage(results):
    passages = []
    for match in results["matches"]:
        passages.append(match["metadata"]["text"])
    return "\n\n".join(passages)


def build_prompt(context, question):
    prompt = f'''
    "You are an AI assistant that answers questions ONLY from the provided context."

    "Instructions:"
    "1.Use only the given context."
    "2.Do not use outside knowledge."
    "3.If the answer is not found in the context, say:"
      "I couldn't find the answer in the provided documents."
    "4.Give a detailed explanation."
    "5.Include every relevant point from the context."
    "6.If multiple documents discuss the topic, combine the information."
    "7.Use headings and bullet points when appropriate."
    "8.Explain each point instead of listing only keywords."
    "9.Do not infer, assume, or generate information that is not explicitly present in the provided context."
    "10.If the context partially answers the question, answer only the available part and clearly mention what information is missing."
    "11.Do not use your general knowledge, even if you know the answer."
    "12.Every statement in the answer must be directly supported by the provided context."
    "13.Do not add examples, comparisons, explanations, or conclusions unless they appear in the context."
    "14.If multiple retrieved chunks contain relevant information, combine them into a single coherent answer without adding new information."
    "15.If the answer cannot be completely answered from the context, state:"
     "The provided documents do not contain enough information to fully answer this question."
    "16. Do not answer using information that is not present in the retrieved context, even if it exists elsewhere in the original documents or in your own knowledge."

    Context:\n{context}\n\n
    Question: {question}
    Answer:
    '''
    return prompt


def retrieve_top_k_chunks(query_embedding, index, k=5):
    return index.query(vector=query_embedding, top_k=k, include_metadata=True)


def generate_answer_with_prompt_engineering(context, question, client, model_name=None):
    prompt = build_prompt(context, question)
    try:
        response = client.models.generate_content(
            model=model_name or DEFAULT_GEMINI_MODEL,
            contents=prompt,
            config={"temperature": 0.2, "max_output_tokens": 2048},
        )
        return response.text
    except ClientError as exc:
        print(f"Gemini generation failed: {exc}")
        return f"Gemini generation failed: {exc}"
    except Exception as exc:
        print(f"Unexpected error while generating answer: {exc}")
        return f"Unexpected error while generating answer: {exc}"


def run_rag_pipeline(question, embedding_model, index, client, k=5, model_name=None):
    query_embedding = embedding_model.embed_query(question)
    top_chunks = retrieve_top_k_chunks(query_embedding, index, k=k)
    context = build_context_passage(top_chunks)
    answer = generate_answer_with_prompt_engineering(context, question, client, model_name=model_name)
    return answer, context, top_chunks


def main():
    load_dotenv()

    question = os.getenv("USER_QUESTION", "Explain the Kalman Filter and its role in Sensor Fusion.")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    gemini_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

    if not pinecone_api_key or not gemini_api_key:
        print("Missing API keys. Set PINECONE_API_KEY and GEMINI_API_KEY in your environment or .env file.")
        return

    print("Loading documents...")
    documents, cleaned_documents = load_documents(PDF_FOLDER)
    print(f"Loaded {len(documents)} document(s)")

    print("Splitting documents into chunks...")
    _, rec_chunks = split_documents(cleaned_documents)
    print(f"Created {len(rec_chunks)} chunks")

    print("Creating embedding model...")
    embedding_model = create_embedding_model()

    print("Connecting to Pinecone...")
    pc, index = connect_to_pinecone(pinecone_api_key, INDEX_NAME)
    print("Uploading vectors to Pinecone...")
    create_vector_store(index, rec_chunks, embedding_model)

    print("Creating Gemini client...")
    client = genai.Client(api_key=gemini_api_key)

    print("\nQuestion:")
    print(question)

    answer, context, _ = run_rag_pipeline(question, embedding_model, index, client, k=5, model_name=gemini_model)
    print(answer)


if __name__ == "__main__":
    main()
