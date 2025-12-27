from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.rag.loader import load_documents
import os

def build_vector_store():
    documents = load_documents()

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    db = FAISS.from_documents(documents, embeddings)
    return db
