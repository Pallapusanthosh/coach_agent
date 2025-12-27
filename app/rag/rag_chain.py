from dotenv import load_dotenv
import os
from app.rag.prompt import build_prompt

load_dotenv()

def get_vector_db():
    from app.rag.vector_store import build_vector_store
    return build_vector_store()

def get_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )

def run_rag(query: str, context: dict, mode: str = "chat") -> str:
    db = get_vector_db()     # ✅ created only when endpoint is hit
    llm = get_llm()          # ✅ created only when endpoint is hit

    docs = db.similarity_search(query, k=3)

    prompt = build_prompt(
        mode=mode,
        context=context,
        docs=docs,
        query=query
    )

    response = llm.invoke(prompt)
    return response.content
