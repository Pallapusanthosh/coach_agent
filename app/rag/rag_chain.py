from app.rag.vector_store import build_vector_store
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from app.rag.prompt import build_prompt
load_dotenv()
import os
# print(os.getenv("GOOGLE_API_KEY"))

db = build_vector_store()

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

def run_rag(query: str, context: dict, mode: str = "chat") -> str:
    docs = db.similarity_search(query, k=3)
    prompt = build_prompt(
        mode=mode,
        context=context,
        docs=docs,
        query=query
    )
    response = llm.invoke(prompt)
    return response.content

