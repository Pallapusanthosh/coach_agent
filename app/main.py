from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
from app.rag.rag_chain import run_rag
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Nutrition RAG Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RAGRequest(BaseModel):
    user_id: str
    question: str
    context: Dict[str, Any]

class RAGResponse(BaseModel):
    answer: str

@app.get("/")
def health():
    return {"status": "RAG agent running"}



@app.post("/agent/chat", response_model=RAGResponse)
def agent_chat(req: RAGRequest):
    answer = run_rag(
        query=req.question,
        context=req.context,
        mode="chat"
    )
    return {"answer": answer}

# 2️⃣ DAILY ADVICE
@app.post("/agent/daily-advice", response_model=RAGResponse)
def daily_advice(req: RAGRequest):
    answer = run_rag(
        query="Give daily nutrition advice",
        context=req.context,
        mode="daily_advice"
    )
    return {"answer": answer}

# 3️⃣ RECOMMENDATION
@app.post("/agent/recommendation", response_model=RAGResponse)
def recommendation(req: RAGRequest):
    answer = run_rag(
        query="Provide meal recommendations based on my recent meals and nutrition goals.",
        context=req.context,
        mode="recommendation"
    )
    return {"answer": answer}