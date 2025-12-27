from langchain_community.document_loaders import TextLoader
from pathlib import Path

def load_documents():
    docs = []
    base_path = Path("app/rag/knowledge")

    for file in base_path.glob("*.txt"):
        loader = TextLoader(str(file) , encoding="utf-8")
        docs.extend(loader.load())

    return docs
