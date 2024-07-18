from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
from langchain.vectorstores.faiss import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn
import requests

class Question(BaseModel):
    query: str

app = FastAPI()

embeddings_file = 'embeddings/embeddings.pkl'
model_dir = "./models/llama3-small"

with open(embeddings_file, 'rb') as f:
    vector_store = pickle.load(f)

def answer_question(vector_store, query):
    similar_docs = vector_store.similarity_search(query, k=5)
    context = "\n\n".join([f"[{doc.metadata['department']}] {doc.page_content}" for doc in similar_docs])

    data = {
        "model": "llama3",
        "messages": [
            {
                "role": "user",
                "content": context + f"\nQuestion: {query}\nAnswer:"

            }
        ],
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json"
    }

    url = "http://localhost:11434/api/chat"

    response = requests.post(url, headers=headers, json=data)
    return response.json()["message"]["content"]

@app.post("/ask")
async def ask_question(question: Question):
    try:
        answer = answer_question(vector_store, question.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
