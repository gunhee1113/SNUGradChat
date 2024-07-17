from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
from langchain.vectorstores.faiss import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn

class Question(BaseModel):
    query: str

app = FastAPI()

embeddings_file = 'embeddings/embeddings.pkl'
model_dir = "./models/llama3-small"

with open(embeddings_file, 'rb') as f:
    vector_store = pickle.load(f)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

def answer_question(vector_store, query, model, tokenizer):
    similar_docs = vector_store.similarity_search(query, k=5)
    context = " ".join([f"[{doc.metadata['department']}] {doc.page_content}" for doc in similar_docs])

    inputs = tokenizer(context + f"\nQuestion: {query}\nAnswer:", return_tensors='pt')
    inputs = {key: value.to(device) for key, value in inputs.items()}

    print("inputs:", inputs)
    print("decoded inputs:", tokenizer.decode(inputs['input_ids'][0]))
    
    print("Generating answer...")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)

    print(outputs)
    
    answer = tokenizer.decode(outputs[0])
    return answer

@app.post("/ask")
async def ask_question(question: Question):
    try:
        answer = answer_question(vector_store, question.query, model, tokenizer)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
