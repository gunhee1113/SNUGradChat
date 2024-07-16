import streamlit as st
import pickle
from langchain.vectorstores.faiss import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_embeddings(embeddings_file):
    with open(embeddings_file, 'rb') as f:
        vector_store = pickle.load(f)
    return vector_store

def answer_question(vector_store, query, model, tokenizer):
    similar_docs = vector_store.similarity_search(query, k=5)
    context = " ".join([doc.page_content for doc in similar_docs])

    inputs = tokenizer.encode(f"Context: {context}\nQuestion: {query}\nAnswer:", return_tensors='pt')

    with torch.no_grad():
        outputs = model.generate(inputs, max_length=512, num_return_sequences=1)
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    embeddings_file = 'path/to/embeddings.pkl'  # 임베딩 파일 경로
    vector_store = load_embeddings(embeddings_file)

    model_dir = "models/llama3-small"  # 로컬에 저장된 모델 경로
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    st.title("💬 Chatbot")
    st.caption("🚀 A Streamlit chatbot powered by OpenAI")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        answer = answer_question(vector_store, prompt, model, tokenizer)
        msg = answer.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)    

