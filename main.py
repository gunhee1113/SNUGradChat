import streamlit as st
import requests

st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by LLM server")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # LLM 서버에 질문을 보냄
    response = requests.post("http://localhost:8000/ask", json={"query": prompt})
    
    if response.status_code == 200:
        answer = response.json()["answer"]
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
    else:
        st.session_state.messages.append({"role": "assistant", "content": "Sorry, there was an error processing your request."})
        st.chat_message("assistant").write("Sorry, there was an error processing your request.")
