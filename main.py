# main.py
import streamlit as st
from rag import rag_manager


if __name__ == "__main__":
    st.title("💬 SNUGradChat")
    st.caption("🚀 Chatbot for grad conditions")
    manager = rag_manager()
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "졸업규정에 대해 질문해보세요!"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if question := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)
        answer = manager.answer_question(question)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)