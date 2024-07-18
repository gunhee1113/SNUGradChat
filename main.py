# main.py
import streamlit as st
import pickle
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
import os

api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo")

embeddings_dir = './embeddings'

vector_store = Chroma(persist_directory=embeddings_dir, embedding_function=OpenAIEmbeddings())

def format_docs(docs):
    return "\n\n".join([f"[department : {doc.metadata['department']}] [file_name : {doc.metadata['file_name']}] {doc.page_content}" for doc in docs])

metadata_field_info = [
    AttributeInfo(
        name="department",
        description="The department of the document",
        type="string",
    ),AttributeInfo(
        name="file_name",
        description="The file name of the document. If contains information about specific graduation regulations, it can be used to refer to the document.",
        type="string",
    ),
]

#retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
retriever = SelfQueryRetriever.from_llm(
    llm,
    vector_store,
    document_contents="Graduation regulations by department",
    metadata_field_info=metadata_field_info,
)

system_prompt = (
    """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer 
    the question.
    If you don't know the answer, say that you don't know.
    If the question is ambiguous, request user to provide more information.

    When you answer, please make sure to specify which reference
    in the context you used to answer at the last line of your answer.
    The reference should have the format of 
    \n\n[department : (department)] [file_name : (file_name)]. 
    The department and file_name should be the metadata of the document.
    Also, please just use only necessary context among the retrieved context.
    the content of reference can be found in the context below.

    Furthermore, when you answer, don't make answer by yourself. 
    Please answer exactly as it is in context.
    \n
    User answer should be like this:
    \n
    "(relevant text in the context below)
    \n\n
    [department : (department)] [file_name : (file_name)]"
    \n
    ------------------------------------------\n
    Example\n
    context : 
    <context>
    [department : 컴퓨터공학부] [file_name : 주전공(다전공)-졸업규정.pdf] 2015~2018학번
    이수학점
    컴퓨터공학부의 전공학점을 41학점 이상 이수하고 타 학부에서 정하는 필요학점 이수
    전필
    이산수학, 논리설계(4학점), 컴퓨터프로그래밍(4학점), 전기전자회로, 자료구조(4학점), 컴퓨터구조, 
    소프트웨어 개발의 원리와 실습, 시스템프로그래밍(4학점), 하드웨어시스템설계, 알고리즘, 공대 공통교과목
    전선내규필수
    컴퓨터공학세미나 또는 IT-리더십세미나(세미나는 1과목만 이수), 창의적통합설계 1 또는 창의적통합설계 2
    2011~2014학번
    이수학점
    컴퓨터공학부의 전공학점을 41학점 이상 이수하고 타 학부에서 정하는 필요학점 이수
    전필
    이산수학, 논리설계, 논리설계실험, 컴퓨터프로그래밍, 전기전자회로, 자료구조, 프로그래밍의 원리, 컴퓨터구조, 
    운영체제, 프로그래밍언어, 알고리즘, 공대 공통교과목
    전선내규필수
    컴퓨터공학세미나, IT-리더십세미나, 프로젝트1 또는 프로젝트2
    2 / 2

    [department : 컴퓨터공학부] [file_name : 주전공(단일전공)-졸업규정.pdf] 전필
    이산수학(3) 논리설계(4), 컴퓨터프로그래밍(4), 전기전자회로(3), 자료구조(4), 컴퓨터구조(3), 
    소프트웨어 개발의 원리와 실습(4), 시스템프로그래밍(4), 알고리즘(3), 공대 공통교과목(3)
    전선내규필수
    컴퓨터공학세미나(1) 또는 IT-리더십세미나(1) (세미나는 1과목만 이수), 
    창의적통합설계1 (3) 또는 창의적통합설계2 (3)
    2015~2018학번
    이수학점
    전공학점 63학점 이수(전필 37학점 + 전선 내규 4학점을 포함한 63학점 이수)
    전필
    이산수학, 논리설계(4), 컴퓨터프로그래밍(4), 전기전자회로, 자료구조(4), 컴퓨터구조, 
    소프트웨어 개발의 원리와 실습, 시스템프로그래밍(4), 하드웨어시스템설계, 알고리즘, 공대 공통교과목
    전선내규필수
    컴퓨터공학세미나(1) 또는 IT-리더십세미나(1)(세미나는 1과목만 이수), 
    창의적통합설계 1(3) 또는 창의적통합설계 2(3)
    졸업규정
    주전공(단일전공)
    ※ 컴퓨터공학부 소속 학생들의 졸업기준을 '입학년도' 이후 기준 중 학생이 선택하여 졸업기준을 정할 수 있다.
    ※ 자유전공학부 주전공 졸업기준
    컴퓨터공학부를 주전공하는 자유전공학부생은 컴퓨터공학부 주전공 졸업기준을 ‘입학년도’ 또는 ‘주전공 선택년도’ 기준으로 학생이 선택하여 졸업기준을 
    정할 수 있다. 단, 졸업기준을 변경하는 학생은 졸업 마지막 학기에 컴퓨터공학부에서 졸업예정 확인서를 받아 자유전공학부로 제출하여야 한다. (2018년 
    8월 졸업자부터 적용)
    1 / 2
    </context>
   
    question : 
    <question>
    2020학번 주전공 단일전공 졸업기준 알려줘.
    </question>

    answer : 
    <answer>
    이수학점 : 전공학점 63학점 이수(전필 31학점 + 전선 내규필수 8학점을 포함한 63학점 이수)\n
    전필 : 이산수학(3), 논리설계(4), 컴퓨터프로그래밍(4), 전기전자회로(3), 자료구조(4), 컴퓨터구조(3),  
    시스템프로그래밍(4), 알고리즘(3), 공대 공통교과목(3)\n
    전선내규필수 : 소프트웨어 개발의 원리와 실습(4), 컴퓨터공학세미나(1) 또는 IT-리더십세미나(1)(세미나는 1과목만 이수),  
    창의적통합설계 1(3) 또는 창의적통합설계 2(3)\n

    reference : [department : 컴퓨터공학부] [file_name : ./docs/컴퓨터공학부/주전공(단일전공)-졸업규정.pdf]
    </answer>\n
    ------------------------------------------\n
    \n
    context : {context}
    """
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def answer_question(question):
    context = retriever.invoke(question)
    formatted_context = format_docs(context)
    print("context : \n", formatted_context)
    response = rag_chain.invoke(question)
    return response

if __name__ == "__main__":
    st.title("💬 Chatbot")
    st.caption("🚀 A Streamlit chatbot powered by OpenAI")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if question := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)
        answer = answer_question(question)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)