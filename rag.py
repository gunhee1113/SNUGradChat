import boto3.session
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.bedrock import AmazonKnowledgeBasesRetriever
from langchain.chains.query_constructor.base import AttributeInfo
import os
import boto3
from s3 import presigned_url

class rag_manager:
    def __init__(self):
        os.getenv("OPENAI_API_KEY")

        self.llm = ChatOpenAI(model="gpt-4o-mini")

        self.embeddings_dir = './embeddings'

        self.vector_store = Chroma(persist_directory=self.embeddings_dir, embedding_function=OpenAIEmbeddings())

        self.metadata_field_info = [
            AttributeInfo(
                name="department",
                description="The department of the document",
                type="string",
            ),AttributeInfo(
                name="file_name",
                description="The file name of the document. If contains information about specific graduation regulations, it can be used to refer to the document.",
                type="string",
            ),AttributeInfo(
                name="file_location",
                description="The location of the document.",
                type="string",
            ),
        ]

        #retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        self.retriever = SelfQueryRetriever.from_llm(
            self.llm,
            self.vector_store,
            document_contents="Graduation regulations by department",
            metadata_field_info=self.metadata_field_info,
        )

        self.system_prompt = (
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
            reference : (file_location)"
            \n
            ------------------------------------------\n
            Example\n
            context : 
            <context>
            [department : 컴퓨터공학부] [file_location : ./docs/컴퓨터공학부/주전공(다전공)-졸업규정.pdf]
            2015~2018학번
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

            [department : 컴퓨터공학부] [file_location : ./docs/컴퓨터공학부/주전공(단일전공)-졸업규정.pdf]
            전필
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

            reference : docs/컴퓨터공학부/주전공(단일전공)-졸업규정.pdf
            </answer>\n
            ------------------------------------------\n
            \n
            context : {context}
            """
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{question}"),
            ]
        )

        self.rag_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def format_docs(self, docs):
        return "\n\n".join([f"[department : {doc.metadata['department']}] [file_location : {doc.metadata['file_location']}]\n{doc.page_content}" for doc in docs])
    
    def answer_question(self, question):
        context = self.retriever.invoke(question)
        formatted_context = self.format_docs(context)
        print("context : \n", formatted_context)
        response = self.rag_chain.invoke(question)
        url = response.split(" ")[-1]
        doc_url = presigned_url(os.getenv("S3_BUCKET_NAME"), url)
        response = response + "\n\n" + "url: " + doc_url
        return response

class aoss_rag_manager(rag_manager):
    def __init__(self):
        os.getenv("OPENAI_API_KEY")

        self.llm = ChatOpenAI(model="gpt-4o-mini")

        session = boto3.Session(profile_name='default')
        region = "ap-northeast-1"
        bedrock_client = boto3.client('bedrock-runtime', region_name = region)

        #retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        self.retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id="45DY7IMJKJ",
            retrieval_config={"vectorSearchConfiguration": 
                                {"numberOfResults": 4,
                                'overrideSearchType': "SEMANTIC", # optional
                                }
                            },
        )

        self.system_prompt = (
            """You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer 
            the question.
            If you don't know the answer, say that you don't know.
            If the question is ambiguous, request user to provide more information.

            When you answer, please make sure to specify which reference
            in the context you used to answer at the last line of your answer.
            The reference should have the format of 
            \n\n[file_location : (file_name)]. 
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
            [file_location : (file_name)]"
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

            reference : [file_location : s3://snugradchat-bucket/docs/컴퓨터공학부/부전공-졸업규정.pdf]
            </answer>\n
            ------------------------------------------\n
            \n
            context : {context}
            """
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{question}"),
            ]
        )

        self.rag_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def format_docs(self, docs):
        return "\n\n".join([f"[file : {doc.metadata['location']['s3Location']['uri']}] {doc.page_content}" for doc in docs])
    
    def answer_question(self, question):
        context = self.retriever.invoke(question)
        formatted_context = self.format_docs(context)
        print("context : \n", formatted_context)
        response = self.rag_chain.invoke(question)
        return response

if __name__ == "__main__":
    manager = aoss_rag_manager()
    query = "컴퓨터공학부 21학번 복수전공 졸업규정 알려줘"
    print(manager.retriever.invoke(query))