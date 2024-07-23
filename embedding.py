# embedding.py
import os
import glob
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings

api_key = os.getenv("OPENAI_API_KEY")

def create_embeddings(pdf_root_directory, embeddings_dir):
    if os.path.exists(embeddings_dir):
        os.remove(embeddings_dir)
    os.makedirs(embeddings_dir)

    embeddings = OpenAIEmbeddings()

    documents = []
    # 학과별 디렉토리를 순회
    for department_name in os.listdir(pdf_root_directory):
        department_path = os.path.join(pdf_root_directory, department_name)
        if not os.path.isdir(department_path):
            continue

        pdf_files = glob.glob(os.path.join(department_path, '*.pdf'))
        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)
            pdf_documents = loader.load()

            # 모든 페이지의 텍스트를 하나로 결합
            full_text = '\n\n'.join([pdf_doc.page_content for pdf_doc in pdf_documents])

            #pdf_file에서 앞에 ./를 제거
            pdf_file = pdf_file[2:]

            # 하나의 Document 객체로 추가
            documents.append(Document(page_content=full_text, metadata={"department": department_name, "file_location": pdf_file}))
            print(f"Added document from {pdf_file}")

    # Chroma 벡터 스토어 생성
    vector_store = Chroma.from_documents(documents, embeddings, persist_directory=embeddings_dir)
    
    print(f"Embeddings saved to {embeddings_dir}") 

if __name__ == "__main__":
    pdf_root_directory = './docs'  # 학과별 PDF 파일들이 있는 루트 디렉토리
    embeddings_dir = './embeddings'  # 저장할 임베딩 파일 경로
    create_embeddings(pdf_root_directory, embeddings_dir)
