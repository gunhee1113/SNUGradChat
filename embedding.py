# embedding.py
import os
import glob
import pickle
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def create_embeddings(pdf_root_directory, embeddings_file):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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

            # TextSplitter를 사용하여 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            for pdf_doc in pdf_documents:
                texts = text_splitter.split_text(pdf_doc.page_content)
                for text in texts:
                    # 학과명을 메타데이터로 포함
                    documents.append(Document(page_content=text, metadata={"department": department_name}))

    # FAISS 벡터 스토어 생성
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # 벡터 스토어 저장
    with open(embeddings_file, 'wb') as f:
        pickle.dump(vector_store, f)
    
    print(f"Embeddings saved to {embeddings_file}")

if __name__ == "__main__":
    pdf_root_directory = './docs'  # 학과별 PDF 파일들이 있는 루트 디렉토리
    embeddings_file = './embeddings/embeddings.pkl'  # 저장할 임베딩 파일 경로
    create_embeddings(pdf_root_directory, embeddings_file)
