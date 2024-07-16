# embedding.py
import os
import glob
import pickle
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

def create_embeddings(pdf_directory, embeddings_file):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    pdf_files = glob.glob(os.path.join(pdf_directory, '*.pdf'))
    documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        documents.extend(loader.load()) 

    # FAISS 벡터 스토어 생성
    vector_store = FAISS.from_documents(documents, embeddings)
    # 벡터 스토어 저장
    with open(embeddings_file, 'wb') as f:
        pickle.dump(vector_store, f)
    
    print(f"Embeddings saved to {embeddings_file}")

if __name__ == "__main__":
    pdf_directory = './docs'  # PDF 파일들이 있는 디렉토리
    embeddings_file = './embeddings/embeddings.pkl'  # 저장할 임베딩 파일 경로
    create_embeddings(pdf_directory, embeddings_file)
