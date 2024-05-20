from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import os
from chromadb.config import Settings


#Define the chroma setting
CHROMA_SETTINGS = Settings(
    chroma_db_impl = 'duckdb+parquet',
    persist_directory = 'db',
    anonymized_telemetry = False
)

persist_directory = "./db"

def main():
    documents = []
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root,file))
                documents.extend(loader.load())

    if not documents:
        print("No documents found.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    #create embeddings here
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-l6-v2")
    #create vector store here
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    # db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db=None


if __name__ == "__main__":
    main()