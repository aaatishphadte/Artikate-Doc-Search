import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from backend.knowledge_base.vectorstore import VECTORSTORE_PATH

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def ingest_documents(doc_folder="docs"):
    docs = []

    # Load PDF and TXT files
    for filename in os.listdir(doc_folder):
        file_path = os.path.join(doc_folder, filename)

        if filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            docs.extend(loader.load())

        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_docs = []

    for doc in docs:
        splits = text_splitter.split_text(doc.page_content)
        for chunk in splits:
            chunked_docs.append(
                Document(
                    page_content=chunk,
                    metadata=doc.metadata,  # preserve source + page
                )
            )

    # Build FAISS from scratch (NO loading first)
    vectorstore = FAISS.from_documents(chunked_docs, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)

    print(f"Created new FAISS index with {len(chunked_docs)} chunks.")


if __name__ == "__main__":
    ingest_documents()
