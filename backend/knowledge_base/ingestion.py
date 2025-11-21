import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter

from backend.knowledge_base.vectorstore import VECTORSTORE_PATH, get_vectorstore


def ingest_documents(doc_folder="docs"):
    # Initialize vectorstore
    vectorstore = get_vectorstore()

    # Read all files from folder
    docs = []
    for filename in os.listdir(doc_folder):
        file_path = os.path.join(doc_folder, filename)
        if filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            docs.extend(loader.load())
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())

    # Split large texts into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_docs = []
    for doc in docs:
        chunked_texts = text_splitter.split_text(doc.page_content)
        for chunk in chunked_texts:
            chunked_docs.append({"page_content": chunk, "metadata": doc.metadata})

    # Add to vectorstore
    vectorstore.add_texts([d["page_content"] for d in chunked_docs])
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"Ingested {len(chunked_docs)} chunks into vectorstore.")


if __name__ == "__main__":
    ingest_documents()
