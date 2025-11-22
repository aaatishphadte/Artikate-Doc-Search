# import os

# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS

# # Path to store the FAISS index locally
# VECTORSTORE_PATH = os.path.join(os.path.dirname(__file__), "faiss_index")

# # Initialize Hugging Face embeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# def get_vectorstore():
#     """Load existing FAISS index or create a new one."""
#     if os.path.exists(VECTORSTORE_PATH):
#         vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings)
#         print("Loaded existing FAISS index.")
#     else:
#         vectorstore = FAISS.from_texts(["dummy text"], embeddings)
#         vectorstore.save_local(VECTORSTORE_PATH)
#         print("Created new FAISS index.")
#     return vectorstore

import os

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Path to store the FAISS index locally
VECTORSTORE_PATH = os.path.join(os.path.dirname(__file__), "faiss_index")

# Initialize Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_vectorstore():
    """
    Load existing FAISS index.
    If the index is missing, instruct user to run ingestion.py.
    """
    if os.path.exists(VECTORSTORE_PATH):
        vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings)
        print("Loaded existing FAISS index.")
        return vectorstore

    raise ValueError(
        "FAISS index not found! Run ingestion.py first to create the vectorstore."
    )
