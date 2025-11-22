# from rest_framework import status
# from rest_framework.response import Response
# from rest_framework.views import APIView

# from backend.knowledge_base.vectorstore import get_vectorstore  # your FAISS retrieval
# from backend.rag.llm import generate_answer  # local Hugging Face model

# # Load FAISS vectorstore once
# vectorstore = get_vectorstore()


# class AskQuestionView(APIView):
#     """
#     POST request with JSON: {"question": "Your question here"}
#     Returns answer generated using local LLM + retrieved context.
#     """

#     def post(self, request):
#         question = request.data.get("question", "")
#         if not question:
#             return Response(
#                 {"error": "Question is required"}, status=status.HTTP_400_BAD_REQUEST
#             )

#         # Retrieve similar documents from FAISS
#         retrieved_docs = vectorstore.similarity_search(question, k=3)
#         context = "\n".join([doc.page_content for doc in retrieved_docs])

#         # Create prompt for local LLM
#         prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

#         # Generate answer using local Hugging Face model
#         answer = generate_answer(prompt)

#         return Response({"answer": answer})

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from backend.knowledge_base.vectorstore import get_vectorstore
from backend.rag.llm import generate_answer

vectorstore = get_vectorstore()


class AskQuestionView(APIView):
    """
    POST request with JSON: {"question": "Your question here"}
    Returns: { "answer": "...", "sources": ["file.pdf - Page 1"] }
    """

    def post(self, request):
        question = request.data.get("question", "")
        if not question:
            return Response(
                {"error": "Question is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        # Retrieve top 3 similar documents
        retrieved_docs = vectorstore.similarity_search(question, k=3)

        # Build context string from chunks
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # LLM prompt for GPT-2
        prompt = (
            "Answer the question based strictly on the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )

        # Generate answer using GPT-2 local model
        answer = generate_answer(prompt)

        # Extract sources from FAISS metadata
        sources = []
        for doc in retrieved_docs:
            src = doc.metadata.get("source", "Unknown Source")
            page = doc.metadata.get("page")
            if page is not None:
                sources.append(f"{src} - Page {page}")
            else:
                sources.append(src)

        # Return required JSON structure
        return Response({"answer": answer, "sources": sources})
