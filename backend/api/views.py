from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from backend.knowledge_base.vectorstore import get_vectorstore  # your FAISS retrieval
from backend.rag.llm import generate_answer  # local Hugging Face model

# Load FAISS vectorstore once
vectorstore = get_vectorstore()


class AskQuestionView(APIView):
    """
    POST request with JSON: {"question": "Your question here"}
    Returns answer generated using local LLM + retrieved context.
    """

    def post(self, request):
        question = request.data.get("question", "")
        if not question:
            return Response(
                {"error": "Question is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        # Retrieve similar documents from FAISS
        retrieved_docs = vectorstore.similarity_search(question, k=3)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # Create prompt for local LLM
        prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

        # Generate answer using local Hugging Face model
        answer = generate_answer(prompt)

        return Response({"answer": answer})
