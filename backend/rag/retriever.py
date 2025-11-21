from backend.knowledge_base.vectorstore import get_vectorstore

collection = get_vectorstore()


def retrieve_similar(query, k=5):
    query_embedding = collection.embedding.embed_query(query)
    results = collection.similarity_search_by_vector(query_embedding, k=k)
    return results
