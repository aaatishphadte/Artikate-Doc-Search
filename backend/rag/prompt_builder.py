def build_prompt(question, context_chunks):
    context_text = "\n\n".join(context_chunks)
    return f"""
You are a knowledge assistant. Answer ONLY using the provided context.
If the answer is not in the context, say "The document does not contain this information."

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:
"""
