import ollama



# Generate Answer using Ollama (Llama 3.2 1B)


def generate_answer(question, reranked_results):

    # --------------------------------------------------------
    # Build context from RERANKED results 
    # --------------------------------------------------------

    context_parts = []
    source_pages = []

    for document, score in reranked_results[:3]:

        if isinstance(document, dict):
            text = document["text"]
            page = document["metadata"].get("page", "?")
        else:
            text = document.page_content
            page = document.metadata.get("page", "?")

        context_parts.append(text)
        source_pages.append(str(page))

    context = "\n\n".join(context_parts)

    # --------------------------------------------------------
    # Prompt (generic — no document-specific examples or values)
    # --------------------------------------------------------

    prompt = f"""You are a contract document question answering system.

Answer the question using ONLY information explicitly stated in the context below.

Rules:
- Do not guess.
- Do not use outside knowledge.
- Give a concise, direct answer.
- If the answer is not explicitly stated, say exactly: The answer is not available in the provided context.
- If the context contains a table with multiple columns of values, state clearly which column or row the answer comes from, so the source of the number is unambiguous.

Context:
{context}

Question:
{question}

Direct answer:"""

    # --------------------------------------------------------
    # Call Ollama
    # --------------------------------------------------------

    response = ollama.generate(
        model="llama3.2:1b",
        prompt=prompt,
        options={"temperature": 0}
    )

    answer = response["response"].strip()

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "answer": answer,
        "page": source_pages[0] if source_pages else "?",
        "evidence": context_parts[0] if context_parts else ""
    }
