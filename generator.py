import ollama


# ============================================================
# Generate Answer using Ollama (Llama 3.2 3B)
# ============================================================

def generate_answer(question, reranked_results):

    # --------------------------------------------------------
    # Build context from RERANKED results (top 3 فقط)
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
    # Prompt
    # --------------------------------------------------------

    prompt = f"""You are a contract document question answering system.

Answer the question using ONLY information explicitly stated in the context below.

Rules:
- Do not guess.
- Do not use outside knowledge.
- Give a concise, direct answer.
- If the answer is not explicitly stated, say exactly: The answer is not available in the provided context.
- Some context may contain pricing tables where each row lists a value for "One Block" followed by a value for "Two Blocks" (in that order, the "Two Blocks" value always comes last on the line). When asked about "two blocks" or a combined/total amount, always use the LAST number on the relevant line, not the first or middle one.
- The row labeled "Grand Total" is the sum of Basic Scope + Optional Scope combined. It is NOT the value of "Basic Scope" alone, and NOT the value of "Optional Scope" alone. If asked specifically for the Basic Scope amount, use the row labeled "Basic Scope (Stage 1)" (not "Grand Total"). If asked specifically for the Optional Scope amount, use the row labeled "Total Optional" (not "Grand Total"). Never reuse the "Grand Total" number for a sub-scope question.

Example of this pattern:
"Basic Scope (Stage 1) -(skeleton + Finishes +MEP 1st Fix ) 143,792,498 287,584,996"
Here, 143,792,498 is the One Block value and 287,584,996 is the Two Blocks value for Basic Scope specifically.
"Total Optional 70,769,131 141,538,263"
Here, 141,538,263 is the Two Blocks value for Optional Scope specifically — this is different from Grand Total.
"Grand Total 214,561,629 429,123,258"
This row is the SUM of Basic Scope + Optional Scope. It answers "what is the total price", but never answers "what is the Basic Scope amount" or "what is the Optional Scope amount" individually.

Context:
{context}

Question:
{question}

Direct answer:"""

    # --------------------------------------------------------
    # Call Ollama
    # --------------------------------------------------------

    response = ollama.generate(
        model="llama3.2:3b",
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