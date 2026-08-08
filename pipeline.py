from retriever import (
    load_vectorstore,
    search as dense_search
)

from bm25_retriever import (
    load_documents,
    build_bm25,
    search as bm25_search
)

from hybrid_retriever import (
    reciprocal_rank_fusion
)

from reranker import rerank

from generator import generate_answer


# ============================================================
# Helper Functions
# ============================================================

def get_document_text(document):

    if isinstance(document, dict):

        return document.get(
            "text",
            ""
        )

    return document.page_content


def get_document_page(document):

    if isinstance(document, dict):

        return document.get(
            "metadata",
            {}
        ).get(
            "page",
            "?"
        )

    return document.metadata.get(
        "page",
        "?"
    )


# ============================================================
# Load Resources
# ============================================================

print("Loading vectorstore...")

vectorstore = load_vectorstore()


print("Loading documents...")

documents = load_documents(
    vectorstore
)

print(
    f"Number of documents: {len(documents)}"
)


print("Building BM25...")

bm25 = build_bm25(
    documents
)


# ============================================================
# Query
# ============================================================

query = input(
    "\nEnter your question: "
)


print(
    f"\nQuestion: {query}\n"
)


# ============================================================
# Dense Retrieval
# ============================================================

print(
    "Running Dense Retrieval..."
)

dense_results = dense_search(
    query,
    vectorstore,
    k=10
)


# ============================================================
# BM25 Retrieval
# ============================================================

print(
    "Running BM25 Retrieval..."
)

bm25_results = bm25_search(
    query,
    bm25,
    documents,
    k=10
)


# ============================================================
# Hybrid Retrieval
# ============================================================

print(
    "Running Hybrid Retrieval..."
)

hybrid_results = reciprocal_rank_fusion(
    dense_results,
    bm25_results
)


print(
    f"Hybrid results: {len(hybrid_results)}"
)


# ============================================================
# Reranking
# ============================================================

print(
    "Running Cross-Encoder Reranker..."
)

reranked_results = rerank(
    query,
    hybrid_results,
    top_k=5
)


# ============================================================
# Display Reranked Results
# ============================================================

print(
    "\n======================================"
)

print(
    "FINAL RERANKED RESULTS"
)

print(
    "======================================\n"
)


for i, (
    document,
    score
) in enumerate(
    reranked_results,
    start=1
):

    page = get_document_page(
        document
    )

    content = get_document_text(
        document
    )

    print(
        f"Result {i}"
    )

    print(
        f"Page: {page}"
    )

    print(
        f"Reranker Score: {score:.6f}"
    )

    print(
        "Text:"
    )

    print(
        content[:1000]
    )

    print(
        "\n--------------------------------------\n"
    )


# ============================================================
# Generate Final Answer
# ============================================================

print(
    "\nGenerating final answer..."
)

result = generate_answer(
    query,
    reranked_results
)


# ============================================================
# Final Answer
# ============================================================

print(
    "\n======================================"
)

print(
    "CONTRACT AI ANSWER"
)

print(
    "======================================"
)


print(
    f"\nAnswer:\n{result['answer']}"
)


print(
    f"\nSource Page:\n{result['page']}"
)


print(
    f"\nEvidence:\n{result['evidence']}"
)


print(
    "\n======================================"
)