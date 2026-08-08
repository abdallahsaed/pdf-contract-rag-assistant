"""
evaluation.py
Runs the retrieval + reranking pipeline on a fixed test question,
useful for manually inspecting retrieval quality during development.
"""

from retriever import load_vectorstore, search as dense_search
from bm25_retriever import load_documents, build_bm25, search as bm25_search
from hybrid_retriever import reciprocal_rank_fusion
from reranker import rerank


def get_document_text(document):
    if isinstance(document, dict):
        return document.get("text", "")
    return document.page_content


def get_document_page(document):
    if isinstance(document, dict):
        return document.get("metadata", {}).get("page", "?")
    return document.metadata.get("page", "?")


if __name__ == "__main__":

    print("Loading vectorstore...")
    vectorstore = load_vectorstore()

    print("Loading documents...")
    documents = load_documents(vectorstore)
    print(f"Number of documents: {len(documents)}")

    print("Building BM25...")
    bm25 = build_bm25(documents)

    query = "What is the defects notification period?"
    print(f"\nQuestion: {query}\n")

    print("Running Dense Retrieval...")
    dense_results = dense_search(query, vectorstore, k=10)

    print("Running BM25 Retrieval...")
    bm25_results = bm25_search(query, bm25, documents, k=10)

    print("Running Hybrid Retrieval...")
    hybrid_results = reciprocal_rank_fusion(dense_results, bm25_results)

    print("\n========== HYBRID TOP RESULTS ==========\n")
    for i, (document, score) in enumerate(hybrid_results[:10], start=1):
        page = get_document_page(document)
        content = get_document_text(document)
        print(f"--- Hybrid Result {i} | Page: {page} | RRF Score: {score:.6f} ---")
        print(content[:500])
        print()

    print("\nRunning Reranker...")
    reranked_results = rerank(query, hybrid_results[:20], top_k=5)

    print("\n========== RERANKED TOP RESULTS ==========\n")
    for i, (document, score) in enumerate(reranked_results, start=1):
        page = get_document_page(document)
        content = get_document_text(document)
        print(f"--- Reranked Result {i} | Page: {page} | Reranker Score: {score:.6f} ---")
        print(content[:500])
        print()

    print("\n==========================================")