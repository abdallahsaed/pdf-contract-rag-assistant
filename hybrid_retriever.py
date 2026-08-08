"""
hybrid_retriever.py
Combines Dense Retrieval and BM25 (sparse) results using
Reciprocal Rank Fusion (RRF).

RRF is a generic, document-agnostic fusion technique: it only
uses the *rank* each document received in each retrieval method,
not the document's content. This makes it reusable across any
PDF, unlike keyword-specific boosting rules.
"""


def get_document_text(document):
    if isinstance(document, dict):
        return document.get("text", "")
    return document.page_content


def reciprocal_rank_fusion(dense_results, bm25_results, k=60):
    """
    Combine Dense Retrieval and BM25 results using Reciprocal Rank Fusion.

    Parameters:
        dense_results: list of (document, score) from dense/semantic search
        bm25_results: list of (document, score) from BM25 search
        k: RRF constant (default 60, a common standard value that
           dampens the influence of very high individual ranks)

    Returns:
        list of (document, rrf_score), sorted by combined rank
    """
    rrf_scores = {}
    documents = {}

    for rank, (document, _) in enumerate(dense_results, start=1):
        text = get_document_text(document)
        if not text:
            continue
        doc_id = text
        documents[doc_id] = document
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

    for rank, (document, _) in enumerate(bm25_results, start=1):
        text = get_document_text(document)
        if not text:
            continue
        doc_id = text
        documents[doc_id] = document
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

    ranked_documents = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return [(documents[doc_id], score) for doc_id, score in ranked_documents]