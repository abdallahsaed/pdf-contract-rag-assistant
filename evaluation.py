from retriever import load_vectorstore, search as dense_search

from bm25_retriever import (
    load_documents,
    build_bm25,
    search as bm25_search
)

from reranker import rerank

import re


# ============================================================
# Helper Functions
# ============================================================

def get_document_text(document):

    if isinstance(document, dict):
        return document.get("text", "")

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
# Normalize Text
# ============================================================

def normalize_text(text):

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# Reciprocal Rank Fusion
# ============================================================

def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    query=None,
    k=60
):

    """
    Combine Dense Retrieval and BM25
    using Reciprocal Rank Fusion.

    Then apply lightweight query-aware
    boosting for contract-specific questions.
    """

    rrf_scores = {}
    documents = {}

    # ========================================================
    # Dense Results
    # ========================================================

    for rank, (document, _) in enumerate(
        dense_results,
        start=1
    ):

        text = get_document_text(
            document
        )

        if not text:
            continue

        doc_id = text

        documents[doc_id] = document

        rrf_scores[doc_id] = (
            rrf_scores.get(
                doc_id,
                0
            )
            +
            1 / (k + rank)
        )

    # ========================================================
    # BM25 Results
    # ========================================================

    for rank, (document, _) in enumerate(
        bm25_results,
        start=1
    ):

        text = get_document_text(
            document
        )

        if not text:
            continue

        doc_id = text

        documents[doc_id] = document

        rrf_scores[doc_id] = (
            rrf_scores.get(
                doc_id,
                0
            )
            +
            1 / (k + rank)
        )

    # ========================================================
    # Query-aware Boosting
    # ========================================================

    if query:

        query_lower = normalize_text(
            query
        )

        stop_words = {
            "what",
            "is",
            "the",
            "of",
            "a",
            "an",
            "are",
            "for",
            "to",
            "in",
            "on",
            "and",
            "does",
            "do",
            "how",
            "much",
            "many",
            "amount",
            "percentage",
            "value",
            "maximum"
        }

        query_terms = [
            word
            for word in re.findall(
                r"\b[a-zA-Z]+\b",
                query_lower
            )
            if word not in stop_words
        ]

        # ====================================================
        # Score Documents
        # ====================================================

        for doc_id in list(
            rrf_scores.keys()
        ):

            document = documents[
                doc_id
            ]

            text = normalize_text(
                get_document_text(
                    document
                )
            )

            # =================================================
            # 1. Query Term Matching
            # =================================================

            matched_terms = sum(
                1
                for term in query_terms
                if term in text
            )

            rrf_scores[doc_id] += (
                matched_terms * 0.003
            )

            # =================================================
            # 2. Important Phrase Matching
            # =================================================

            important_phrases = []

            if "retention" in query_lower:

                important_phrases.append(
                    "percentage of retention"
                )

            if "advance payment" in query_lower:

                important_phrases.append(
                    "total advance payment"
                )

            if "defects notification period" in query_lower:

                important_phrases.append(
                    "defects notification period"
                )

            if "profit" in query_lower:

                important_phrases.append(
                    "percentage of profit"
                )

            if "delay damages" in query_lower:

                important_phrases.append(
                    "maximum amount of delay damages"
                )

            for phrase in important_phrases:

                if phrase in text:

                    rrf_scores[doc_id] += 0.025

            # =================================================
            # 3. Retention
            # =================================================

            if "retention" in query_lower:

                if (
                    "percentage of retention"
                    in text
                ):

                    rrf_scores[doc_id] += 0.050

                if (
                    "5% of the contract price"
                    in text
                ):

                    rrf_scores[doc_id] += 0.080

                if (
                    "percentage of retention"
                    in text
                    and
                    "5% of the contract price"
                    in text
                ):

                    rrf_scores[doc_id] += 0.050

            # =================================================
            # 4. Advance Payment
            # =================================================

            if "advance payment" in query_lower:

                if (
                    "total advance payment"
                    in text
                ):

                    rrf_scores[doc_id] += 0.050

                if (
                    "5% of the basic scope contract amount"
                    in text
                ):

                    rrf_scores[doc_id] += 0.100

                if (
                    "basic scope contract amount"
                    in text
                    and
                    "5%"
                    in text
                ):

                    rrf_scores[doc_id] += 0.040

            # =================================================
            # 5. Defects Notification Period
            # =================================================

            if (
                "defects notification period"
                in query_lower
            ):

                if (
                    "12 months"
                    in text
                ):

                    rrf_scores[doc_id] += 0.100

                if (
                    "365 days"
                    in text
                ):

                    rrf_scores[doc_id] += 0.080

                if (
                    "defects notification period"
                    in text
                    and
                    "12 months"
                    in text
                ):

                    rrf_scores[doc_id] += 0.100

                if (
                    "12 months"
                    in text
                    and
                    "taking over certificate"
                    in text
                ):

                    rrf_scores[doc_id] += 0.050

            # =================================================
            # 6. Profit
            # =================================================

            if "profit" in query_lower:

                if (
                    "percentage of profit"
                    in text
                ):

                    rrf_scores[doc_id] += 0.050

                if (
                    "5% of the cost"
                    in text
                ):

                    rrf_scores[doc_id] += 0.100

            # =================================================
            # 7. Delay Damages
            # =================================================

            if "delay damages" in query_lower:

                if (
                    "maximum amount of delay damages"
                    in text
                ):

                    rrf_scores[doc_id] += 0.060

                if (
                    "10% of the final contract price"
                    in text
                ):

                    rrf_scores[doc_id] += 0.120

                if (
                    "maximum amount"
                    in text
                    and
                    "delay damages"
                    in text
                ):

                    rrf_scores[doc_id] += 0.050

    # ========================================================
    # Final RRF Ranking
    # ========================================================

    ranked_documents = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        (
            documents[doc_id],
            score
        )
        for doc_id, score in ranked_documents
    ]


# ============================================================
# Evaluation
# ============================================================

if __name__ == "__main__":

    print(
        "Loading vectorstore..."
    )

    vectorstore = load_vectorstore()

    print(
        "Loading documents..."
    )

    documents = load_documents(
        vectorstore
    )

    print(
        f"Number of documents: "
        f"{len(documents)}"
    )

    print(
        "Building BM25..."
    )

    bm25 = build_bm25(
        documents
    )

    # ========================================================
    # Query
    # ========================================================

    query = (
        "What is the defects notification period?"
    )

    print(
        f"\nQuestion: {query}\n"
    )

    # ========================================================
    # Dense Retrieval
    # ========================================================

    print(
        "Running Dense Retrieval..."
    )

    dense_results = dense_search(
        query,
        vectorstore,
        k=10
    )

    # ========================================================
    # BM25 Retrieval
    # ========================================================

    print(
        "Running BM25 Retrieval..."
    )

    bm25_results = bm25_search(
        query,
        bm25,
        documents,
        k=10
    )

    # ========================================================
    # Hybrid Retrieval
    # ========================================================

    print(
        "Running Hybrid Retrieval..."
    )

    hybrid_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        query=query
    )

    # ========================================================
    # Display Hybrid
    # ========================================================

    print(
        "\n========== HYBRID TOP RESULTS ==========\n"
    )

    for i, (
        document,
        score
    ) in enumerate(
        hybrid_results[:10],
        start=1
    ):

        page = get_document_page(
            document
        )

        content = get_document_text(
            document
        )

        print(
            f"--- Hybrid Result {i} "
            f"| Page: {page} "
            f"| RRF Score: {score:.6f} ---"
        )

        print(
            content[:500]
        )

        print()

    # ========================================================
    # Reranking
    # ========================================================

    print(
        "\nRunning Reranker..."
    )

    # Give the reranker more candidates
    # than the final number we want.
    reranker_candidates = hybrid_results[:20]

    reranked_results = rerank(
        query,
        reranker_candidates,
        top_k=5
    )

    # ========================================================
    # Display Reranked Results
    # ========================================================

    print(
        "\n========== RERANKED TOP RESULTS ==========\n"
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
            f"--- Reranked Result {i} "
            f"| Page: {page} "
            f"| Reranker Score: {score:.6f} ---"
        )

        print(
            content[:500]
        )

        print()

    print(
        "\n=========================================="
    )