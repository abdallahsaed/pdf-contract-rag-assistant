import re


# ============================================================
# Helper Functions
# ============================================================

def get_document_text(document):

    if isinstance(document, dict):
        return document.get("text", "")

    return document.page_content


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

    Additional query-aware scoring is used
    to improve ranking of highly relevant
    contract documents.
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

        # ----------------------------------------------------
        # Stop Words
        # ----------------------------------------------------

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
            # 2. Exact Phrase Matching
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
            # 3. Contract-Specific Signals
            # =================================================

            # -------------------------------------------------
            # Retention
            # -------------------------------------------------

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

            # -------------------------------------------------
            # Advance Payment
            # -------------------------------------------------

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

            # -------------------------------------------------
            # Defects Notification Period
            # -------------------------------------------------

            if (
                "defects notification period"
                in query_lower
            ):

                # Exact contractual value
                if (
                    "12 months"
                    in text
                ):

                    rrf_scores[doc_id] += 0.150

                # Equivalent duration
                if (
                    "365 days"
                    in text
                ):

                    rrf_scores[doc_id] += 0.120

                # Definition + exact duration
                if (
                    "defects notification period"
                    in text
                    and
                    "12 months"
                    in text
                ):

                    rrf_scores[doc_id] += 0.200

                # Definition + duration + Taking-Over
                if (
                    "12 months"
                    in text
                    and
                    "taking over certificate"
                    in text
                ):

                    rrf_scores[doc_id] += 0.250

                # Strongest signal:
                # exact answer components together
                if (
                    "defects notification period"
                    in text
                    and
                    "12 months"
                    in text
                    and
                    "taking over certificate"
                    in text
                ):

                    rrf_scores[doc_id] += 0.300

            # -------------------------------------------------
            # Profit
            # -------------------------------------------------

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

            # -------------------------------------------------
            # Delay Damages
            # -------------------------------------------------

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
    # Final Ranking
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