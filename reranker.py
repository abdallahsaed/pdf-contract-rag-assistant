from sentence_transformers import CrossEncoder


# ============================================================
# Load Reranker Model
# ============================================================

model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ============================================================
# Rerank Documents
# ============================================================

def rerank(
    query,
    results,
    top_k=5
):
    """
    Rerank retrieved documents using a Cross-Encoder.

    Parameters:
        query: user question
        results: list of (document, score)
        top_k: number of final documents

    Returns:
        list of (document, reranker_score)
    """

    if not results:
        return []

    pairs = []

    for document, _ in results:

        if isinstance(document, dict):

            text = document.get(
                "text",
                ""
            )

        else:

            text = document.page_content

        pairs.append(
            [
                query,
                text
            ]
        )

    # ========================================================
    # Predict Cross-Encoder Scores
    # ========================================================

    scores = model.predict(
        pairs
    )

    # ========================================================
    # Combine Documents + Scores
    # ========================================================

    reranked = list(
        zip(
            results,
            scores
        )
    )

    # ========================================================
    # Sort by Reranker Score
    # ========================================================

    reranked.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # ========================================================
    # Return Top K
    # ========================================================

    return [
        (
            document,
            float(score)
        )
        for (
            (document, _),
            score
        ) in reranked[:top_k]
    ]