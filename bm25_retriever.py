from rank_bm25 import BM25Okapi
from retriever import load_vectorstore


def load_documents(vectorstore):
    """
    Load all documents/chunks from ChromaDB
    while preserving their metadata.
    """

    data = vectorstore.get(
        include=["documents", "metadatas"]
    )

    documents = []

    for text, metadata in zip(
        data["documents"],
        data["metadatas"]
    ):
        documents.append(
            {
                "text": text,
                "metadata": metadata
            }
        )

    return documents


def build_bm25(documents):
    """
    Build a BM25 index from all document chunks.
    """

    tokenized_documents = [
        document["text"].lower().split()
        for document in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)

    return bm25


def search(query, bm25, documents, k=5):
    """
    Search using BM25 and return documents
    together with their scores.
    """

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]

    results = [
        (documents[i], scores[i])
        for i in top_indices
    ]

    return results


if __name__ == "__main__":

    # Load ChromaDB
    vectorstore = load_vectorstore()

    # Load all documents
    documents = load_documents(vectorstore)

    print(f"Number of documents: {len(documents)}")

    # Build BM25
    bm25 = build_bm25(documents)

    # Test query
    test_query = "Percentage of Retention"

    print(f"\nQuestion: {test_query}\n")

    # Search
    results = search(
        test_query,
        bm25,
        documents,
        k=5
    )

    # Display results
    for i, (document, score) in enumerate(
        results,
        start=1
    ):

        page = document["metadata"].get(
            "page",
            "?"
        )

        print(
            f"--- Result {i} "
            f"| Page: {page} "
            f"| BM25 Score: {score:.4f} ---"
        )

        print(document["text"][:500])
        print()