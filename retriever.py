from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def load_vectorstore(persist_directory: str = "chroma_db"):
    """
    Load the existing ChromaDB vector store.
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
    )

    return vectorstore


def search(query: str, vectorstore, k: int = 5):
    """
    Retrieve the top-k most relevant chunks
    and return their similarity distances.
    """

    results = vectorstore.similarity_search_with_score(
        query,
        k=k
    )

    return results


if __name__ == "__main__":

    vectorstore = load_vectorstore()

    # Test query
    test_query = "Percentage of Retention"

    print(f"السؤال: {test_query}\n")

    results = search(
        test_query,
        vectorstore,
        k=5
    )

    for i, (doc, score) in enumerate(results):

        print(
            f"Result {i + 1} | "
            f"Page: {doc.metadata.get('page', '?')} | "
            f"Score: {score}"
        )