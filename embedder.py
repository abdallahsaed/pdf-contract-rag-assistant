"""
embedder.py
Responsible for converting chunks into embeddings and storing them in ChromaDB.

Why ChromaDB specifically?
- A local, lightweight vector database that persists directly to disk.
- No separate server to run, unlike some other solutions — it works directly from the code.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def build_vectorstore(chunks, persist_directory: str = "chroma_db"):
    """
    Converts chunks into embeddings and stores them in ChromaDB on disk.

    Parameters:
        chunks: list of Document objects (from chunker.py)
        persist_directory: the directory where the database will be stored on disk

    Returns:
        A Chroma vectorstore object ready to be searched
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
    )

    return vectorstore


if __name__ == "__main__":
    from loader import load_pdf
    from chunker import chunk_text

    docs = load_pdf("data/Contract RI8 (BP#03) - ABEC.pdf")
    chunks = chunk_text(docs)

    print(f"Number of chunks to be converted into embeddings: {len(chunks)}")
    print("Building the vectorstore... (first run takes longer to download the model)")

    vectorstore = build_vectorstore(chunks)

    print("Done! The vectorstore has been built and stored in the chroma_db folder")