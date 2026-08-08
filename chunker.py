"""
chunker.py
Splits PDF text into small chunks suitable for embedding and retrieval.

Why do we chunk in the first place?
- LLMs and embedding models don't have an infinite context window.
- If we passed the entire PDF as a single chunk, retrieval would be
  inaccurate (it would return the whole page instead of the specific
  sentence actually needed).
- A smaller size (e.g. 500-1000 characters) gives higher retrieval
  accuracy, but if too small it can lose the context around a sentence.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(documents, chunk_size: int = 800, chunk_overlap: int = 100):
    """
    Splits a list of documents (coming from the PDF loader) into chunks.

    Parameters:
        documents: list of LangChain Document objects (from loader.py)
        chunk_size: size of each chunk in characters (default 800 —
                    can be tuned later)
        chunk_overlap: how much adjacent chunks overlap with each other
                       (so information isn't cut in half between chunks)

    Returns:
        list of chunked Document objects
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        # Order matters: it first tries to split at a paragraph break,
        # then a line break, then a sentence, and so on down to the
        # character itself as a last resort fallback
    )

    chunks = splitter.split_documents(documents)
    return chunks


if __name__ == "__main__":
    from loader import load_pdf

    # Load the same real contract we tested in loader.py
    docs = load_pdf("data/Contract RI8 (BP#03) - ABEC.pdf")

    # Split it into chunks using the default size (800 chars, 100 overlap)
    chunks = chunk_text(docs)

    print(f"Number of original pages: {len(docs)}")
    print(f"Number of chunks after splitting: {len(chunks)}")

    print("\n--- First 3 chunks as a sample ---")
    for i, c in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} (from page {c.metadata.get('page', '?')}) ---")
        print(c.page_content)