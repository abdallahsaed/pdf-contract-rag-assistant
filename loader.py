"""
loader.py
Responsible for reading a PDF file and converting it into Document objects
ready to be passed to chunker.py afterward

Uses pdfplumber's plain text extraction. We tried converting detected
tables into Markdown, but pdfplumber's table detection was inconsistent
for visually complex tables (merged cells, uneven columns), producing
output that was harder to parse than the raw text. The raw text layout
preserves row/column order well enough for the LLM when paired with a
clear prompt (see generator.py).
"""

import pdfplumber
from langchain_core.documents import Document


def load_pdf(file_path: str):
    """
    Reads a PDF and returns a list of Documents (one per page).

    Parameters:
        file_path: the full path to the PDF file

    Returns:
        list of Document objects, each containing:
            - page_content: the page's raw extracted text
            - metadata: data such as page number and file name
    """
    documents = []

    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            documents.append(
                Document(
                    page_content=page_text,
                    metadata={"page": page_number, "source": file_path},
                )
            )

    return documents


if __name__ == "__main__":
    # Test: replace with the path to any contract PDF
    test_path = "data/your_contract.pdf"
    docs = load_pdf(test_path)
    print(f"Number of pages loaded: {len(docs)}")
    if docs:
        print("\n--- First 300 characters of the first page ---")
        print(docs[0].page_content[:300])
