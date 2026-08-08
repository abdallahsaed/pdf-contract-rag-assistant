# Contract AI — RAG-Based Contract Question Answering

A local, fully offline Retrieval-Augmented Generation (RAG) system that answers natural-language questions about any PDF contract, using hybrid retrieval, cross-encoder reranking, and a locally-run LLM. It was built and tested end-to-end on a FIDIC-based construction contract, but the pipeline itself is document-agnostic — point it at any contract PDF and rebuild the vector store.

This project was built as a hands-on foundation before developing **ContractAI**, a larger multi-agent system for comparing construction contracts against FIDIC standards and flagging contractor-risk clauses.

## Architecture

```
PDF Contract
    │
    ▼
Loader (pdfplumber — plain text extraction)
    │
    ▼
Chunker (RecursiveCharacterTextSplitter, 800 chars / 100 overlap)
    │
    ▼
Embedder (sentence-transformers/all-MiniLM-L6-v2 → ChromaDB)
    │
    ├── Dense Retrieval (semantic similarity search)
    │
    ├── BM25 Retrieval (keyword-based sparse search)
    │
    ▼
Hybrid Retrieval (Reciprocal Rank Fusion of dense + BM25 results)
    │
    ▼
Cross-Encoder Reranker (reorders top candidates by relevance to the query)
    │
    ▼
Generator (Ollama — Llama 3.2 1B, local inference)
    │
    ▼
Grounded Answer + Source Page + Evidence
```

## Why Hybrid Retrieval + Reranking?

Dense (embedding-based) retrieval is good at matching *meaning*, but can miss exact terminology (e.g. a specific clause number or defined term). BM25 is good at exact keyword matches but misses semantic similarity. Combining both with Reciprocal Rank Fusion, then reordering the merged results with a cross-encoder reranker, produces noticeably more accurate retrieval than either method alone — this is the same approach used in production-grade RAG systems, not just introductory tutorials.

## Tech Stack

| Component | Tool |
|---|---|
| PDF parsing | `pdfplumber` (plain text extraction per page) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, HuggingFace) |
| Vector store | ChromaDB (local, persisted to disk) |
| Sparse retrieval | BM25 (`rank_bm25`) |
| Fusion | Reciprocal Rank Fusion (document-agnostic — ranks only, no hardcoded logic) |
| Reranking | Cross-encoder model |
| LLM generation | Ollama, running `llama3.2:1b` locally |
| UI | Streamlit |

Everything runs **fully offline and locally** — no paid APIs (e.g. OpenAI) are used anywhere in the pipeline.

**Note on model choice:** the pipeline was tested with `llama3.2:1b` rather than the larger `3b` variant, since the development GPU (Quadro M1200) does not have enough VRAM to run `3b` without out-of-memory errors. Swapping in a larger model is a one-line change in `generator.py` on more capable hardware.

## Project Structure

```
pdf-rag-mvp/
├── data/                   # source PDF contract
├── loader.py               # PDF → LangChain Documents (via pdfplumber)
├── chunker.py               # Documents → chunks
├── embedder.py               # chunks → embeddings → ChromaDB
├── retriever.py               # dense (semantic) retrieval
├── bm25_retriever.py           # sparse (keyword) retrieval
├── hybrid_retriever.py          # Reciprocal Rank Fusion of dense + BM25
├── reranker.py                # cross-encoder reranking
├── generator.py               # LLM answer generation via Ollama
├── pipeline.py                # orchestrates the full flow (CLI)
├── evaluation.py               # retrieval/answer evaluation
├── app.py                    # Streamlit UI
└── requirements.txt
```

## Setup

### 1. Install Ollama and pull the model
```
# Download Ollama from https://ollama.com/download
ollama pull llama3.2:1b
```

### 2. Set up the Python environment
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Build the vector store (first run only)
```
python embedder.py
```

### 4. Run via CLI
```
python pipeline.py
```

### 5. Or run the web UI
```
streamlit run app.py
```

## Example

Ask a question in plain English (e.g. *"What is the defects notification period?"* or *"What is the penalty for late payment?"*), and the system returns a concise, grounded answer along with the source page and the underlying contract text it was drawn from — so every answer can be traced back and verified.

## Known Limitations

- **Multi-column tables can confuse the LLM.** Contracts often contain pricing or schedule tables with several parallel numeric columns (e.g. per-unit vs. combined totals) and ambiguous row labels (e.g. a "Grand Total" row vs. a sub-total row). Because `pdfplumber` extracts these as plain or lightly-structured text, the LLM can sometimes pick the wrong column or conflate a subtotal with a grand total — even with explicit prompt rules. This is a known constraint of small, locally-run LLMs on visually complex tabular layouts, not a retrieval failure (the correct table is reliably retrieved; the generation step is where the error occurs).
- **PDF table extraction is not fully structured.** `pdfplumber`'s automatic table detection is inconsistent on visually complex tables (merged cells, uneven column counts), so the pipeline currently relies on plain text extraction rather than reconstructing tables into a guaranteed-accurate structured format.
- **Smaller local model.** Running `llama3.2:1b` (due to GPU VRAM limits) trades some reasoning depth for the ability to run fully offline on modest hardware. A larger model would likely improve accuracy on harder questions.
- **Single-document scope:** The pipeline processes one contract at a time. It is document-agnostic (any PDF can be loaded by changing the file path and rebuilding the vector store), but no multi-document or cross-document comparison logic exists yet.
- **No standard-contract comparison:** The system answers questions about the loaded contract; it does not yet compare clauses against a reference standard (e.g. the FIDIC Red Book) to flag deviations. That capability is the focus of the next project, ContractAI.

## Roadmap → ContractAI

This project served as a proof of concept for the core RAG pipeline. The next stage, **ContractAI**, extends this foundation into a multi-agent system that:
- Compares an owner-drafted contract against a FIDIC baseline, clause by clause
- Classifies each deviation as normal industry variance vs. a contractor-risk item
- Focuses on the clause categories most likely to carry risk (Claims, EOT, Variations, Liquidated Damages, Termination, Retention, Payment Terms, Force Majeure)
- Outputs a structured risk report (Reference / Clause / Standard Position / Current Contract Position / Risk Assessment / Reason)

## Author

Abdallah Saed — Computational Science & AI, Zewail City of Science and Technology
