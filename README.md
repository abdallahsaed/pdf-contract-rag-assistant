# Contract AI — RAG-Based Contract Question Answering

A local, fully offline Retrieval-Augmented Generation (RAG) system that answers questions about a construction contract (FIDIC-based) using hybrid retrieval, cross-encoder reranking, and a locally-run LLM.

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
Generator (Ollama — Llama 3.2 3B, local inference)
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
| Fusion | Reciprocal Rank Fusion |
| Reranking | Cross-encoder model |
| LLM generation | Ollama, running `llama3.2:3b` locally |
| UI | Streamlit |

Everything runs **fully offline and locally** — no paid APIs (e.g. OpenAI) are used anywhere in the pipeline.

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
ollama pull llama3.2:3b
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

**Question:** *What is the defects notification period?*

**Answer:** The defects notification period is stated in Clause 11.1 as being in the condition required by the Contract (fair wear and tear excepted) by the expiry date of the relevant Defects Notification Period or as soon as practicable thereafter, which means it can be up to two years after the expiry date of the Defects Notification Period for the Works or Section.

**Source:** page 52

## Known Limitations

- **Multi-column pricing tables can confuse the LLM.** The test contract contains tables with parallel "One Block" / "Two Blocks" pricing columns, and row labels like "Grand Total" vs "Total Optional" vs "Basic Scope (Stage 1)". When `pdfplumber` extracts these tables as plain or lightly-structured text, `llama3.2:3b` sometimes picks the wrong column, or conflates the "Grand Total" row with a sub-scope total — even after adding explicit prompt rules and few-shot examples distinguishing the columns and row types. This is a known constraint of small, locally-run LLMs on visually complex tabular layouts, not a retrieval failure (the correct table is reliably retrieved; the generation step is where the error occurs).
- **PDF table extraction is not fully structured.** `pdfplumber`'s automatic table detection was inconsistent on visually complex tables (merged cells, uneven column counts), so the pipeline currently relies on plain text extraction rather than reconstructing tables into a guaranteed-accurate structured format.
- **Single-document scope:** The system is currently built and tested on one contract. The pipeline itself is generic (any PDF can be loaded by changing the file path and rebuilding the vector store), but no multi-document or cross-document comparison logic exists yet.
- **No standard-contract comparison:** The system answers questions about the loaded contract; it does not yet compare clauses against a reference standard (e.g. the FIDIC Red Book) to flag deviations. That capability is the focus of the next project, ContractAI.

## Roadmap → ContractAI

This project served as a proof of concept for the core RAG pipeline. The next stage, **ContractAI**, extends this foundation into a multi-agent system that:
- Compares an owner-drafted contract against a FIDIC baseline, clause by clause
- Classifies each deviation as normal industry variance vs. a contractor-risk item
- Focuses on the clause categories most likely to carry risk (Claims, EOT, Variations, Liquidated Damages, Termination, Retention, Payment Terms, Force Majeure)
- Outputs a structured risk report (Reference / Clause / Standard Position / Current Contract Position / Risk Assessment / Reason)

## Author

Abdallah Saed — Computational Science & AI, Zewail City of Science and Technology
