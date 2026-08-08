import streamlit as st
from retriever import load_vectorstore, search as dense_search
from bm25_retriever import load_documents, build_bm25, search as bm25_search
from hybrid_retriever import reciprocal_rank_fusion
from reranker import rerank
from generator import generate_answer


st.set_page_config(page_title="Contract AI", page_icon="🏗️")

# --------------------------------------------------------
# Light visual touch: soft gradient background instead of
# the plain default, and a subtle engineering-blueprint tone.
# Purely cosmetic — no effect on the pipeline logic below.
# --------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f4f7fa 0%, #eef2f6 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏗️ Contract AI — Ask Your Contract")
st.caption("RAG system with Hybrid Retrieval (BM25 + Dense) + Cross-Encoder Reranking")
st.caption("Built and tested on construction contracts — works with any PDF contract")


@st.cache_resource
def load_resources():
    vectorstore = load_vectorstore()
    documents = load_documents(vectorstore)
    bm25 = build_bm25(documents)
    return vectorstore, documents, bm25


vectorstore, documents, bm25 = load_resources()

query = st.text_input("Ask a question about the contract:", placeholder="What is the defects notification period?")

if st.button("Ask") and query:
    with st.spinner("Searching and analyzing..."):
        dense_results = dense_search(query, vectorstore, k=10)
        bm25_results = bm25_search(query, bm25, documents, k=10)
        hybrid_results = reciprocal_rank_fusion(dense_results, bm25_results)
        reranked_results = rerank(query, hybrid_results, top_k=5)
        result = generate_answer(query, reranked_results)

    st.subheader("Answer")
    st.write(result["answer"])

    st.caption(f"Source: page {result['page']}")

    with st.expander("View reference text (Evidence)"):
        st.write(result["evidence"])