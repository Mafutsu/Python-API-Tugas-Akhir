import streamlit as st

from bm25_engine import BM25Engine
from sbert_engine import SBERTSearchEngine
from fusion import HybridSearchEngine

# =========================
# CONFIG
# =========================
DATA_PATH = "UU_Pidana_clean_updated3.json"

INDEX_PATH = "index/faiss_index.bin"

METADATA_PATH = "index/metadata.pkl"

# =========================
# LOAD ENGINES
# =========================
@st.cache_resource
def load_engines():

    bm25_engine = BM25Engine(
        DATA_PATH
    )

    sbert_engine = SBERTSearchEngine(
        data_path=DATA_PATH,
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH
    )

    hybrid_engine = HybridSearchEngine(
        bm25_engine,
        sbert_engine
    )

    return hybrid_engine


engine = load_engines()