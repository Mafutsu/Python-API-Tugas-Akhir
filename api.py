from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from bm25_engine import BM25Engine
from sbert_engine import SBERTSearchEngine
from fusion import HybridSearchEngine

app = FastAPI()

# IZINKAN LARAVEL AKSES API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = "UU_Pidana_clean_updated3.json"

INDEX_PATH = "index/faiss_index.bin"

METADATA_PATH = "index/metadata.pkl"

# LOAD ENGINE
bm25_engine = BM25Engine(DATA_PATH)

sbert_engine = SBERTSearchEngine(
    data_path=DATA_PATH,
    index_path=INDEX_PATH,
    metadata_path=METADATA_PATH
)

engine = HybridSearchEngine(
    bm25_engine,
    sbert_engine
)

@app.get("/search")
def search(query: str):

    results = engine.search(query)

    return results