from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from bm25_engine import BM25Engine
from sbert_engine import SBERTSearchEngine
from fusion import HybridSearchEngine
import json

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

with open(DATA_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# mapping id -> data
law_map = {}

for item in dataset:
    law_map[str(item["id"])] = item

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

    for result in results:

        related_laws = []

        hyperlinks = result.get("hyperlink", [])

        if hyperlinks:

            for law_id in hyperlinks:

                law_id = str(law_id)

                if law_id in law_map:

                    related = law_map[law_id]

                    related_laws.append({

                        "nomor": related.get("nomor"),

                        "tahun": related.get("tahun"),

                        "pasal": related.get("pasal"),

                        "ayat": related.get("ayat"),

                        "content": related.get("content")

                    })

        result["related_laws"] = related_laws

    return results