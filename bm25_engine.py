import json
from rank_bm25 import BM25Okapi

from utils import preprocess


class BM25Engine:

    def __init__(self, data_path):

        self.data_path = data_path

        self.documents = []

        self.doc_lookup = {}

        self.corpus = []

        self.bm25 = None

        self.load_documents()

        self.build_index()

    # LOAD DATASET
    def load_documents(self):

        with open(self.data_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        # mapping id -> document
        self.doc_lookup = {
            doc["id"]: doc
            for doc in self.documents
        }

    # BUILD BM25 INDEX
    def build_index(self):

        corpus = []

        for doc in self.documents:

            clean_text = preprocess(doc["content"])

            tokens = clean_text.split()

            corpus.append(tokens)

        self.corpus = corpus

        self.bm25 = BM25Okapi(corpus)

    # SEARCH BM25
    def search(self, query, top_k=20):

        query = preprocess(query)

        tokenized_query = query.split()

        scores = self.bm25.get_scores(tokenized_query)

        max_score = max(scores)

        # hanya ambil dokumen dengan skor >= 10% dari skor tertinggi
        threshold = max_score * 0.1

        results = []

        for idx, score in enumerate(scores):

            if score < threshold or score == 0:
                continue
            else:
                doc = self.documents[idx]
                print(
                    f"[ACCEPT] "
                    f"{doc['id']} | "
                    f"score={score:.4f}"
                )

                

                results.append({
                    "id": doc["id"],
                    "nomor": doc.get("nomor"),
                    "tahun": doc.get("tahun"),
                    "pasal": doc.get("pasal"),
                    "ayat": doc.get("ayat"),
                    "content": doc["content"],
                    "hyperlink": doc.get("hyperlink", []),
                    "score": float(score)
                })

        # sort descending
        results = sorted(
            results,
            key=lambda x: x["score"],
            reverse=True
        )

        return results[:top_k]