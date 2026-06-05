import json
import os
import pickle

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


class SBERTSearchEngine:

    def __init__(
        self,
        data_path,
        index_path,
        metadata_path,
        model_name="all-MiniLM-L6-v2"
    ):

        self.data_path = data_path

        self.index_path = index_path

        self.metadata_path = metadata_path

        self.model_name = model_name

        self.model = None

        self.index = None

        self.metadata = None

        self.doc_lookup = {}

        # load everything
        self.load_model()

        self.load_or_build_index()

    # =========================
    # LOAD MODEL
    # =========================
    def load_model(self):

        self.model = SentenceTransformer(
            self.model_name
        )

    # =========================
    # LOAD DATASET
    # =========================
    def load_dataset(self):

        with open(
            self.data_path,
            "r",
            encoding="utf-8"
        ) as f:

            documents = json.load(f)

        return documents

    # =========================
    # BUILD INDEX
    # =========================
    def build_index(self):

        documents = self.load_dataset()

        texts = []

        metadata = []

        for doc in documents:

            texts.append(doc["content"])

            metadata.append({
                "id": doc["id"],
                "nomor": doc.get("nomor"),
                "tahun": doc.get("tahun"),
                "pasal": doc.get("pasal"),
                "ayat": doc.get("ayat"),
                "content": doc["content"],
                "hyperlink": doc.get("hyperlink", [])
            })

        # embedding
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True
        )

        embeddings = np.array(
            embeddings,
            dtype=np.float32
        )

        # cosine similarity
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]

        index = faiss.IndexFlatIP(dim)

        index.add(embeddings)

        # save index
        faiss.write_index(
            index,
            self.index_path
        )

        # save metadata
        with open(
            self.metadata_path,
            "wb"
        ) as f:

            pickle.dump(metadata, f)

        self.index = index

        self.metadata = metadata

        # lookup table
        self.doc_lookup = {
            item["id"]: item
            for item in metadata
        }

    # =========================
    # LOAD INDEX
    # =========================
    def load_index(self):

        self.index = faiss.read_index(
            self.index_path
        )

        with open(
            self.metadata_path,
            "rb"
        ) as f:

            self.metadata = pickle.load(f)

        # lookup table
        self.doc_lookup = {
            item["id"]: item
            for item in self.metadata
        }

    # =========================
    # LOAD OR BUILD
    # =========================
    def load_or_build_index(self):

        index_exists = os.path.exists(
            self.index_path
        )

        metadata_exists = os.path.exists(
            self.metadata_path
        )

        if index_exists and metadata_exists:

            self.load_index()

        else:

            self.build_index()

    # =========================
    # SEARCH
    # =========================
    def search(
        self,
        query,
        top_k=20
    ):

        query_embedding = self.model.encode(
            [query]
        )

        query_embedding = np.array(
            query_embedding,
            dtype=np.float32
        )

        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            doc = self.metadata[idx]

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

        return results