from collections import defaultdict


class HybridSearchEngine:

    def __init__(
        self,
        bm25_engine,
        sbert_engine
    ):

        self.bm25_engine = bm25_engine

        self.sbert_engine = sbert_engine

    # ====================================
    # MIN-MAX NORMALIZATION
    # ====================================
    def min_max_normalize(
        self,
        results
    ):

        if not results:
            return results

        scores = [
            r["score"]
            for r in results
        ]

        min_score = min(scores)

        max_score = max(scores)

        # hindari division by zero
        if max_score == min_score:

            for r in results:
                r["normalized_score"] = 1.0

            return results

        for r in results:

            normalized = (
                (r["score"] - min_score)
                /
                (max_score - min_score)
            )

            r["normalized_score"] = normalized

        return results

    # ADAPTIVE WEIGHT
    def adaptive_weights(
        self,
        query
    ):

        tokens = query.split()

        if len(tokens) == 1:
            return 0.9, 0.1

        elif len(tokens) <= 3:
            return 0.3, 0.7

        elif len(tokens) <= 8:
            return 0.2, 0.8
        else:
            return 0.1, 0.9

    # HYBRID SEARCH
    def search(
        self,
        query,
        top_k=20,
        relevance_threshold=0.4
    ):

        # BM25 SEARCH
        bm25_results = self.bm25_engine.search(
            query,
            top_k=top_k
        )

        # SBERT SEARCH
        sbert_results = self.sbert_engine.search(
            query,
            top_k=top_k
        )

        # -----------------------------
        # NORMALIZATION
        # -----------------------------
        bm25_results = self.min_max_normalize(
            bm25_results
        )

        sbert_results = self.min_max_normalize(
            sbert_results
        )

        # ADAPTIVE WEIGHT
        w_bm25, w_sbert = (
            self.adaptive_weights(query)
        )

        # COMBINE RESULTS
        combined = defaultdict(
            lambda: {
                "id": None,
                "nomor": None,
                "tahun": None,
                "pasal": None,
                "ayat": None,
                "content": None,
                "hyperlink": [],
                "bm25_score": 0.0,
                "sbert_score": 0.0,
                "hybrid_score": 0.0
            }
        )

        # BM25 contribution
        for r in bm25_results:

            combined[r["id"]]["id"] = r["id"]

            combined[r["id"]]["content"] = (
                r["content"]
            )

            combined[r["id"]]["hyperlink"] = (
                r["hyperlink"]
            )

            combined[r["id"]]["bm25_score"] = (
                r["normalized_score"]
            )
            combined[r["id"]]["nomor"] = r.get("nomor")
            combined[r["id"]]["tahun"] = r.get("tahun")
            combined[r["id"]]["pasal"] = r.get("pasal")
            combined[r["id"]]["ayat"] = r.get("ayat")

        # SBERT contribution
        for r in sbert_results:

            combined[r["id"]]["id"] = r["id"]

            combined[r["id"]]["content"] = (
                r["content"]
            )

            combined[r["id"]]["hyperlink"] = (
                r["hyperlink"]
            )

            combined[r["id"]]["sbert_score"] = (
                r["normalized_score"]
            )
            combined[r["id"]]["nomor"] = r.get("nomor")
            combined[r["id"]]["tahun"] = r.get("tahun")
            combined[r["id"]]["pasal"] = r.get("pasal")
            combined[r["id"]]["ayat"] = r.get("ayat")

        # HYBRID SCORE
        final_results = []

        for _, item in combined.items():

            hybrid_score = (
                (
                    w_bm25
                    *
                    item["bm25_score"]
                )
                +
                (
                    w_sbert
                    *
                    item["sbert_score"]
                )
            )

            # citation boost
            citation_boost = min(
                len(item["hyperlink"]) * 0.005,
                0.05
            )

            final_score = (
                hybrid_score
                +
                citation_boost
            )

            item["hybrid_score"] = (
                final_score
            )

            # relevance threshold
            if final_score >= relevance_threshold:

                final_results.append(item)

        # SORT FINAL
        final_results = sorted(
            final_results,
            key=lambda x: x["hybrid_score"],
            reverse=True
        )

        return final_results[:top_k]