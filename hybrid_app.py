# import streamlit as st

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
# @st.cache_resource
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
# # =========================
# # UI
# # =========================
# st.title(
#     "Hybrid Search Engine Hukum"
# )

# query = st.text_input(
#     "Masukkan query pencarian:"
# )

# if st.button("Search"):

#     results = engine.search(query)

#     st.success(
#         f"Ditemukan {len(results)} hasil"
#     )

#     for result in results:

#         st.markdown("---")

#         title = f"UU No.{result.get('nomor')} Tahun {result.get('tahun')}"

#         pasal = result.get("pasal")
#         ayat = result.get("ayat")

#         if pasal not in [None, "", "-"]:
#             title += f" Pasal {pasal}"

#         if ayat not in [None, "", "-"]:
#             title += f" Ayat {ayat}"

#         st.subheader(title)

#         st.write(
#             f"Hybrid Score: "
#             f"{round(result['hybrid_score'], 4)}"
#         )

#         st.write(
#             f"BM25: "
#             f"{round(result['bm25_score'], 4)}"
#         )

#         st.write(
#             f"SBERT: "
#             f"{round(result['sbert_score'], 4)}"
#         )

#         st.write(result["content"])

#         # =========================
#         # HYPERLINK
#         # =========================
#         if result["hyperlink"]:

#             st.markdown(
#                 "### Pasal Terkait"
#             )

#             for link in result["hyperlink"]:

#                 linked_doc = (
#                     engine
#                     .bm25_engine
#                     .doc_lookup
#                     .get(link)
#                 )

#                 if linked_doc:

#                     with st.expander(
#                         f"🔗 {link}"
#                     ):

#                         st.write(
#                             linked_doc[
#                                 "content"
#                             ]
#                         )