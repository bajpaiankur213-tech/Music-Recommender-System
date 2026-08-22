"""
hybrid_search.py
==================
Step 6: hybrid retrieval over the catalog, combining:

  - BM25 (lexical/sparse retrieval, via rank_bm25)
  - Dense embedding similarity (via sentence-transformers + ChromaDB)

fused with Reciprocal Rank Fusion (RRF), and driven by the query
understanding layer (Step 5) for mood-synonym expansion before retrieval.
"""

from dataclasses import dataclass

import chromadb
import numpy as np
import pandas as pd
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import config
from query_understanding import build_known_entities, rectify_query


@dataclass
class HybridSearchIndex:
    catalog_df: pd.DataFrame
    bm25: BM25Okapi
    embedding_model: SentenceTransformer
    collection: "chromadb.Collection"
    known_tracks: tuple
    known_artists: tuple


def _build_search_text(catalog_df: pd.DataFrame) -> pd.Series:
    return (
        catalog_df["track_name"] + " " + catalog_df["artist_name"] + " "
        + catalog_df["genre"] + " " + catalog_df["tags"]
    ).str.lower()


def build_bm25_index(search_text: pd.Series) -> BM25Okapi:
    tokenized_corpus = [t.split() for t in search_text.tolist()]
    return BM25Okapi(tokenized_corpus)


def build_dense_index(
    search_text: pd.Series,
    embedding_model: SentenceTransformer,
    persist_dir: str = config.CHROMA_PERSIST_DIR,
    collection_name: str = config.CHROMA_COLLECTION_NAME,
    batch_size: int = config.EMBEDDING_BATCH_SIZE,
) -> "chromadb.Collection":
    """Embed the catalog in batches and load it into a Chroma collection."""
    client = chromadb.Client(Settings(persist_directory=persist_dir))
    collection = client.get_or_create_collection(name=collection_name)

    documents = search_text.tolist()
    ids = [str(i) for i in range(len(documents))]

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        batch_embeddings = embedding_model.encode(batch_docs)

        collection.add(documents=batch_docs, embeddings=batch_embeddings.tolist(), ids=batch_ids)

    return collection


def build_hybrid_search_index(catalog_df: pd.DataFrame) -> HybridSearchIndex:
    """One-time setup: build the BM25 index and the dense embedding index
    over the catalog. Expensive -- call once and reuse the returned index."""
    search_text = _build_search_text(catalog_df)

    bm25 = build_bm25_index(search_text)
    embedding_model = SentenceTransformer(config.DENSE_EMBEDDING_MODEL)
    collection = build_dense_index(search_text, embedding_model)

    known_tracks, known_artists = build_known_entities(catalog_df)

    return HybridSearchIndex(
        catalog_df=catalog_df,
        bm25=bm25,
        embedding_model=embedding_model,
        collection=collection,
        known_tracks=tuple(known_tracks),
        known_artists=tuple(known_artists),
    )


def bm25_search(index: HybridSearchIndex, query: str, top_k: int = config.HYBRID_SEARCH_TOP_K):
    scores = index.bm25.get_scores(query.split())
    ranked_idx = np.argsort(scores)[::-1][:top_k]
    return [(str(i), rank + 1, scores[i]) for rank, i in enumerate(ranked_idx)]


def dense_search(index: HybridSearchIndex, query: str, top_k: int = config.HYBRID_SEARCH_TOP_K):
    res = index.collection.query(query_texts=[query], n_results=top_k)
    ids = res["ids"][0]
    distances = res["distances"][0]
    return [(doc_id, rank + 1, distances[rank]) for rank, doc_id in enumerate(ids)]


def reciprocal_rank_fusion(results: list, k: int = config.RRF_K):
    """Combine multiple ranked result lists into one fused ranking.
    Each result list is ``[(doc_id, rank, raw_score), ...]``; ``raw_score``
    is unused here -- RRF only cares about rank position, which makes it
    robust to BM25 and cosine-distance scores living on different scales."""
    scores = {}
    for result_list in results:
        for doc_id, rank, _ in result_list:
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(
    index: HybridSearchIndex, query: str, top_k: int = config.HYBRID_SEARCH_TOP_K
) -> pd.DataFrame:
    """Full retrieval stage: NLU-expand the query, run BM25 + dense search,
    fuse with RRF, and return the top-k tracks with their fused score."""
    nlu_result = rectify_query(query, index.known_tracks, index.known_artists)
    expanded_query = " ".join(nlu_result["expansion"]) if nlu_result["expansion"] else query

    bm25_results = bm25_search(index, expanded_query, top_k=top_k)
    dense_results = dense_search(index, expanded_query, top_k=top_k)

    fused = reciprocal_rank_fusion([bm25_results, dense_results])

    rows = []
    for doc_id, score in fused[:top_k]:
        row = index.catalog_df.iloc[int(doc_id)]
        rows.append({"track": row["track_name"], "artist": row["artist_name"], "score": score})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from data_preprocessing import load_datasets

    datasets = load_datasets()
    index = build_hybrid_search_index(datasets.catalog_df)

    print(hybrid_search(index, "shape off you"))
    print(hybrid_search(index, "sad romantic songs"))
