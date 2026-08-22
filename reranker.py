"""
reranker.py
============
Step 8: cross-encoder reranking, the final stage of the search pipeline.

Full pipeline: hybrid retrieval (Step 6) -> LambdaMART ranking (Step 7) ->
cross-encoder reranking (this module).

A cross-encoder scores each (query, candidate) pair jointly (rather than
comparing independent embeddings), which is more accurate but too slow to
run over the full catalog -- so it only reranks the small top-K pool that
LambdaMART has already narrowed things down to.
"""

from dataclasses import dataclass

import pandas as pd
from sentence_transformers import CrossEncoder

import config
from learning_to_rank import LambdaMARTModel, rank_candidates


@dataclass
class RerankerModel:
    cross_encoder: CrossEncoder
    lambdamart: LambdaMARTModel


def load_reranker(lambdamart: LambdaMARTModel, model_name: str = config.CROSS_ENCODER_MODEL) -> RerankerModel:
    return RerankerModel(cross_encoder=CrossEncoder(model_name), lambdamart=lambdamart)


def colbert_rerank(
    reranker: RerankerModel,
    query_text: str,
    lambdamart_ranked_df: pd.DataFrame,
    top_k: int = config.FINAL_TOP_K,
    rerank_pool: int = config.RERANK_POOL_SIZE,
) -> pd.DataFrame:
    """Rerank the top ``rerank_pool`` LambdaMART candidates with a
    cross-encoder, returning the best ``top_k``."""
    pool = lambdamart_ranked_df.head(rerank_pool).reset_index(drop=True)
    if pool.empty:
        return pool

    doc_texts = (pool["track"].astype(str) + " " + pool["artist"].astype(str)).tolist()
    pairs = [[query_text, d] for d in doc_texts]

    scores = reranker.cross_encoder.predict(pairs)  # higher = more relevant

    pool["colbert_score"] = scores
    reranked = pool.sort_values("colbert_score", ascending=False).reset_index(drop=True)

    return reranked.head(top_k)


def search(
    reranker: RerankerModel,
    query_text: str,
    top_k: int = config.FINAL_TOP_K,
    lambdamart_pool: int = config.LTR_CANDIDATE_TOP_K,
    rerank_pool: int = config.RERANK_POOL_SIZE,
) -> pd.DataFrame:
    """Full pipeline: hybrid retrieval -> LambdaMART ranking -> cross-encoder reranking."""
    lm_ranked = rank_candidates(reranker.lambdamart, query_text, top_k=lambdamart_pool)
    final = colbert_rerank(reranker, query_text, lm_ranked, top_k=top_k, rerank_pool=rerank_pool)
    return final[["track", "artist", "lambdamart_score", "colbert_score"]]


if __name__ == "__main__":
    from data_preprocessing import load_datasets
    from hybrid_search import build_hybrid_search_index
    from learning_to_rank import build_all_training_queries, build_ranking_dataframe, train_lambdamart

    datasets = load_datasets()
    search_index = build_hybrid_search_index(datasets.catalog_df)
    ranking_df = build_ranking_dataframe(datasets.catalog_df, datasets.listening_df)
    all_queries = build_all_training_queries(ranking_df, datasets.merged_df)
    lambdamart = train_lambdamart(search_index, ranking_df, all_queries)

    reranker = load_reranker(lambdamart)

    print("\n--- Full pipeline: hybrid -> LambdaMART -> cross-encoder ---")
    print(search(reranker, "sad romantic songs", top_k=10))
    print(search(reranker, "songs like shape of you", top_k=10))
