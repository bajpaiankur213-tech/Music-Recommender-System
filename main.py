"""
main.py
========
End-to-end demo of the full pipeline, in order:

  1. data_preprocessing   - load and clean the raw CSVs
  2. content_based_recommender - TF-IDF + cosine similarity
  3. collaborative_filtering    - user-user similarity
  4. hybrid_recommender         - blend of (2) and (3)
  5. query_understanding        - NLU: intent, entities, mood expansion
  6. hybrid_search               - BM25 + dense embeddings + RRF fusion
  7. learning_to_rank            - LambdaMART re-ranking
  8. reranker                    - cross-encoder reranking (final search())

Run with:  python main.py
Requires ``Music Info.csv`` and ``User Listening History.csv`` in the
directory pointed to by ``config.DATA_DIR`` (see config.py / README).
"""

import config
from data_preprocessing import load_datasets
from content_based_recommender import fit_content_based_model, recommend
from collaborative_filtering import fit_collaborative_filtering_model, recommend_songs_with_metadata
from hybrid_recommender import fit_hybrid_model, hybrid_recommend
from hybrid_search import build_hybrid_search_index, hybrid_search
from learning_to_rank import build_all_training_queries, build_ranking_dataframe, train_lambdamart, rank_candidates
from reranker import load_reranker, search


def main():
    print("=" * 70)
    print("STEP 1: Loading and preprocessing data")
    print("=" * 70)
    datasets = load_datasets()
    print(f"catalog_df: {datasets.catalog_df.shape}, user_item_matrix: {datasets.user_item_matrix.shape}")

    print("\n" + "=" * 70)
    print("STEP 2: Content-based recommender (TF-IDF + cosine similarity)")
    print("=" * 70)
    content_model = fit_content_based_model(datasets.catalog_df)
    print(recommend(content_model, "shape of you", top_n=5))

    print("\n" + "=" * 70)
    print("STEP 3: Collaborative filtering")
    print("=" * 70)
    cf_model = fit_collaborative_filtering_model(datasets.merged_df, datasets.user_item_matrix)
    sample_user = datasets.user_item_matrix.index[0]
    print(recommend_songs_with_metadata(cf_model, datasets.music_df, sample_user).head(5))

    print("\n" + "=" * 70)
    print("STEP 4: Hybrid recommender (CF + content blend)")
    print("=" * 70)
    hybrid_model = fit_hybrid_model(cf_model, datasets.music_df)
    print(hybrid_recommend(hybrid_model, sample_user, alpha=0.5, n=5))

    print("\n" + "=" * 70)
    print("STEPS 5-6: Query understanding + hybrid search (BM25 + dense + RRF)")
    print("=" * 70)
    search_index = build_hybrid_search_index(datasets.catalog_df)
    print(hybrid_search(search_index, "sad romantic songs", top_k=10))

    print("\n" + "=" * 70)
    print("STEP 7: Learning to rank (LambdaMART)")
    print("=" * 70)
    ranking_df = build_ranking_dataframe(datasets.catalog_df, datasets.listening_df)
    all_queries = build_all_training_queries(ranking_df, datasets.merged_df)
    lambdamart = train_lambdamart(search_index, ranking_df, all_queries)
    print(rank_candidates(lambdamart, "sad songs", top_k=10))

    print("\n" + "=" * 70)
    print("STEP 8: Cross-encoder reranking (final search pipeline)")
    print("=" * 70)
    reranker = load_reranker(lambdamart)
    print(search(reranker, "songs like shape of you", top_k=10))


if __name__ == "__main__":
    main()
