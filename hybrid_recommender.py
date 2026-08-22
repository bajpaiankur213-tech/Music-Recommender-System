"""
hybrid_recommender.py
=======================
Step 4: blend collaborative filtering with content-based similarity into a
single score.

The content-based recommender (Step 2) runs on a random 10k-track sample
keyed by ``track_name``. Collaborative filtering (Step 3) runs on a
filtered pool of the 500 most popular tracks keyed by ``track_id``. To
blend them meaningfully, this module rebuilds a *second* content
similarity matrix restricted to that same 500-track CF universe, so both
signals can be combined by ``track_id``.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from collaborative_filtering import CollaborativeFilteringModel, recommend_users


@dataclass
class HybridModel:
    cf_model: CollaborativeFilteringModel
    music_df: pd.DataFrame
    content_similarity_df: pd.DataFrame


def _build_content_similarity_on_cf_universe(
    music_df: pd.DataFrame, cf_track_ids: list
) -> pd.DataFrame:
    """Build a content-based similarity matrix restricted to the same
    track universe used by collaborative filtering, so both signals can
    be joined on ``track_id``."""
    content_df = (
        music_df[music_df["track_id"].isin(cf_track_ids)]
        .drop_duplicates(subset="track_id")
        .set_index("track_id")
        .loc[cf_track_ids]  # preserve the same order as the CF matrix columns
    )

    content_df["features"] = (
        content_df["track_name"] + " " + content_df["artist_name"] + " "
        + content_df["genre"] + " " + content_df["tags"]
    )

    vectorizer = TfidfVectorizer(
        stop_words=config.TFIDF_STOP_WORDS,
        max_features=config.HYBRID_CONTENT_TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
    )
    tfidf_matrix = vectorizer.fit_transform(content_df["features"])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return pd.DataFrame(similarity_matrix, index=content_df.index, columns=content_df.index)


def fit_hybrid_model(
    cf_model: CollaborativeFilteringModel, music_df: pd.DataFrame
) -> HybridModel:
    cf_track_ids = cf_model.user_item_matrix.columns.tolist()
    content_similarity_df = _build_content_similarity_on_cf_universe(music_df, cf_track_ids)
    return HybridModel(cf_model=cf_model, music_df=music_df, content_similarity_df=content_similarity_df)


def _normalize(s: pd.Series) -> pd.Series:
    """Min-max normalize a pandas Series to [0, 1]. Returns all-zero if there's no variance."""
    if s.max() == s.min():
        return s * 0
    return (s - s.min()) / (s.max() - s.min())


def hybrid_recommend(
    model: HybridModel, user_id, alpha: float = config.DEFAULT_HYBRID_ALPHA, n: int = 10
):
    """Blend collaborative filtering with content-based similarity.

    alpha : weight on the collaborative filtering signal (0-1).
            (1 - alpha) is the weight on the content-based signal.
            alpha=1   -> pure collaborative filtering
            alpha=0   -> pure content-based
            alpha=0.5 -> equal blend (default)
    n     : number of recommendations to return
    """
    cf_model = model.cf_model
    user_item_matrix = cf_model.user_item_matrix

    if user_id not in user_item_matrix.index:
        return f"User '{user_id}' not found in the filtered user-item matrix (top {config.TOP_N_USERS} users only)."

    # ---- Collaborative filtering signal: playcount mass from similar users ----
    similar_users = recommend_users(cf_model, user_id, n=10).index
    cf_candidates = cf_model.merged_df[cf_model.merged_df["user_id"].isin(similar_users)]
    cf_scores = cf_candidates.groupby("track_id")["playcount"].sum()
    cf_scores = cf_scores.reindex(user_item_matrix.columns, fill_value=0)

    # ---- Content-based signal: avg similarity to tracks this user already listened to ----
    user_history = user_item_matrix.loc[user_id]
    listened_tracks = user_history[user_history > 0].index.tolist()

    if listened_tracks:
        content_scores = model.content_similarity_df[listened_tracks].mean(axis=1)
    else:
        content_scores = pd.Series(0.0, index=model.content_similarity_df.index)

    # ---- Normalize both signals to [0, 1] and blend ----
    cf_norm = _normalize(cf_scores)
    content_norm = _normalize(content_scores)

    hybrid_score = alpha * cf_norm + (1 - alpha) * content_norm
    hybrid_score = hybrid_score.drop(index=listened_tracks, errors="ignore")  # don't re-recommend history

    top = hybrid_score.sort_values(ascending=False).head(n).reset_index()
    top.columns = ["track_id", "hybrid_score"]
    top = top.merge(model.music_df[["track_id", "track_name", "artist_name"]], on="track_id", how="left")

    return top[["track_id", "track_name", "artist_name", "hybrid_score"]].round({"hybrid_score": 4})


if __name__ == "__main__":
    from data_preprocessing import load_datasets
    from collaborative_filtering import fit_collaborative_filtering_model

    datasets = load_datasets()
    cf_model = fit_collaborative_filtering_model(datasets.merged_df, datasets.user_item_matrix)
    model = fit_hybrid_model(cf_model, datasets.music_df)

    sample_user = datasets.user_item_matrix.index[0]

    print("Pure collaborative filtering (alpha=1):")
    print(hybrid_recommend(model, sample_user, alpha=1, n=5))

    print("\nPure content-based (alpha=0):")
    print(hybrid_recommend(model, sample_user, alpha=0, n=5))

    print("\nHybrid blend (alpha=0.5):")
    print(hybrid_recommend(model, sample_user, alpha=0.5, n=5))
