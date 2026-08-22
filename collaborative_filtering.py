"""
collaborative_filtering.py
============================
Step 3: user-based collaborative filtering.

Builds user-user cosine similarity from the playcount matrix, then
recommends tracks by aggregating the playcounts of a target user's most
similar users.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class CollaborativeFilteringModel:
    merged_df: pd.DataFrame
    user_item_matrix: pd.DataFrame
    user_similarity_df: pd.DataFrame


def fit_collaborative_filtering_model(
    merged_df: pd.DataFrame, user_item_matrix: pd.DataFrame
) -> CollaborativeFilteringModel:
    """Compute user-user cosine similarity over the user-item playcount matrix."""
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index
    )

    return CollaborativeFilteringModel(
        merged_df=merged_df,
        user_item_matrix=user_item_matrix,
        user_similarity_df=user_similarity_df,
    )


def recommend_users(model: CollaborativeFilteringModel, user_id, n: int = 10) -> pd.Series:
    """Return the ``n`` users most similar to ``user_id`` (excluding themselves)."""
    similar_users = model.user_similarity_df[user_id].sort_values(ascending=False)
    similar_users = similar_users.drop(user_id)
    return similar_users.head(n)


def recommend_songs(model: CollaborativeFilteringModel, user_id, n: int = 200) -> pd.Series:
    """Aggregate playcounts from a user's most similar users to surface
    candidate tracks, ranked by total playcount among that neighborhood."""
    similar_users = recommend_users(model, user_id, n=10).index
    plays_from_similar_users = model.merged_df[model.merged_df["user_id"].isin(similar_users)]
    recommended = (
        plays_from_similar_users.groupby("track_id")["playcount"].sum().sort_values(ascending=False)
    )
    return recommended.head(n)


def recommend_songs_with_metadata(
    model: CollaborativeFilteringModel, music_df: pd.DataFrame, user_id, n: int = 200
) -> pd.DataFrame:
    """Same as ``recommend_songs`` but joined back to track name/artist for display."""
    recommended_tracks = recommend_songs(model, user_id, n=n).reset_index()
    final = recommended_tracks.merge(
        music_df[["track_id", "track_name", "artist_name"]], on="track_id", how="left"
    )
    return final[["track_name", "artist_name", "playcount"]]


if __name__ == "__main__":
    from data_preprocessing import load_datasets

    datasets = load_datasets()
    model = fit_collaborative_filtering_model(datasets.merged_df, datasets.user_item_matrix)

    sample_user = datasets.user_item_matrix.index[0]
    print(f"Users similar to {sample_user}:")
    print(recommend_users(model, sample_user))

    print(f"\nSong recommendations for {sample_user}:")
    print(recommend_songs_with_metadata(model, datasets.music_df, sample_user).head(20))
