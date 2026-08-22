"""
data_preprocessing.py
======================
Step 1 of the pipeline: load the raw CSVs and produce the cleaned
dataframes every other module builds on.

Two dataframes are produced here and reused everywhere downstream:

- ``catalog_df``   : a deduplicated, lowercase-normalized sample of the
                     song catalog (with ``track_id`` kept intact), used by
                     the content-based recommender, the NLU layer, hybrid
                     search, and the learning-to-rank labels.
- ``user_item_df``/``merged_df`` : the filtered listening-history data used
                     by collaborative filtering.

Keeping ``track_id`` on ``catalog_df`` from the start (instead of dropping
it for content-based work and re-deriving it later for ranking, as the
original notebook did) means every module can join back to listening
history without re-sampling or re-asserting row alignment.
"""

from dataclasses import dataclass

import pandas as pd

import config


@dataclass
class Datasets:
    """Container bundling every dataframe the rest of the pipeline needs."""

    music_df: pd.DataFrame        # full raw catalog (renamed columns, not sampled)
    listening_df: pd.DataFrame    # raw user listening history
    catalog_df: pd.DataFrame      # sampled catalog used for content-based + search
    merged_df: pd.DataFrame       # filtered listening history joined to music_df
    user_item_matrix: pd.DataFrame


def load_raw_data(
    music_info_csv: str = config.MUSIC_INFO_CSV,
    listening_history_csv: str = config.LISTENING_HISTORY_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the two source CSVs.

    ``User Listening History.csv`` ships with a header row that pandas
    can't infer cleanly, so the columns are named explicitly.
    """
    music_df = pd.read_csv(music_info_csv)
    listening_df = pd.read_csv(
        listening_history_csv,
        skiprows=1,
        header=None,
        names=["track_id", "user_id", "playcount"],
    )
    return music_df, listening_df


def _normalize_music_df(music_df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns and lowercase text fields, without sampling or
    dropping ``track_id`` (that keeps this reusable for both the
    content-based sample and the full catalog used for CF joins)."""
    df = music_df.rename(columns={"name": "track_name", "artist": "artist_name"})
    df["tags"] = df["tags"].fillna("")
    df["genre"] = df["genre"].fillna("")

    for col in ("track_name", "artist_name", "tags", "genre"):
        df[col] = df[col].str.lower()

    return df


def build_catalog(
    music_df: pd.DataFrame,
    sample_size: int = config.CATALOG_SAMPLE_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Build the sampled, feature-engineered catalog used for content-based
    recommendations, the NLU layer, and hybrid search.

    Returns a dataframe with columns:
    ``track_id, track_name, artist_name, tags, genre, features``.
    """
    df = _normalize_music_df(music_df)
    df = df[["track_id", "track_name", "artist_name", "tags", "genre"]]

    df["features"] = (
        df["track_name"] + " " + df["artist_name"] + " " + df["genre"] + " " + df["tags"]
    )

    df = df.sample(sample_size, random_state=random_state).reset_index(drop=True)
    return df


def build_collaborative_filtering_data(
    listening_df: pd.DataFrame,
    music_df: pd.DataFrame,
    top_n_users: int = config.TOP_N_USERS,
    top_n_tracks: int = config.TOP_N_TRACKS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter listening history down to the most active users and most
    popular tracks, then build a user-item playcount matrix.

    Returns ``(merged_df, user_item_matrix)``.
    """
    normalized_music_df = _normalize_music_df(music_df)

    top_users = listening_df["user_id"].value_counts().head(top_n_users).index
    top_tracks = listening_df["track_id"].value_counts().head(top_n_tracks).index

    filtered_df = listening_df[
        listening_df["user_id"].isin(top_users) & listening_df["track_id"].isin(top_tracks)
    ]

    merged_df = pd.merge(filtered_df, normalized_music_df, on="track_id", how="inner")

    user_item_matrix = merged_df.pivot_table(
        index="user_id", columns="track_id", values="playcount", fill_value=0
    )

    return merged_df, user_item_matrix


def load_datasets() -> Datasets:
    """Convenience entry point: load everything and return a ``Datasets``
    bundle ready to hand to every other module."""
    music_df_raw, listening_df = load_raw_data()

    catalog_df = build_catalog(music_df_raw)
    merged_df, user_item_matrix = build_collaborative_filtering_data(listening_df, music_df_raw)

    music_df = _normalize_music_df(music_df_raw)

    return Datasets(
        music_df=music_df,
        listening_df=listening_df,
        catalog_df=catalog_df,
        merged_df=merged_df,
        user_item_matrix=user_item_matrix,
    )


if __name__ == "__main__":
    datasets = load_datasets()
    print(f"catalog_df:        {datasets.catalog_df.shape}")
    print(f"merged_df:         {datasets.merged_df.shape}")
    print(f"user_item_matrix:  {datasets.user_item_matrix.shape}")
    print(datasets.catalog_df.head())
