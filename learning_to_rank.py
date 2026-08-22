"""
learning_to_rank.py
=====================
Step 7: train a LambdaMART model (LightGBM's ``LGBMRanker``) to re-rank the
candidates that ``hybrid_search()`` (Step 6) returns.

Pipeline:
  1. Turn playcount into a graded relevance label (0-3) per track.
  2. Generate synthetic training queries (by genre, mood, artist) plus
     real user-session queries (each top user's most-played genre).
  3. For every training query, run hybrid_search() to get candidates,
     build numeric LTR features per (query, candidate) pair, and label
     them.
  4. Train ``LGBMRanker`` with ``objective="lambdarank"``.
  5. ``rank_candidates()`` re-scores and re-sorts hybrid_search() output
     for a new query at inference time.
"""

import random
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import config
from hybrid_search import HybridSearchIndex, hybrid_search

QUERY_TEMPLATES_GENRE = ["{genre} songs", "{genre} music", "some {genre}"]
QUERY_TEMPLATES_MOOD = ["{mood} music", "{mood} songs", "something {mood}"]
QUERY_TEMPLATES_ARTIST = ["songs like {artist}", "{artist} {genre}", "music by {artist}"]


# ---------------------------------------------------------------------------
# Step 7.1: relevance labels
# ---------------------------------------------------------------------------

def _bucket_relevance(playcount: float, q33: float, q66: float) -> int:
    """0 = never played. 1/2/3 = tercile of playcount among played tracks."""
    if playcount <= 0:
        return 0
    if playcount <= q33:
        return 1
    if playcount <= q66:
        return 2
    return 3


def build_ranking_dataframe(catalog_df: pd.DataFrame, listening_df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``total_playcount`` and a graded ``relevance_grade`` (0-3) to
    the catalog, based on aggregate playcount across all listening history."""
    ranking_df = catalog_df.copy()

    track_playcount = listening_df.groupby("track_id")["playcount"].sum()
    ranking_df["total_playcount"] = ranking_df["track_id"].map(track_playcount).fillna(0)

    nonzero = ranking_df.loc[ranking_df["total_playcount"] > 0, "total_playcount"]
    if len(nonzero) > 0:
        q33, q66 = nonzero.quantile([0.33, 0.66])
    else:
        q33, q66 = 0, 0

    ranking_df["relevance_grade"] = ranking_df["total_playcount"].apply(
        lambda pc: _bucket_relevance(pc, q33, q66)
    )
    return ranking_df


# ---------------------------------------------------------------------------
# Step 7.2: synthetic + user-session training queries
# ---------------------------------------------------------------------------

def generate_synthetic_queries(
    ranking_df: pd.DataFrame,
    n_genre: int = config.LTR_N_SYNTHETIC_GENRE_QUERIES,
    n_artist: int = config.LTR_N_SYNTHETIC_ARTIST_QUERIES,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Builds (query_id, query_text, type, anchor) rows. ``anchor`` is the
    ground-truth genre/mood/artist string the query was generated from --
    used later to decide which candidates are actually relevant."""
    rng = random.Random(random_state)
    rows = []

    genres = sorted(g for g in ranking_df["genre"].unique() if g)
    for i, genre in enumerate(rng.sample(genres, min(n_genre, len(genres)))):
        rows.append({
            "query_id": f"genre_{i}",
            "query_text": rng.choice(QUERY_TEMPLATES_GENRE).format(genre=genre),
            "type": "genre",
            "anchor": genre,
        })

    for i, mood in enumerate(sorted(config.MOOD_GENRE_KEYWORDS)):
        rows.append({
            "query_id": f"mood_{i}",
            "query_text": rng.choice(QUERY_TEMPLATES_MOOD).format(mood=mood),
            "type": "mood",
            "anchor": mood,
        })

    artists = sorted(a for a in ranking_df["artist_name"].unique() if a)
    for i, artist in enumerate(rng.sample(artists, min(n_artist, len(artists)))):
        artist_genre = ranking_df.loc[ranking_df["artist_name"] == artist, "genre"].iloc[0]
        rows.append({
            "query_id": f"artist_{i}",
            "query_text": rng.choice(QUERY_TEMPLATES_ARTIST).format(artist=artist, genre=artist_genre),
            "type": "artist",
            "anchor": artist,
        })

    return pd.DataFrame(rows)


def generate_user_session_queries(
    merged_df: pd.DataFrame, n_users: int = config.LTR_N_USER_SESSION_QUERIES
) -> pd.DataFrame:
    """For each top-listening user, their most-played genre becomes an
    implied "intent" query -- real behavioral signal, not templated."""
    rows = []
    top_users = merged_df["user_id"].value_counts().head(n_users).index

    for i, user_id in enumerate(top_users):
        user_plays = merged_df[merged_df["user_id"] == user_id]
        if user_plays.empty:
            continue

        top_genre = user_plays["genre"].mode()
        if top_genre.empty or not top_genre.iloc[0]:
            continue

        genre = top_genre.iloc[0]
        rows.append({
            "query_id": f"user_{i}",
            "query_text": f"{genre} songs",
            "type": "user_session",
            "anchor": genre,
            "anchor_user_id": user_id,
        })

    return pd.DataFrame(rows)


def build_all_training_queries(ranking_df: pd.DataFrame, merged_df: pd.DataFrame) -> pd.DataFrame:
    synthetic_queries = generate_synthetic_queries(ranking_df)
    user_queries = generate_user_session_queries(merged_df)
    return pd.concat([synthetic_queries, user_queries], ignore_index=True)


# ---------------------------------------------------------------------------
# Step 7.3: LTR feature engineering + labeling
# ---------------------------------------------------------------------------

def _relevance_label(row: pd.Series, anchor: str, qtype: str) -> int:
    """Ground-truth label used to train LambdaMART, 0-3.
    Base label = playcount-derived relevance_grade. Bonus: if the
    candidate's own genre/artist matches the query's anchor, nudge the
    label up by 1 (capped at 3) -- teaches the model that e.g. "genre
    songs" queries should prefer genre-matching tracks even when playcount
    alone doesn't distinguish them."""
    base = int(row["relevance_grade"])

    matched = False
    if qtype == "genre" and row["genre"] == anchor:
        matched = True
    elif qtype == "mood" and anchor in row["tags"]:
        matched = True
    elif qtype in ("artist", "user_session") and row["artist_name"] == anchor:
        matched = True

    return min(base + 1, 3) if matched else base


def build_ltr_features(
    query_text: str, candidates_df: pd.DataFrame, ranking_df: pd.DataFrame
) -> pd.DataFrame:
    """``candidates_df``: output of hybrid_search(), columns
    ``track, artist, score`` (rank == row position). Returns a dataframe
    of numeric features, one row per candidate, aligned 1:1 with
    ``candidates_df``."""
    feats = []
    q_lower = query_text.lower()
    q_tokens = set(q_lower.split())

    for rank, row in enumerate(candidates_df.itertuples(index=False), start=1):
        track_lower = str(row.track).lower()
        artist_lower = str(row.artist).lower()

        cat_matches = ranking_df[
            (ranking_df["track_name"] == track_lower) & (ranking_df["artist_name"] == artist_lower)
        ]
        if cat_matches.empty:
            genre, total_playcount, rel_grade = "", 0.0, 0
        else:
            cat_row = cat_matches.iloc[0]
            genre = cat_row["genre"]
            total_playcount = float(cat_row["total_playcount"])
            rel_grade = int(cat_row["relevance_grade"])

        track_tokens = set(track_lower.split())
        artist_tokens = set(artist_lower.split())

        feats.append({
            "hybrid_rrf_score": float(row.score),
            "hybrid_rank": rank,
            "hybrid_rank_recip": 1.0 / rank,
            "track_title_overlap": len(q_tokens & track_tokens),
            "artist_overlap": len(q_tokens & artist_tokens),
            "genre_in_query": 1.0 if genre and genre in q_lower else 0.0,
            "artist_in_query": 1.0 if artist_lower and artist_lower in q_lower else 0.0,
            "log_playcount": np.log1p(total_playcount),
            "relevance_grade_prior": rel_grade,  # weak prior signal, not the label itself
            "query_len": len(q_tokens),
        })

    return pd.DataFrame(feats)


def _lookup_relevance_row(ranking_df: pd.DataFrame, track: str, artist: str) -> pd.Series:
    matches = ranking_df[
        (ranking_df["track_name"] == str(track).lower()) & (ranking_df["artist_name"] == str(artist).lower())
    ]
    if matches.empty:
        return pd.Series({"relevance_grade": 0, "genre": "", "tags": "", "artist_name": ""})
    return matches.iloc[0]


def build_training_set(
    search_index: HybridSearchIndex,
    ranking_df: pd.DataFrame,
    all_queries: pd.DataFrame,
    top_k: int = config.LTR_CANDIDATE_TOP_K,
):
    """Runs hybrid_search() for every training query, builds features and
    labels, and returns ``(X, y, group_sizes, meta)`` ready for
    ``LGBMRanker.fit()``. ``group_sizes`` gives the number of candidate
    rows per query, in order -- LightGBM's ranking objective needs this to
    know query boundaries."""
    X_rows, y_rows, groups, meta_rows = [], [], [], []

    for q in all_queries.itertuples(index=False):
        cands = hybrid_search(search_index, q.query_text, top_k=top_k)
        if cands.empty:
            continue

        feats = build_ltr_features(q.query_text, cands, ranking_df)

        labels = [
            _relevance_label(_lookup_relevance_row(ranking_df, t, a), q.anchor, q.type)
            for t, a in zip(cands["track"], cands["artist"])
        ]

        X_rows.append(feats)
        y_rows.extend(labels)
        groups.append(len(feats))
        meta_rows.append(pd.DataFrame({
            "query_id": q.query_id,
            "query_text": q.query_text,
            "track": cands["track"],
            "artist": cands["artist"],
        }))

    X = pd.concat(X_rows, ignore_index=True)
    y = np.array(y_rows)
    meta = pd.concat(meta_rows, ignore_index=True)
    return X, y, groups, meta


# ---------------------------------------------------------------------------
# Step 7.4: train/val split + LambdaMART training
# ---------------------------------------------------------------------------

@dataclass
class LambdaMARTModel:
    model: lgb.LGBMRanker
    ranking_df: pd.DataFrame
    search_index: HybridSearchIndex


def split_train_val(X: pd.DataFrame, y: np.ndarray, meta: pd.DataFrame, group_sizes: list):
    """Query-level train/val split -- never splits within a query group."""
    meta = meta.copy()
    meta["group_idx"] = np.repeat(np.arange(len(group_sizes)), group_sizes)

    splitter = GroupShuffleSplit(n_splits=1, test_size=config.LTR_VAL_SPLIT_SIZE, random_state=config.RANDOM_STATE)
    train_grp_idx, _ = next(splitter.split(X, y, groups=meta["group_idx"]))

    train_mask = meta["group_idx"].isin(meta["group_idx"].iloc[train_grp_idx].unique())
    val_mask = ~train_mask

    def sizes_for(mask):
        return meta[mask].groupby("group_idx", sort=False).size().tolist()

    return (
        X[train_mask], y[train_mask], sizes_for(train_mask),
        X[val_mask], y[val_mask], sizes_for(val_mask),
    )


def train_lambdamart(
    search_index: HybridSearchIndex, ranking_df: pd.DataFrame, all_queries: pd.DataFrame
) -> LambdaMARTModel:
    print("Building LambdaMART training set (this runs hybrid_search per query)...")
    X, y, group_sizes, meta = build_training_set(search_index, ranking_df, all_queries)
    print(f"Training rows: {X.shape}, groups: {len(group_sizes)}, label dist: {np.bincount(y)}")

    X_train, y_train, group_train, X_val, y_val, group_val = split_train_val(X, y, meta, group_sizes)
    print(f"Train: {X_train.shape[0]} rows / {len(group_train)} queries | "
          f"Val: {X_val.shape[0]} rows / {len(group_val)} queries")

    model = lgb.LGBMRanker(**config.LAMBDAMART_PARAMS)
    model.fit(
        X_train, y_train,
        group=group_train,
        eval_set=[(X_val, y_val)],
        eval_group=[group_val],
        eval_at=[5, 10],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(20)],
    )

    print("\nFeature importance (gain):")
    print(pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False))

    return LambdaMARTModel(model=model, ranking_df=ranking_df, search_index=search_index)


def rank_candidates(
    lambdamart: LambdaMARTModel, query_text: str, top_k: int = config.LTR_CANDIDATE_TOP_K
) -> pd.DataFrame:
    """Full ranking stage: hybrid_search() generates candidates, then
    LambdaMART re-scores and re-sorts them."""
    cands = hybrid_search(lambdamart.search_index, query_text, top_k=top_k)
    if cands.empty:
        return cands

    feats = build_ltr_features(query_text, cands, lambdamart.ranking_df)
    cands = cands.copy()
    cands["lambdamart_score"] = lambdamart.model.predict(feats)
    return cands.sort_values("lambdamart_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from data_preprocessing import load_datasets
    from hybrid_search import build_hybrid_search_index

    datasets = load_datasets()
    search_index = build_hybrid_search_index(datasets.catalog_df)
    ranking_df = build_ranking_dataframe(datasets.catalog_df, datasets.listening_df)
    all_queries = build_all_training_queries(ranking_df, datasets.merged_df)

    lambdamart = train_lambdamart(search_index, ranking_df, all_queries)

    print("\n--- LambdaMART ranked example ---")
    print(rank_candidates(lambdamart, "sad songs", top_k=20))
