"""
content_based_recommender.py
=============================
Step 2: content-based recommendations using TF-IDF over
``track_name + artist_name + genre + tags``, ranked by cosine similarity.

``recommend()`` accepts either:
- a known track name, e.g. "shape of you"  -> uses the precomputed
  similarity row for that track (fast, O(1) lookup), or
- a free-form feature description, e.g. "romantic sad piano indie" -> not
  found in the catalog, so it's TF-IDF-transformed live and compared
  against every track.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config


@dataclass
class ContentBasedModel:
    catalog_df: pd.DataFrame
    vectorizer: TfidfVectorizer
    tfidf_matrix: "scipy.sparse.spmatrix"          # noqa: F821 (type hint only)
    similarity_matrix: "numpy.ndarray"             # noqa: F821


def fit_content_based_model(catalog_df: pd.DataFrame) -> ContentBasedModel:
    """Fit a TF-IDF vectorizer over the catalog's ``features`` column and
    precompute the full pairwise cosine similarity matrix."""
    vectorizer = TfidfVectorizer(
        stop_words=config.TFIDF_STOP_WORDS,
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
    )
    tfidf_matrix = vectorizer.fit_transform(catalog_df["features"])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return ContentBasedModel(
        catalog_df=catalog_df,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        similarity_matrix=similarity_matrix,
    )


def recommend(model: ContentBasedModel, song: str, top_n: int = 10):
    """Recommend ``top_n`` tracks similar to ``song``.

    ``song`` may be a track name already in the catalog, or a free-form
    feature description (genre/mood/artist words) if it isn't.
    """
    df = model.catalog_df
    song = song.lower().strip()
    input_artist = None

    matches = df[df["track_name"].str.contains(song, na=False)]

    if not matches.empty:
        idx = matches.index[0]
        input_artist = df.iloc[idx].artist_name
        sim_scores = model.similarity_matrix[idx]  # precomputed row, O(1)
        print(f"Found: '{df.iloc[idx].track_name}' by {input_artist}")
    else:
        print(f"'{song}' not in catalog -- treating as a feature string, computing live similarity.")
        new_vec = model.vectorizer.transform([song])
        sim_scores = cosine_similarity(new_vec, model.tfidf_matrix)[0]

    scores = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)

    result = []
    artist_count = {}

    for i, score in scores:
        track = df.iloc[i].track_name
        artist = df.iloc[i].artist_name

        if song in track:
            continue  # skip the query song itself

        if input_artist and artist == input_artist:
            artist_count[artist] = artist_count.get(artist, 0) + 1
            if artist_count[artist] > 2:
                continue  # cap how many tracks from the same artist can appear

        result.append({"track_name": track, "artist": artist, "similarity": round(score, 4)})

        if len(result) == top_n:
            break

    if not result:
        return "No recommendations found."

    return pd.DataFrame(result)


if __name__ == "__main__":
    from data_preprocessing import load_datasets

    datasets = load_datasets()
    model = fit_content_based_model(datasets.catalog_df)

    print(recommend(model, "shape of you", top_n=5))
    print(recommend(model, "romantic sad piano indie", top_n=5))
