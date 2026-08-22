"""
config.py
=========
Central place for file paths and tunable constants used across the project.
Edit these values instead of hunting through every module.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.environ.get("MUSIC_RECOMMENDER_DATA_DIR", "./data")

MUSIC_INFO_CSV = os.path.join(DATA_DIR, "Music Info.csv")
LISTENING_HISTORY_CSV = os.path.join(DATA_DIR, "User Listening History.csv")

CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "music"

# ---------------------------------------------------------------------------
# Sampling / dataset size
# ---------------------------------------------------------------------------
CATALOG_SAMPLE_SIZE = 10000     # number of tracks used for content-based + search
RANDOM_STATE = 42

TOP_N_USERS = 500               # collaborative filtering: most active users
TOP_N_TRACKS = 500              # collaborative filtering: most played tracks

# ---------------------------------------------------------------------------
# Content-based (TF-IDF) recommender
# ---------------------------------------------------------------------------
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_STOP_WORDS = "english"

HYBRID_CONTENT_TFIDF_MAX_FEATURES = 3000

# ---------------------------------------------------------------------------
# Hybrid CF + content recommender
# ---------------------------------------------------------------------------
DEFAULT_HYBRID_ALPHA = 0.5      # weight on collaborative filtering signal

# ---------------------------------------------------------------------------
# Query understanding (NLU)
# ---------------------------------------------------------------------------
SPACY_MODEL = "en_core_web_sm"
FUZZY_MATCH_THRESHOLD = 80

MOOD_GENRE_KEYWORDS = {
    "sad", "happy", "romantic", "party", "chill", "workout", "love",
    "breakup", "melancholy", "energetic", "relaxing", "dance", "acoustic",
    "upbeat", "calm", "motivational",
}

SYNONYM_EXPANSION = {
    "sad": ["sad", "melancholy", "heartbreak", "emotional", "blue"],
    "happy": ["happy", "upbeat", "cheerful", "joyful"],
    "romantic": ["romantic", "love", "affection"],
    "love": ["love", "romantic", "affection"],
    "party": ["party", "dance", "club", "upbeat"],
    "chill": ["chill", "relax", "relaxing", "calm", "lofi"],
    "workout": ["workout", "gym", "energetic", "motivational"],
}

# Generic filler words that add noise to mood/artist/track matching but
# carry no meaning. Negation words ("not"/"no") are intentionally excluded
# so negation detection still works downstream.
STOPWORDS_PARTIAL = {
    "a", "an", "the", "of", "for", "please", "some", "me", "give", "play",
    "show", "find", "any", "want", "need",
}

# ---------------------------------------------------------------------------
# Hybrid search (BM25 + dense embeddings + Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------
DENSE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RRF_K = 60
HYBRID_SEARCH_TOP_K = 20
EMBEDDING_BATCH_SIZE = 5000

# ---------------------------------------------------------------------------
# Learning to rank (LambdaMART)
# ---------------------------------------------------------------------------
LTR_CANDIDATE_TOP_K = 30
LTR_N_SYNTHETIC_GENRE_QUERIES = 150
LTR_N_SYNTHETIC_ARTIST_QUERIES = 150
LTR_N_USER_SESSION_QUERIES = 200
LTR_VAL_SPLIT_SIZE = 0.2

LAMBDAMART_PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    boosting_type="gbdt",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=5,
    label_gain=[0, 1, 3, 7],
    random_state=RANDOM_STATE,
)

# ---------------------------------------------------------------------------
# Reranking (cross-encoder)
# ---------------------------------------------------------------------------
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_POOL_SIZE = 20
FINAL_TOP_K = 10
