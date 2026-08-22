# Music Recommender System

A multi-stage music recommendation and search pipeline, built up from
simple content-based filtering to a full retrieval + learning-to-rank +
reranking search stack.

The project is organized as one Python module per pipeline stage, so each
technique can be read, run, and tested independently, or chained together
end-to-end via `main.py`.

## Pipeline Overview

| Step | File | What it does |
|------|------|---------------|
| 1 | `config.py` | Central paths and tunable constants |
| 1 | `data_preprocessing.py` | Loads and cleans the raw CSVs into reusable dataframes |
| 2 | `content_based_recommender.py` | TF-IDF over track/artist/genre/tags + cosine similarity |
| 3 | `collaborative_filtering.py` | User-user similarity over a playcount matrix |
| 4 | `hybrid_recommender.py` | Blends collaborative filtering and content-based signals |
| 5 | `query_understanding.py` | NLU layer: tokenization, entity extraction, intent detection, mood-synonym expansion |
| 6 | `hybrid_search.py` | BM25 + dense embedding retrieval fused with Reciprocal Rank Fusion (RRF) |
| 7 | `learning_to_rank.py` | LambdaMART (LightGBM `LGBMRanker`) trained on synthetic + behavioral queries |
| 8 | `reranker.py` | Cross-encoder reranking of the final candidate pool; exposes the end-to-end `search()` function |
| - | `main.py` | Runs every stage above in order, as a working demo |

### How the stages build on each other

```
data_preprocessing
        │
        ├──> content_based_recommender ─┐
        │                               ├──> hybrid_recommender
        └──> collaborative_filtering  ──┘

data_preprocessing ──> query_understanding ──> hybrid_search
                                                     │
                                                     ▼
                                          learning_to_rank (LambdaMART)
                                                     │
                                                     ▼
                                          reranker (cross-encoder) ──> search()
```

The first three stages (content-based, collaborative filtering, hybrid
recommender) form one self-contained "recommend for a known user" flow.
The last four stages (query understanding, hybrid search,
learning-to-rank, reranking) form a separate "free-text search" flow that
answers queries like `"sad romantic songs"` or `"songs like shape of
you"`. `main.py` demonstrates both.

## Project Structure

```
music_recommender_system/
├── config.py
├── data_preprocessing.py
├── content_based_recommender.py
├── collaborative_filtering.py
├── hybrid_recommender.py
├── query_understanding.py
├── hybrid_search.py
├── learning_to_rank.py
├── reranker.py
├── main.py
├── requirements.txt
└── README.md
```

## Dataset

This project expects two CSV files, placed in a `data/` directory (or any
directory set via the `MUSIC_RECOMMENDER_DATA_DIR` environment variable —
see `config.py`):

- **`Music Info.csv`** — catalog metadata with columns including
  `track_id`, `name`, `artist`, `genre`, `tags`.
- **`User Listening History.csv`** — user playcounts with columns
  `track_id`, `user_id`, `playcount` (the file ships with a header row
  that isn't parsed automatically, which `data_preprocessing.py` handles).

Both files come from the "Music Recommender System" dataset available on
Kaggle. Point `config.DATA_DIR` at wherever you've downloaded them, e.g.:

```bash
export MUSIC_RECOMMENDER_DATA_DIR=/path/to/data
```

## Installation

Requires **Python 3.10+** (the codebase uses `dataclasses` and PEP 604
type hints like `tuple[int, int]`).

```bash
git clone <your-repo-url>
cd music_recommender_system
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Dependencies

| Package | Used for |
|---|---|
| `numpy`, `pandas` | Data manipulation |
| `scikit-learn` | TF-IDF vectorization, cosine similarity, train/val group splitting |
| `spacy` (+ `en_core_web_sm` model) | Tokenization, lemmatization, NER for the query-understanding layer |
| `rapidfuzz` | Fuzzy string matching for typo-tolerant artist/track lookup |
| `rank_bm25` | BM25 lexical retrieval |
| `sentence-transformers` | Dense embeddings (`all-MiniLM-L6-v2`) and cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) |
| `chromadb` | Vector storage/search for dense retrieval |
| `lightgbm` | `LGBMRanker` (LambdaMART) for learning-to-rank |

> **Note:** `sentence-transformers`, `chromadb`, and `lightgbm` pull in
> PyTorch and can take a while to install and to run on first use
> (model weights are downloaded on first call). A GPU isn't required, but
> speeds up embedding and reranking noticeably.

## Usage

### Run the full pipeline

```bash
python main.py
```

This loads the data, fits every model in order, and prints example output
from each stage.

### Use a single stage

Every module is independently importable, e.g.:

```python
from data_preprocessing import load_datasets
from content_based_recommender import fit_content_based_model, recommend

datasets = load_datasets()
model = fit_content_based_model(datasets.catalog_df)
print(recommend(model, "shape of you", top_n=5))
```

```python
from data_preprocessing import load_datasets
from hybrid_search import build_hybrid_search_index, hybrid_search

datasets = load_datasets()
index = build_hybrid_search_index(datasets.catalog_df)
print(hybrid_search(index, "sad romantic songs"))
```

Each module also has a `if __name__ == "__main__":` block, so you can run
it directly to see a quick demo of just that stage:

```bash
python content_based_recommender.py
python collaborative_filtering.py
python hybrid_recommender.py
python query_understanding.py
python hybrid_search.py
python learning_to_rank.py   # trains LambdaMART — slower, runs hybrid_search per training query
python reranker.py           # runs the full 8-stage pipeline
```

## Notes on Design Decisions

- **`track_id` is kept from the start.** The content-based sample
  (`catalog_df`) retains `track_id` throughout, so the ranking stage can
  join playcount labels back onto it directly, instead of re-loading and
  re-sampling the raw CSV a second time to recover a dropped column.
- **Word-level tokenization, not subword.** The query-understanding layer
  intentionally uses spaCy's word-level tokenizer rather than
  BERT/WordPiece subword tokenization, because the mood/genre keyword
  dictionaries it relies on (`MOOD_GENRE_KEYWORDS`, `SYNONYM_EXPANSION`)
  are whole-word lookups that WordPiece would silently break.
- **RRF over raw score blending.** BM25 scores and cosine distances live
  on incompatible scales, so hybrid retrieval fuses them by rank position
  (Reciprocal Rank Fusion) rather than trying to normalize and average the
  raw scores directly.
- **Why LambdaMART sits between retrieval and reranking.** Hybrid search
  is fast but coarse; a cross-encoder is accurate but too slow to run over
  the full catalog. LambdaMART narrows thousands of candidates down to a
  small top-K pool using cheap features, and the cross-encoder does the
  expensive, precise pass only over that pool.

## Limitations

- Collaborative filtering and the hybrid recommender are restricted to
  the top 500 most active users and top 500 most-played tracks
  (`config.TOP_N_USERS` / `config.TOP_N_TRACKS`) for tractability on a
  single machine; this is a demonstration-scale project, not a
  production-scale one.
- The content-based sample is capped at 10,000 tracks
  (`config.CATALOG_SAMPLE_SIZE`) for the same reason.
- Chroma's collection is created fresh (or reused) each run via
  `get_or_create_collection`; re-running `hybrid_search.py` repeatedly
  will re-add the same documents unless you clear `./chroma_db` first.
