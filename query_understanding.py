"""
query_understanding.py
========================
Step 5: query rectification pipeline for the hybrid music recommender.

Runs *before* retrieval to turn a raw, possibly messy user query into a
structured, typo-corrected, intent-tagged representation that downstream
retrieval (hybrid_search.py) can consume.

Pipeline stages, in order:
  1. tokenize()         - spaCy word-level tokens, stopword removal, lemmatization
  2. extract_entities()  - hybrid NER (spaCy pretrained NER + rule-based mood/genre
                           keyword lookup + fuzzy matching for typo correction)
  3. detect_intent()     - confidence-weighted intent classification
  4. expand_query()      - mood synonym expansion
  5. embed_query()       - optional semantic fallback (sentence-transformers),
                           used only when the rule-based stages are low-confidence

Design note: word-level (spaCy) tokenization was deliberately kept instead
of BERT/WordPiece subword tokenization. WordPiece splits words into
subwords ("workout" -> "work", "##out"), which silently breaks every
whole-word dictionary lookup below (MOOD_GENRE_KEYWORDS, SYNONYM_EXPANSION)
with no error, just degraded mood/entity detection.
"""

from functools import lru_cache

import spacy
from rapidfuzz import fuzz, process

import config

nlp = spacy.load(config.SPACY_MODEL)

# Auto-build a "value -> canonical key" map from SYNONYM_EXPANSION, so any
# mood word that only appears as a synonym value (e.g. 'love' under
# 'romantic', 'blue' under 'sad') resolves back to its canonical key. A key
# always maps to itself first.
MOOD_CANONICAL = {key: key for key in config.SYNONYM_EXPANSION}
for _key, _synonyms in config.SYNONYM_EXPANSION.items():
    for _syn in _synonyms:
        MOOD_CANONICAL.setdefault(_syn, _key)

_st_model = None  # lazily loaded by embed_query(), only if semantic fallback is used


def build_known_entities(catalog_df) -> tuple[list, list]:
    """Extract the lookup lists used for fuzzy artist/track grounding.
    Not used for retrieval itself -- only for spell correction and NER."""
    known_tracks = catalog_df["track_name"].dropna().unique().tolist()
    known_artists = catalog_df["artist_name"].dropna().unique().tolist()
    return known_tracks, known_artists


def detect_negated_tokens(doc) -> set:
    """Find tokens under negation, e.g. "not love songs" -> {'love', ...}.
    Used to exclude negated words from mood matching."""
    negated = set()
    tokens_list = [tok.text.lower() for tok in doc]

    for tok in doc:
        if tok.dep_ == "neg":
            head = tok.head
            for child in head.children:
                negated.add(child.text.lower())
            negated.add(head.text.lower())

    negation_words = {"not", "no", "without", "except"}
    for i, tok in enumerate(tokens_list):
        if tok in negation_words:
            for j in range(i + 1, min(i + 3, len(tokens_list))):
                negated.add(tokens_list[j])
            negated.add(tok)

    return negated


def tokenize(query: str):
    """Word-level tokenization with stopword removal and lemmatization."""
    doc = nlp(query.lower().strip())

    tokens = []
    for tok in doc:
        if tok.is_punct or tok.is_space:
            continue
        if tok.text in config.STOPWORDS_PARTIAL:
            continue
        tokens.append(tok.lemma_)

    return tokens, doc


def extract_entities(
    query: str,
    doc,
    known_artists: list,
    known_tracks: list,
    fuzzy_threshold: int = config.FUZZY_MATCH_THRESHOLD,
) -> dict:
    """Hybrid NER combining:
      - spaCy's pretrained NER  -> generic PERSON/ORG entities as artist candidates
      - rule-based keyword lookup -> mood/genre (spaCy has no built-in label for this)
      - fuzzy matching -> corrects typos in artist/track mentions

    Returns an entities dict with a per-entity confidence score.
    """
    entities = {"artist": None, "track": None, "mood": [], "confidence": {}}
    q_lower = " ".join(query.lower().strip().split())  # collapse extra whitespace
    tokens = set(q_lower.split())

    negated = detect_negated_tokens(doc)
    tokens_for_mood = tokens - negated

    # --- mood/genre via rule-based keyword dictionary ---
    found_moods = tokens_for_mood & config.MOOD_GENRE_KEYWORDS
    if found_moods:
        entities["mood"] = list(found_moods)
        entities["confidence"]["mood"] = 1.0

    # --- artist via spaCy PERSON/ORG entities, grounded against known_artists ---
    spacy_candidates = [ent.text.lower() for ent in doc.ents if ent.label_ in ("PERSON", "ORG")]
    for cand in spacy_candidates:
        match = process.extractOne(cand, known_artists, scorer=fuzz.WRatio, score_cutoff=fuzzy_threshold)
        if match:
            entities["artist"] = match[0]
            entities["confidence"]["artist"] = round(match[1] / 100, 2)
            break

    if entities["artist"] is None:
        for artist in known_artists:
            if artist and artist.lower() in q_lower and artist.lower() not in config.MOOD_GENRE_KEYWORDS:
                entities["artist"] = artist
                entities["confidence"]["artist"] = 1.0
                break

    # --- track title via fuzzy match (handles typos). Require a higher bar
    # when a mood was already found, since a coincidental fuzzy track match
    # alongside a confident mood signal is more likely to be a false positive. ---
    has_rule_signal = bool(entities["mood"])
    base_threshold = max(fuzzy_threshold, 88)
    effective_threshold = 95 if has_rule_signal else base_threshold
    track_match = process.extractOne(query, known_tracks, scorer=fuzz.WRatio, score_cutoff=effective_threshold)
    if track_match:
        entities["track"] = track_match[0]
        entities["confidence"]["track"] = round(track_match[1] / 100, 2)

    return entities


def detect_intent(entities: dict) -> str:
    """Confidence-weighted intent classifier off already-extracted entities.
    Returns one of: 'artist_lookup', 'track_lookup', 'mood_query', 'unknown'.

    Weighted scoring (rather than a hardcoded track > artist > mood
    priority) so the most trustworthy entity decides intent -- e.g. a
    weak/coincidental fuzzy track match no longer beats a confident mood
    match.
    """
    scores = {
        "track_lookup": entities["confidence"].get("track", 0.0),
        "artist_lookup": entities["confidence"].get("artist", 0.0) * 0.9,
        "mood_query": entities["confidence"].get("mood", 0.0) * 0.85,
    }

    best_intent = max(scores, key=scores.get)

    # A track-only match with no supporting artist/mood signal and
    # confidence under 0.92 is more likely a coincidental fuzzy match than
    # a genuine track lookup -- safer to call it "unknown".
    if (
        best_intent == "track_lookup"
        and scores["track_lookup"] < 0.92
        and not entities.get("artist")
        and not entities.get("mood")
    ):
        return "unknown"

    if scores[best_intent] <= 0:
        return "unknown"

    return best_intent


def expand_query(tokens: list, mood_entities: list) -> list:
    """Expand recognized mood/genre tokens with synonyms before retrieval.
    Only expands recognized mood entities, so artist/track tokens are
    never polluted."""
    expanded = list(tokens)
    seen = set(expanded)

    for mood in mood_entities:
        canonical_mood = MOOD_CANONICAL.get(mood, mood)
        if canonical_mood in config.SYNONYM_EXPANSION:
            for synonym in config.SYNONYM_EXPANSION[canonical_mood]:
                if synonym not in seen:
                    expanded.append(synonym)
                    seen.add(synonym)

    return expanded


def embed_query(normalized_query: str):
    """Optional semantic embedding fallback for queries the rule-based
    stages can't confidently resolve. Lazily loads sentence-transformers
    on first use; returns None if it isn't installed, so callers must
    handle a None embedding (e.g. by skipping the semantic fallback)."""
    global _st_model
    if _st_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            _st_model = SentenceTransformer(config.DENSE_EMBEDDING_MODEL)
        except ImportError:
            return None
    return _st_model.encode(normalized_query, normalize_embeddings=True)


@lru_cache(maxsize=2048)
def rectify_query(
    raw_query: str,
    known_tracks: tuple,
    known_artists: tuple,
    fuzzy_threshold: int = config.FUZZY_MATCH_THRESHOLD,
    use_semantic_fallback: bool = False,
    low_confidence_threshold: float = 0.5,
) -> dict:
    """Run the full query-rectification pipeline.

    Returns a structured dict:
    {original_query, tokens, negated, entities, intent, intent_confidence,
     expansion, embedding (only if use_semantic_fallback=True and
     confidence is low)}

    Notes:
      - Repeated identical calls hit ``@lru_cache`` and skip the entire
        pipeline (especially the relatively expensive spaCy ``nlp()``
        call). Because of this, ``known_tracks``/``known_artists`` must be
        hashable tuples rather than lists.
      - ``intent_confidence`` is exposed so callers (a "did you mean..."
        UI, or a logging/eval pipeline) can distinguish a confident match
        from a borderline one.
      - The semantic fallback triggers whenever confidence is below
        ``low_confidence_threshold``, not only when intent is literally
        "unknown" -- a low-confidence mood_query is barely more useful
        than unknown for retrieval purposes.
    """
    if not raw_query or not raw_query.strip():
        raise ValueError("rectify_query: raw_query must be a non-empty string")

    tokens, doc = tokenize(raw_query)
    negated = detect_negated_tokens(doc)
    entities = extract_entities(raw_query, doc, list(known_artists), list(known_tracks), fuzzy_threshold)
    intent = detect_intent(entities)

    intent_scores = {
        "track_lookup": entities["confidence"].get("track", 0.0),
        "artist_lookup": entities["confidence"].get("artist", 0.0) * 0.9,
        "mood_query": entities["confidence"].get("mood", 0.0) * 0.85,
    }
    intent_confidence = round(intent_scores.get(intent, 0.0), 2)

    expanded_tokens = expand_query(tokens, entities["mood"])

    result = {
        "original_query": raw_query,
        "tokens": tokens,
        "negated": sorted(negated),
        "entities": entities,
        "intent": intent,
        "intent_confidence": intent_confidence,
        "expansion": expanded_tokens,
    }

    if use_semantic_fallback and (intent == "unknown" or intent_confidence < low_confidence_threshold):
        result["embedding"] = embed_query(" ".join(expanded_tokens) or raw_query)

    return result


if __name__ == "__main__":
    from data_preprocessing import load_datasets

    datasets = load_datasets()
    known_tracks, known_artists = build_known_entities(datasets.catalog_df)

    for q in [
        "something moody and reflective",
        " shape off you   ",
        "not love songs",
        "sad songs by adele",
    ]:
        result = rectify_query(q, tuple(known_tracks), tuple(known_artists))
        print(q, "->", result["intent"], result["entities"])
