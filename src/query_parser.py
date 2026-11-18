"""
Turn a free-text request ("dark synthwave for a night drive") into a normalized query.

The output shape is independent of the retriever's input format:

    {
        "genres": ["synthwave"],
        "moods": ["dark", "moody"],
        "scenes": ["night drive"],
        "synonyms": ["atmospheric"],
        "numeric_features": {"energy": 0.7, "tempo": 0.6},   # every value in 0..1
    }

The LLM does the parsing when an API key is available. Otherwise (no key, API error,
bad JSON) a keyword heuristic produces the same shape, so the system keeps working.
"""

import logging
import re
from typing import Dict, List, Optional

from .config import get_model
from .llm_errors import short_reason
from .llm_utils import extract_json, make_client
from .vocabulary import GENRE_SYNONYMS, MOOD_SYNONYMS, find_phrases

logger = logging.getLogger(__name__)

LIST_FIELDS = ("genres", "moods", "scenes", "synonyms")
NUMERIC_FIELDS = ("energy", "valence", "acousticness", "danceability", "tempo")

SYSTEM_PROMPT = """You convert a music request into JSON. Reply with JSON only, no prose.
Schema:
{"genres": [str], "moods": [str], "scenes": [str], "synonyms": [str],
 "numeric_features": {"energy": 0-1, "valence": 0-1, "acousticness": 0-1,
                      "danceability": 0-1, "tempo": 0-1}}
Rules:
- Only include numeric features the request implies. tempo 0 = very slow, 1 = very fast.
- moods/genres: prefer words from the catalog lists given by the user when they fit.
- scenes: situations or places (e.g. "night drive", "studying"). synonyms: related descriptive words.
- Use lowercase. Use empty lists when nothing applies."""


def empty_query() -> Dict:
    return {"genres": [], "moods": [], "scenes": [], "synonyms": [], "numeric_features": {}}


def normalize_query(raw: object) -> Dict:
    """Coerce anything (e.g. model output) into the normalized shape. Never raises."""
    result = empty_query()
    if not isinstance(raw, dict):
        return result

    for field in LIST_FIELDS:
        values = raw.get(field)
        if isinstance(values, str):
            values = [values]
        if isinstance(values, list):
            cleaned = [v.strip().lower() for v in values if isinstance(v, str) and v.strip()]
            result[field] = list(dict.fromkeys(cleaned))  # de-duplicate, keep order

    numerics = raw.get("numeric_features")
    if isinstance(numerics, dict):
        for name in NUMERIC_FIELDS:
            value = numerics.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result["numeric_features"][name] = min(1.0, max(0.0, float(value)))
    return result


# ---------------------------------------------------------------- heuristic fallback

_STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "to", "of", "in", "on", "with", "something",
    "some", "music", "song", "songs", "play", "me", "i", "want", "need", "like", "that",
    "is", "it", "my", "but", "not", "too", "very", "really",
    "at", "by", "from", "into", "about", "up", "out", "as", "be", "so", "just",
}

# keyword -> (feature, target)
_NUMERIC_KEYWORDS = {
    "energetic": ("energy", 0.85), "loud": ("energy", 0.9), "intense": ("energy", 0.9),
    "hype": ("energy", 0.9), "mellow": ("energy", 0.3), "calm": ("energy", 0.25),
    "quiet": ("energy", 0.25), "soft": ("energy", 0.3), "relaxing": ("energy", 0.3),
    "acoustic": ("acousticness", 0.85), "electronic": ("acousticness", 0.15),
    "danceable": ("danceability", 0.85), "dance": ("danceability", 0.85),
    "upbeat": ("valence", 0.85), "happy": ("valence", 0.85), "cheerful": ("valence", 0.85),
    "sad": ("valence", 0.2), "dark": ("valence", 0.25), "gloomy": ("valence", 0.2),
    "fast": ("tempo", 0.85), "slow": ("tempo", 0.2),
}


def heuristic_parse(query: str, vocab: Optional[Dict[str, List[str]]] = None) -> Dict:
    """Keyword-based parse used when the LLM is unavailable. Same output shape."""
    vocab = vocab or {}
    tokens = re.findall(r"[a-z0-9&][a-z0-9&\-]*", query.lower())
    result = empty_query()

    genres = {g.lower() for g in vocab.get("genres", [])}
    moods = {m.lower() for m in vocab.get("moods", [])}
    lowered = query.lower()

    result["genres"] = sorted(
        g for g in genres if re.search(rf"(?<![a-z0-9]){re.escape(g)}(?![a-z0-9])", lowered)
    )
    result["moods"] = [t for t in tokens if t in moods]

    # Synonym normalization: "sad" -> "melancholic", "rap" -> "hip hop", when the catalog has them.
    for token in tokens:
        mood = MOOD_SYNONYMS.get(token)
        if mood in moods and mood not in result["moods"]:
            result["moods"].append(mood)
    for genre in find_phrases(query, GENRE_SYNONYMS):
        if genre in genres and genre not in result["genres"]:
            result["genres"].append(genre)

    for token in tokens:
        if token in _NUMERIC_KEYWORDS:
            feature, target = _NUMERIC_KEYWORDS[token]
            result["numeric_features"].setdefault(feature, target)

    # Everything else becomes a synonym term so text retrieval can still use it.
    used = {t for t in tokens if t in moods}
    result["synonyms"] = [t for t in tokens if t not in _STOPWORDS and t not in used]
    return normalize_query(result)


# ---------------------------------------------------------------- LLM parsing

def parse_query(
    query: str,
    vocab: Optional[Dict[str, List[str]]] = None,
    client=None,
    model: Optional[str] = None,
    events: Optional[List[str]] = None,
) -> Dict:
    """
    Parse a free-text request into the normalized query shape.

    `vocab` ({"genres": [...], "moods": [...]}) lists catalog values; it is shown to the model
    as a hint and used by the fallback. Pass `client` to inject a fake in tests. If the model
    call fails, a note is appended to `events` (when given) so the fallback is never silent.
    """
    if not isinstance(query, str) or not query.strip():
        logger.info("Empty query; returning empty normalized query")
        return empty_query()

    client = client or make_client()
    if client is None:
        logger.info("No API client available; using heuristic parser")
        return heuristic_parse(query, vocab)

    vocab = vocab or {}
    user_message = (
        f"Catalog genres: {', '.join(vocab.get('genres', []))}\n"
        f"Catalog moods: {', '.join(vocab.get('moods', []))}\n\n"
        f"Request: {query}"
    )
    try:
        response = client.messages.create(
            model=model or get_model(),
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        parsed = normalize_query(extract_json(response.content[0].text))
        logger.info("LLM parsed query %r -> %s", query, parsed)
        return parsed
    except Exception as exc:  # API error, timeout, bad JSON: never crash the pipeline
        reason = short_reason(exc)
        logger.warning("LLM parse failed (%s); using heuristic parser", reason)
        if events is not None:
            events.append(f"parse fell back to keyword parser: {reason}")
        return heuristic_parse(query, vocab)
