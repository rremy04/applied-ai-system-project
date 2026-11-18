"""
Hybrid retriever: TF-IDF text similarity + the weighted scorer from recommender.py.

Input is the normalized query from query_parser.py (this module never calls the parser
or any API). Output is a list of dicts, best first, each with its score breakdown and
human-readable reasons.
"""

import logging
from typing import Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .recommender import (
    GENRE_POINTS,
    MOOD_POINTS,
    NUMERIC_WEIGHTS,
    build_description,
    score_song,
)
from .vocabulary import (
    CONFLICT_PENALTY,
    MOOD_CONFLICTS,
    MOOD_IMPLIED_FEATURES,
    canonical_genre,
    canonical_mood,
)

logger = logging.getLogger(__name__)

# Keeps "r&b" and hyphenated words as single tokens (the default pattern would split them).
TOKEN_PATTERN = r"[a-z0-9][a-z0-9&\-]*"


def _tempo_words(tempo: float) -> List[str]:
    if tempo <= 0.35:
        return ["slow", "tempo"]
    if tempo >= 0.65:
        return ["fast", "tempo"]
    return []


def query_to_text(query: Dict) -> str:
    """Flatten a normalized query into text for TF-IDF."""
    parts: List[str] = []
    for field in ("genres", "moods", "scenes", "synonyms"):
        parts.extend(query.get(field, []))
    # Also search the catalog's own words ("sad" -> "melancholic") so descriptions match.
    parts.extend(canonical_mood(m) for m in query.get("moods", []))
    parts.extend(canonical_genre(g) for g in query.get("genres", []))
    tempo = query.get("numeric_features", {}).get("tempo")
    if tempo is not None:
        parts.extend(_tempo_words(tempo))
    return " ".join(parts)


def requested_moods(query: Dict, catalog_moods: Optional[List[str]] = None) -> List[str]:
    """Requested moods after synonym normalization, limited to catalog moods if given."""
    valid = {m.lower() for m in catalog_moods} if catalog_moods else None
    moods = []
    for mood in query.get("moods", []):
        mood = canonical_mood(mood)
        if (valid is None or mood in valid) and mood not in moods:
            moods.append(mood)
    return moods


def query_to_prefs(
    query: Dict,
    catalog_genres: Optional[List[str]] = None,
    catalog_moods: Optional[List[str]] = None,
) -> Dict:
    """
    Convert a normalized query into the dict score_song expects.

    - Synonyms are normalized first ("sad" -> "melancholic", "rap" -> "hip hop").
    - score_song matches one genre and one mood exactly, so we take the first requested
      value that exists in the catalog.
    - The chosen mood fills in numeric targets the user did not give (a "sad" request
      implies low energy), so one matching number cannot carry an incompatible song.
    - `tempo` has no scorer weight and is used on the text side only.
    """
    prefs: Dict = {}
    genres = {g.lower() for g in catalog_genres} if catalog_genres else None

    for genre in query.get("genres", []):
        genre = canonical_genre(genre)
        if genres is None or genre in genres:
            prefs["genre"] = genre
            break

    moods = requested_moods(query, catalog_moods)
    if moods:
        prefs["mood"] = moods[0]

    for feature in NUMERIC_WEIGHTS:
        value = query.get("numeric_features", {}).get(feature)
        if value is not None:
            prefs[feature] = value

    for feature, value in MOOD_IMPLIED_FEATURES.get(prefs.get("mood"), {}).items():
        prefs.setdefault(feature, value)
    return prefs


def compatibility_factor(song: Dict, moods: List[str]) -> Tuple[float, Optional[str]]:
    """(multiplier, reason). Penalizes a song whose mood clashes with a requested mood."""
    if song["mood"] in moods:
        return 1.0, None
    for mood in moods:
        if song["mood"] in MOOD_CONFLICTS.get(mood, set()):
            return CONFLICT_PENALTY, (
                f"mood conflict: song is {song['mood']}, request wants {mood} "
                f"(x{CONFLICT_PENALTY})"
            )
    return 1.0, None


def max_possible_score(prefs: Dict) -> float:
    """Highest score_song could award for the preferences actually supplied."""
    total = 0.0
    if prefs.get("mood"):
        total += MOOD_POINTS
    if prefs.get("genre"):
        total += GENRE_POINTS
    for feature, weight in NUMERIC_WEIGHTS.items():
        if prefs.get(feature) is not None:
            total += weight
    return total


class Retriever:
    def __init__(self, songs: List[Dict]):
        if not songs:
            raise ValueError("Retriever needs at least one song")
        self.songs = songs
        self.descriptions = [s.get("description") or build_description(s) for s in songs]
        self.vectorizer = TfidfVectorizer(token_pattern=TOKEN_PATTERN)
        self.matrix = self.vectorizer.fit_transform(self.descriptions)
        self._analyze = self.vectorizer.build_analyzer()
        self.genres = sorted({s["genre"] for s in songs})
        self.moods = sorted({s["mood"] for s in songs})

    def retrieve(self, query: Dict, k: int = 5, text_weight: float = 0.5) -> List[Dict]:
        """
        Rank songs for a normalized query. Returns up to k dicts:
        song, text_sim, struct_score, struct_norm, combined, compat_factor,
        matched_terms, reasons.
        """
        text = query_to_text(query)
        prefs = query_to_prefs(query, self.genres, self.moods)
        moods = requested_moods(query, self.moods)
        has_text = bool(self._analyze(text))
        has_prefs = bool(prefs)

        if has_text:
            sims = cosine_similarity(self.vectorizer.transform([text]), self.matrix)[0]
        else:
            sims = [0.0] * len(self.songs)
        query_terms = set(self._analyze(text))
        max_score = max_possible_score(prefs)

        results = []
        for song, description, sim in zip(self.songs, self.descriptions, sims):
            struct_score, struct_reasons = score_song(prefs, song) if has_prefs else (0.0, [])
            struct_norm = min(1.0, max(0.0, struct_score / max_score)) if max_score else 0.0

            if has_text and has_prefs:
                combined = text_weight * sim + (1 - text_weight) * struct_norm
            elif has_text:
                combined = float(sim)
            else:
                combined = struct_norm

            factor, conflict_reason = compatibility_factor(song, moods)
            combined *= factor

            matched = sorted(query_terms & set(self._analyze(description)))
            reasons = list(struct_reasons)
            if conflict_reason:
                reasons.append(conflict_reason)
            if matched:
                reasons.append(f"text match ({sim:.2f}): {', '.join(matched)}")

            results.append({
                "song": song,
                "text_sim": float(sim),
                "struct_score": struct_score,
                "struct_norm": struct_norm,
                "combined": float(combined),
                "compat_factor": factor,
                "matched_terms": matched,
                "reasons": reasons,
            })

        results.sort(key=lambda r: (-r["combined"], r["song"]["title"]))
        top = results[:k]
        logger.info(
            "retrieved %s for text=%r prefs=%s",
            [r["song"]["id"] for r in top], text, prefs,
        )
        return top
