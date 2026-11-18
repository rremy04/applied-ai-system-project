"""
Shared vocabulary tables for the retriever and the fallback parser.

- Synonym normalization: map everyday words ("sad", "rap") onto the catalog's own
  moods and genres so the scorer's exact-match rules can fire.
- Compatibility: what a mood implies about the numbers, and which moods clash.
"""

import re
from typing import Dict, List, Set

# catalog mood -> everyday words that mean (roughly) the same thing
_MOOD_WORDS: Dict[str, List[str]] = {
    "happy": ["cheerful", "joyful", "bubbly", "fun", "sunny", "glad"],
    "chill": ["mellow", "calm", "laid-back", "soothing", "easygoing", "quiet", "soft"],
    "relaxed": ["relaxing", "peaceful", "cozy", "unwinding"],
    "intense": ["powerful", "fierce", "hype", "pumped"],
    "moody": ["dark", "brooding", "atmospheric", "edgy", "noir"],
    "focused": ["focus", "study", "studying", "concentration", "productive"],
    "confident": ["bold", "swagger", "boastful"],
    "melancholic": ["sad", "sorrowful", "gloomy", "somber", "lonely", "depressing", "blue"],
    "nostalgic": ["wistful", "retro", "throwback", "reminiscent"],
    "romantic": ["sensual", "loving", "love", "intimate"],
    "angry": ["furious", "rage", "aggressive", "mad", "brutal"],
    "uplifting": ["positive", "inspiring", "hopeful", "feel-good"],
    "euphoric": ["ecstatic", "blissful"],
    "dreamy": ["floaty", "ethereal", "hazy", "sleepy"],
}

# catalog genre -> alternative names
_GENRE_WORDS: Dict[str, List[str]] = {
    "lofi": ["lo-fi", "lo fi"],
    "hip hop": ["hip-hop", "hiphop", "rap"],
    "r&b": ["rnb", "r and b"],
    "edm": ["electronic", "techno", "house"],
    "synthwave": ["synth", "retrowave", "synth-wave"],
    "metal": ["heavy metal"],
    "indie pop": ["indie"],
    "classical": ["orchestral"],
}

MOOD_SYNONYMS: Dict[str, str] = {w: m for m, ws in _MOOD_WORDS.items() for w in ws}
GENRE_SYNONYMS: Dict[str, str] = {w: g for g, ws in _GENRE_WORDS.items() for w in ws}


def canonical_mood(word: str) -> str:
    word = word.lower().strip()
    return MOOD_SYNONYMS.get(word, word)


def canonical_genre(word: str) -> str:
    word = word.lower().strip()
    return GENRE_SYNONYMS.get(word, word)


def find_phrases(text: str, mapping: Dict[str, str]) -> List[str]:
    """Canonical values whose synonym appears in `text` as a whole word or phrase."""
    text = text.lower()
    found = []
    for phrase, canonical in mapping.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text):
            if canonical not in found:
                found.append(canonical)
    return found


# ---------------------------------------------------------------- compatibility

# What a mood implies about the numbers. Only fills targets the user did not give.
MOOD_IMPLIED_FEATURES: Dict[str, Dict[str, float]] = {
    "melancholic": {"energy": 0.25, "valence": 0.25},
    "chill": {"energy": 0.35},
    "relaxed": {"energy": 0.35},
    "dreamy": {"energy": 0.3},
    "focused": {"energy": 0.4},
    "intense": {"energy": 0.9},
    "angry": {"energy": 0.95},
    "euphoric": {"energy": 0.95, "valence": 0.8},
    "happy": {"valence": 0.8},
    "uplifting": {"valence": 0.85},
}

_CALM = {"chill", "relaxed", "dreamy", "focused", "melancholic", "nostalgic", "romantic"}
_LOUD = {"intense", "angry", "euphoric"}

# requested mood -> song moods that clash with it
MOOD_CONFLICTS: Dict[str, Set[str]] = {
    "melancholic": {"happy", "uplifting", "euphoric", "confident", "intense", "angry"},
    "chill": set(_LOUD), "relaxed": set(_LOUD), "dreamy": set(_LOUD), "focused": set(_LOUD),
    "intense": set(_CALM), "angry": set(_CALM), "euphoric": set(_CALM),
    "happy": {"melancholic", "angry"}, "uplifting": {"melancholic", "angry"},
    "romantic": {"angry", "intense"},
    "moody": {"happy", "uplifting", "euphoric"},
    "nostalgic": {"angry", "euphoric", "intense"},
    "confident": {"melancholic"},
}

CONFLICT_PENALTY = 0.5  # multiplier on a song's combined score
