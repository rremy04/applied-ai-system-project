import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def _prefs(self, user: UserProfile) -> Dict:
        """Translate a UserProfile into the dict shape score_song expects."""
        return {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "acousticness": 0.8 if user.likes_acoustic else 0.2,
        }

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top k songs for this user, best first (ties break by title)."""
        prefs = self._prefs(user)
        ranked = sorted(
            self.songs,
            key=lambda s: (-score_song(prefs, asdict(s))[0], s.title),
        )
        return ranked[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Explain why a song scored the way it did, using score_song's reasons."""
        _, reasons = score_song(self._prefs(user), asdict(song))
        return "; ".join(reasons) or "no strong matches"

def build_description(song: Dict) -> str:
    """
    Build the text the retriever searches over (hybrid vocabulary):
    words generated from the song's numbers, plus the hand-written `notes` column.
    """
    words = [song["genre"], song["mood"]]

    energy = song["energy"]
    words.append("low energy" if energy < 0.35 else "high energy" if energy > 0.65 else "medium energy")

    if song["valence"] >= 0.7:
        words.append("upbeat positive")
    elif song["valence"] <= 0.4:
        words.append("dark somber")

    if song["acousticness"] >= 0.7:
        words.append("acoustic")
    elif song["acousticness"] <= 0.2:
        words.append("electronic produced")

    if song["danceability"] >= 0.75:
        words.append("danceable")

    if song["tempo_bpm"] < 80:
        words.append("slow tempo")
    elif song["tempo_bpm"] >= 120:
        words.append("fast tempo")

    notes = song.get("notes", "")
    if notes:
        words.append(notes)
    return ", ".join(words)


def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV file into dicts, converting numeric columns to int/float.
    Each song also gets a `description` built by build_description."""
    int_fields = ("id",)
    float_fields = ("energy", "tempo_bpm", "valence", "danceability", "acousticness")

    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key in int_fields:
                row[key] = int(row[key])
            for key in float_fields:
                row[key] = float(row[key])
            row["description"] = build_description(row)
            songs.append(row)
    return songs

MOOD_POINTS = 2.0
GENRE_POINTS = 1.0
NUMERIC_WEIGHTS = {"energy": 2.0, "valence": 1.0, "acousticness": 1.0, "danceability": 0.5}

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score one song against the user's prefs; return (points, list of reasons)."""
    score = 0.0
    reasons = []

    if song["mood"] == user_prefs.get("mood"):
        score += MOOD_POINTS
        reasons.append(f"mood matches: {song['mood']} (+{MOOD_POINTS:.1f})")

    if song["genre"] == user_prefs.get("genre"):
        score += GENRE_POINTS
        reasons.append(f"genre matches: {song['genre']} (+{GENRE_POINTS:.1f})")

    # Numeric features: points = weight * (1 - |song - target|). Skipped if the user gave no target.
    for feature, weight in NUMERIC_WEIGHTS.items():
        target = user_prefs.get(feature)
        if target is None:
            continue
        points = weight * (1 - abs(song[feature] - target))
        score += points
        reasons.append(f"{feature} {song[feature]:.2f} vs target {target:.2f} (+{points:.2f})")

    return score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Score every song and return the top k as (song, score, explanation), best first."""
    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        scored.append((song, score, "; ".join(reasons)))

    # Highest score first; ties break alphabetically by title.
    ranked = sorted(scored, key=lambda item: (-item[1], item[0]["title"]))
    return ranked[:k]
