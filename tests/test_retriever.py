import pytest

from src.recommender import build_description
from src.retriever import (
    Retriever, compatibility_factor, max_possible_score, query_to_prefs, query_to_text,
)


def song(id, title, genre, mood, energy, notes="", valence=0.5, acousticness=0.5,
         danceability=0.5, tempo_bpm=100):
    s = dict(id=id, title=title, artist="x", genre=genre, mood=mood, energy=energy,
             valence=valence, acousticness=acousticness, danceability=danceability,
             tempo_bpm=tempo_bpm, notes=notes)
    s["description"] = build_description(s)
    return s


SONGS = [
    song(1, "Rain Study", "lofi", "chill", 0.3, "rainy day studying cozy", acousticness=0.8),
    song(2, "Gym Beast", "metal", "angry", 0.95, "gym workout lifting", acousticness=0.05),
    song(3, "Night Cruise", "synthwave", "moody", 0.7, "night drive neon city", acousticness=0.2),
]


def empty(**overrides):
    q = {"genres": [], "moods": [], "scenes": [], "synonyms": [], "numeric_features": {}}
    q.update(overrides)
    return q


def test_description_mixes_generated_and_handwritten_words():
    d = SONGS[0]["description"]
    assert "lofi" in d and "low energy" in d and "acoustic" in d  # generated
    assert "rainy" in d                                           # hand-written


def test_text_only_query_finds_scene_words():
    results = Retriever(SONGS).retrieve(empty(scenes=["night drive"]), k=3)
    assert results[0]["song"]["title"] == "Night Cruise"
    assert results[0]["matched_terms"] == ["drive", "night"]


def test_structured_only_query_uses_scorer():
    results = Retriever(SONGS).retrieve(
        empty(genres=["metal"], moods=["angry"], numeric_features={"energy": 0.95}), k=3
    )
    assert results[0]["song"]["title"] == "Gym Beast"
    assert results[0]["struct_norm"] == pytest.approx(1.0)
    assert any("genre matches" in r for r in results[0]["reasons"])


def test_hybrid_blends_both_signals():
    results = Retriever(SONGS).retrieve(
        empty(synonyms=["studying"], moods=["chill"], numeric_features={"energy": 0.3}), k=3
    )
    top = results[0]
    assert top["song"]["title"] == "Rain Study"
    assert top["combined"] == pytest.approx(0.5 * top["text_sim"] + 0.5 * top["struct_norm"])
    assert any(r.startswith("text match") for r in top["reasons"])


def test_result_shape_and_k():
    results = Retriever(SONGS).retrieve(empty(moods=["chill"]), k=2)
    assert len(results) == 2
    assert set(results[0]) == {
        "song", "text_sim", "struct_score", "struct_norm", "combined", "compat_factor",
        "matched_terms", "reasons",
    }


def test_empty_query_does_not_crash_and_is_deterministic():
    results = Retriever(SONGS).retrieve(empty(), k=3)
    assert [r["song"]["title"] for r in results] == sorted(s["title"] for s in SONGS)
    assert all(r["combined"] == 0.0 for r in results)


def test_unknown_catalog_mood_is_dropped_from_prefs_but_kept_as_text():
    q = empty(moods=["melancholic", "chill"])
    assert query_to_prefs(q, ["lofi"], ["chill", "moody"])["mood"] == "chill"
    assert "melancholic" in query_to_text(q)


def test_tempo_is_text_only():
    q = empty(numeric_features={"tempo": 0.9, "energy": 0.5})
    assert "fast" in query_to_text(q)
    assert "tempo" not in query_to_prefs(q)


def test_max_possible_score_depends_on_supplied_prefs():
    assert max_possible_score({}) == 0.0
    assert max_possible_score({"mood": "chill", "energy": 0.3}) == 4.0


def test_retriever_requires_songs():
    with pytest.raises(ValueError):
        Retriever([])


# ---- compatibility-aware scoring (regression: "sad piano" used to rank metal second) ----

CATALOG_LIKE = [
    song(1, "Moonlit Sonata", "classical", "melancholic", 0.22, "moonlight piano sad lonely",
         valence=0.25, acousticness=0.97, danceability=0.3, tempo_bpm=66),
    song(2, "Iron Requiem", "metal", "angry", 0.97, "rage furious heavy aggressive dark",
         valence=0.20, acousticness=0.04, danceability=0.45, tempo_bpm=168),
    song(3, "Sunrise City", "pop", "happy", 0.82, "sunny morning bright", valence=0.84),
]
SAD_PIANO = empty(moods=["sad"], synonyms=["sad", "piano"], numeric_features={"valence": 0.2})


def test_sad_piano_ranks_classical_above_metal():
    results = Retriever(CATALOG_LIKE).retrieve(SAD_PIANO, k=3)
    titles = [r["song"]["title"] for r in results]
    assert titles[0] == "Moonlit Sonata"
    assert titles.index("Iron Requiem") > titles.index("Moonlit Sonata")
    metal = next(r for r in results if r["song"]["title"] == "Iron Requiem")
    assert metal["compat_factor"] == 0.5
    assert any("mood conflict" in reason for reason in metal["reasons"])


def test_mood_fills_missing_targets_but_never_overrides():
    assert query_to_prefs(empty(moods=["melancholic"]), None, ["melancholic"])["energy"] == 0.25
    explicit = empty(moods=["melancholic"], numeric_features={"energy": 0.8})
    assert query_to_prefs(explicit, None, ["melancholic"])["energy"] == 0.8


def test_compatibility_factor_cases():
    metal, pop = CATALOG_LIKE[1], CATALOG_LIKE[2]
    assert compatibility_factor(metal, ["angry"]) == (1.0, None)           # same mood
    assert compatibility_factor(metal, ["melancholic"])[0] == 0.5          # clash
    assert compatibility_factor(pop, ["moody"])[0] == 0.5                  # happy vs moody clash
    assert compatibility_factor(pop, ["chill"]) == (1.0, None)             # unrelated, no penalty
    assert compatibility_factor(pop, []) == (1.0, None)


# ---- synonym normalization ----

def test_synonyms_map_to_catalog_values():
    prefs = query_to_prefs(
        empty(moods=["sad"], genres=["rap"]), ["hip hop", "pop"], ["melancholic", "happy"]
    )
    assert prefs["mood"] == "melancholic"
    assert prefs["genre"] == "hip hop"


def test_unmappable_mood_still_dropped():
    assert "mood" not in query_to_prefs(empty(moods=["bubbly-ish"]), None, ["happy"])


def test_synonym_text_reaches_descriptions():
    results = Retriever(CATALOG_LIKE).retrieve(empty(moods=["sad"]), k=3)
    assert results[0]["song"]["title"] == "Moonlit Sonata"
    assert "melancholic" in results[0]["matched_terms"]
