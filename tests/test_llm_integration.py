"""
Optional real-API test. Skipped unless you opt in, so normal runs never spend money:

    RUN_API_TESTS=1 .venv/bin/python -m pytest tests/test_llm_integration.py -v   # key from .env
"""
import os

import pytest

from src.config import get_api_key
from src.query_parser import parse_query

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (os.environ.get("RUN_API_TESTS") == "1" and get_api_key()),
        reason="set RUN_API_TESTS=1 and a key in .env to run real LLM calls",
    ),
]

VOCAB = {
    "genres": ["lofi", "synthwave", "pop", "metal", "classical"],
    "moods": ["chill", "moody", "happy", "angry", "melancholic"],
}


def test_real_llm_parses_a_scene_request():
    result = parse_query("something dark and atmospheric for a late night drive through the city", VOCAB)
    # The keyword fallback never fills `scenes`, so a non-empty value proves the model answered.
    assert result["scenes"], f"fell back to heuristic parser: {result}"
    assert all(0.0 <= v <= 1.0 for v in result["numeric_features"].values())


def test_real_llm_output_has_full_shape():
    result = parse_query("upbeat pop for a morning run", VOCAB)
    assert set(result) == {"genres", "moods", "scenes", "synonyms", "numeric_features"}


def test_real_llm_generates_only_from_retrieved_songs():
    from src.generator import generate_recommendations
    from src.recommender import load_songs
    from src.retriever import Retriever

    retriever = Retriever(load_songs("data/songs.csv"))
    query = "mellow music for studying on a rainy night"
    results = retriever.retrieve(parse_query(query, VOCAB), k=5)
    out = generate_recommendations(query, results)
    assert out["recommendations"], out
    assert out["unknown_ids"] == [], f"model named songs it was not given: {out}"
