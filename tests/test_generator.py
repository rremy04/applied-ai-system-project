import json
from types import SimpleNamespace

import pytest

from src.config import DEFAULT_MODELS, get_model
from src.generator import (
    GenerationError,
    build_prompt,
    generate_recommendations,
    parse_generation,
)


def retrieved(id, title, combined=0.8):
    song = dict(id=id, title=title, artist="Some Artist", genre="lofi", mood="chill",
                energy=0.35, valence=0.6, acousticness=0.86, tempo_bpm=72)
    return {"song": song, "text_sim": 0.5, "struct_norm": 0.9, "combined": combined,
            "reasons": ["mood matches: chill (+2.0)"]}


RESULTS = [retrieved(4, "Library Rain"), retrieved(9, "Focus Flow", 0.6)]


class FakeClient:
    """Mimics an LLM client's messages.create; records what it was sent."""

    def __init__(self, text=None, error=None):
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)
        self._text, self._error = text, error

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return SimpleNamespace(content=[SimpleNamespace(text=self._text)])


def answer(*recs, summary="Here you go."):
    return json.dumps({"summary": summary, "recommendations": list(recs)})


def test_prompt_contains_query_ids_scores_and_reasons():
    prompt = build_prompt("rainy study music", RESULTS)
    assert "rainy study music" in prompt
    assert "ID 4: Library Rain by Some Artist" in prompt
    assert "ID 9: Focus Flow" in prompt
    assert "retrieval score: 0.80" in prompt
    assert "mood matches: chill" in prompt


def test_prompt_with_no_candidates_says_so():
    assert "no candidates" in build_prompt("anything", [])


def test_generate_sends_rules_and_prompt_and_parses_answer():
    client = FakeClient(answer({"song_id": 4, "reason": "Quiet and rainy."}))
    result = generate_recommendations("rainy study music", RESULTS, client=client)

    sent = client.calls[0]
    assert "ONLY songs from the candidate list" in sent["system"]
    assert "rainy study music" in sent["messages"][0]["content"]
    assert sent["model"] == get_model()
    assert result["recommendations"] == [{"song_id": 4, "reason": "Quiet and rainy."}]
    assert result["summary"] == "Here you go."
    assert result["unknown_ids"] == []


def test_hallucinated_ids_are_reported_not_hidden():
    client = FakeClient(answer(
        {"song_id": 4, "reason": "Fits."}, {"song_id": 999, "reason": "Invented song."}
    ))
    result = generate_recommendations("x", RESULTS, client=client)
    assert result["unknown_ids"] == [999]
    assert len(result["recommendations"]) == 2


def test_code_fence_is_tolerated():
    client = FakeClient("```json\n" + answer({"song_id": 9, "reason": "Focus."}) + "\n```")
    assert generate_recommendations("x", RESULTS, client=client)["recommendations"][0]["song_id"] == 9


def test_malformed_items_and_duplicates_are_cleaned():
    parsed = parse_generation({
        "summary": 5,
        "recommendations": [
            {"song_id": "4", "reason": "string id"},
            {"song_id": True, "reason": "bool id"},
            {"song_id": 4, "reason": "  "},
            {"song_id": 4, "reason": "good"},
            {"song_id": 4, "reason": "duplicate"},
            "junk",
        ],
    }, [4])
    assert parsed["recommendations"] == [{"song_id": 4, "reason": "good"}]
    assert parsed["summary"] == ""


def test_empty_recommendations_is_valid():
    result = generate_recommendations("x", RESULTS, client=FakeClient(answer(summary="Nothing fits.")))
    assert result["recommendations"] == [] and result["summary"] == "Nothing fits."


@pytest.mark.parametrize("bad", ["not json", "[1, 2]", '{"summary": "no list"}'])
def test_unusable_output_raises(bad):
    with pytest.raises(GenerationError):
        generate_recommendations("x", RESULTS, client=FakeClient(bad))


def test_api_error_raises_generation_error():
    with pytest.raises(GenerationError):
        generate_recommendations("x", RESULTS, client=FakeClient(error=TimeoutError("slow")))


def test_no_client_raises(monkeypatch):
    with pytest.raises(GenerationError, match="GEMINI_API_KEY"):
        generate_recommendations("x", RESULTS)


def test_model_comes_from_config(monkeypatch):
    assert get_model() == DEFAULT_MODELS["gemini"]
    monkeypatch.setenv("LLM_MODEL", "some-other-model")
    assert get_model() == "some-other-model"
    client = FakeClient(answer({"song_id": 4, "reason": "ok"}))
    generate_recommendations("x", RESULTS, client=client)
    assert client.calls[0]["model"] == "some-other-model"
