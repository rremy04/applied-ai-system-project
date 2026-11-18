from types import SimpleNamespace

from src.query_parser import empty_query, heuristic_parse, normalize_query, parse_query

VOCAB = {"genres": ["lofi", "pop", "indie pop", "synthwave"], "moods": ["chill", "happy", "moody"]}


class FakeClient:
    """Mimics an LLM client's messages.create without any network call."""

    def __init__(self, text=None, error=None):
        self.messages = SimpleNamespace(create=self._create)
        self._text, self._error = text, error

    def _create(self, **kwargs):
        if self._error:
            raise self._error
        return SimpleNamespace(content=[SimpleNamespace(text=self._text)])


def test_normalize_clamps_cleans_and_drops_unknown():
    result = normalize_query({
        "moods": ["Dark", "dark", " ", 5],
        "scenes": "night drive",
        "numeric_features": {"energy": 1.7, "tempo": -2, "bogus": 0.5, "valence": True},
        "extra": "ignored",
    })
    assert result["moods"] == ["dark"]
    assert result["scenes"] == ["night drive"]
    assert result["numeric_features"] == {"energy": 1.0, "tempo": 0.0}
    assert set(result) == set(empty_query())


def test_normalize_handles_garbage():
    assert normalize_query(None) == empty_query()
    assert normalize_query("nope") == empty_query()


def test_heuristic_finds_vocab_and_numeric_keywords():
    result = heuristic_parse("chill lofi to study, low key and mellow", VOCAB)
    assert result["genres"] == ["lofi"]
    assert result["moods"] == ["chill"]
    assert result["numeric_features"]["energy"] == 0.3
    assert "study" in result["synonyms"]


def test_heuristic_genre_needs_whole_word():
    assert heuristic_parse("something popular", VOCAB)["genres"] == []


def test_parse_uses_llm_json_even_in_code_fence():
    client = FakeClient(
        '```json\n{"moods": ["dark"], "scenes": ["night drive"], '
        '"numeric_features": {"energy": 0.8}}\n```'
    )
    result = parse_query("dark night drive", VOCAB, client=client)
    assert result["moods"] == ["dark"]
    assert result["scenes"] == ["night drive"]
    assert result["numeric_features"] == {"energy": 0.8}


def test_parse_falls_back_on_api_error():
    result = parse_query("happy pop", VOCAB, client=FakeClient(error=TimeoutError("slow")))
    assert result["genres"] == ["pop"]
    assert result["moods"] == ["happy"]


def test_parse_falls_back_on_bad_json():
    result = parse_query("chill lofi", VOCAB, client=FakeClient("not json at all"))
    assert result["genres"] == ["lofi"]


def test_parse_without_client_or_key_uses_heuristic(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert parse_query("moody synthwave", VOCAB)["genres"] == ["synthwave"]


def test_parse_empty_query():
    assert parse_query("   ", VOCAB) == empty_query()


def test_stopwords_are_not_kept_as_search_terms():
    # Regression: "at" used to leak into the synonym list.
    result = heuristic_parse("high energy workout at the gym", VOCAB)
    assert "at" not in result["synonyms"]
    assert "gym" in result["synonyms"]


def test_heuristic_normalizes_synonyms_to_catalog_values():
    vocab = {"genres": ["hip hop", "lofi"], "moods": ["melancholic", "chill"]}
    result = heuristic_parse("sad lo-fi and some rap", vocab)
    assert result["moods"] == ["melancholic"]
    assert sorted(result["genres"]) == ["hip hop", "lofi"]
