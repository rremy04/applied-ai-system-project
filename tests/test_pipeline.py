import json
import logging
from types import SimpleNamespace

import pytest

from src import config, llm_utils
from src.logging_setup import setup_logging
from src.pipeline import RecommendationPipeline, answer_query

PARSE_OK = json.dumps({"moods": ["chill"], "scenes": ["studying"],
                       "numeric_features": {"energy": 0.3}})


def answer(*ids, summary="Here are some picks."):
    return json.dumps({
        "summary": summary,
        "recommendations": [{"song_id": i, "reason": f"reason for {i}"} for i in ids],
    })


class ScriptedClient:
    """Answers the parser call and the generation calls separately, in order."""

    def __init__(self, generations, parse=PARSE_OK):
        self.generations = list(generations)  # strings, or exceptions to raise
        self.parse = parse
        self.generation_prompts = []
        self.parse_calls = 0
        self.messages = SimpleNamespace(create=self._create)

    @property
    def generation_calls(self):
        return len(self.generation_prompts)

    def _create(self, **kwargs):
        if kwargs["system"].startswith("You convert"):
            self.parse_calls += 1
            return SimpleNamespace(content=[SimpleNamespace(text=self.parse)])
        self.generation_prompts.append(kwargs["messages"][0]["content"])
        item = self.generations.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(content=[SimpleNamespace(text=item)])


def run(generations, query="mellow music for studying", **kw):
    client = ScriptedClient(generations, **kw)
    result = RecommendationPipeline(client=client).answer_query(query)
    return result, client


def top_ids(result, n=3):
    return [r["song"]["id"] for r in result["retrieved"][:n]]


def test_happy_path_uses_the_model_and_attaches_titles():
    result, client = run([answer(4)])
    assert result["source"] == "llm"
    assert result["guardrail_events"] == []
    rec = result["recommendations"][0]
    assert rec == {"song_id": 4, "title": "Library Rain", "artist": "Paper Lanterns",
                   "reason": "reason for 4"}
    assert client.generation_calls == 1
    assert set(result) >= {"query", "parsed_query", "retrieved", "summary", "recommendations",
                           "source", "guardrail_events", "latency_s"}


def test_partial_invalid_ids_are_dropped_without_retry():
    result, client = run([answer(4, 999)])
    assert [r["song_id"] for r in result["recommendations"]] == [4]
    assert result["source"] == "llm"
    assert client.generation_calls == 1
    assert any("dropped invalid song IDs [999]" in e for e in result["guardrail_events"])


def test_all_invalid_retries_once_with_feedback_then_succeeds():
    result, client = run([answer(999, 998), answer(4)])
    assert result["source"] == "llm_retry"
    assert [r["song_id"] for r in result["recommendations"]] == [4]
    assert client.generation_calls == 2
    assert "rejected" in client.generation_prompts[1] and "999" in client.generation_prompts[1]
    assert "rejected" not in client.generation_prompts[0]


def test_retry_that_fails_falls_back_to_scorer_after_exactly_two_calls():
    result, client = run([answer(999), answer(998)])
    assert result["source"] == "fallback"
    assert client.generation_calls == 2  # retry limit is one
    assert [r["song_id"] for r in result["recommendations"]] == top_ids(result)
    assert all(r["reason"] for r in result["recommendations"])
    assert any("retry failed" in e for e in result["guardrail_events"])


def test_unparseable_answer_and_api_error_are_retried_then_fall_back():
    result, client = run(["not json", TimeoutError("slow")])
    assert result["source"] == "fallback"
    assert client.generation_calls == 2


def test_api_error_then_good_retry():
    result, _ = run([TimeoutError("slow"), answer(4)])
    assert result["source"] == "llm_retry"


def test_model_saying_nothing_fits_is_respected_not_retried():
    result, client = run([answer(summary="Nothing here matches.")])
    assert result["source"] == "llm"
    assert result["recommendations"] == []
    assert result["summary"] == "Nothing here matches."
    assert client.generation_calls == 1


def test_no_client_goes_straight_to_fallback(monkeypatch):
    result = RecommendationPipeline().answer_query("mellow music for studying")
    assert result["source"] == "fallback"
    assert result["recommendations"]
    assert any("no API client" in e for e in result["guardrail_events"])


@pytest.mark.parametrize("query", ["", "   ", None])
def test_empty_query_makes_no_calls(query):
    client = ScriptedClient([])
    result = RecommendationPipeline(client=client).answer_query(query)
    assert result["source"] == "no_match" and result["recommendations"] == []
    assert client.parse_calls == 0 and client.generation_calls == 0


def test_guardrail_events_are_logged(caplog):
    with caplog.at_level(logging.INFO):
        run([answer(4, 999)])
    text = caplog.text
    assert "dropped invalid song IDs [999]" in text
    assert "answered" in text and "source=llm" in text


def test_module_level_answer_query_accepts_a_pipeline():
    pipeline = RecommendationPipeline(client=ScriptedClient([answer(4)]))
    assert answer_query("mellow music for studying", pipeline=pipeline)["source"] == "llm"


# ---- timeout + logging setup ----

def test_timeout_config(monkeypatch):
    monkeypatch.delenv("LLM_TIMEOUT_SECONDS", raising=False)
    assert config.get_timeout() == config.DEFAULT_TIMEOUT_SECONDS
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "5")
    assert config.get_timeout() == 5.0
    for bad in ("abc", "-3", "0"):
        monkeypatch.setenv("LLM_TIMEOUT_SECONDS", bad)
        assert config.get_timeout() == config.DEFAULT_TIMEOUT_SECONDS


def test_client_is_built_with_timeout_and_no_sdk_retries(monkeypatch):
    import anthropic
    seen = {}
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: seen.update(kw) or object())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "7")
    assert llm_utils.make_client() is not None
    assert seen == {"api_key": "test-key", "timeout": 7.0, "max_retries": 0}


def test_setup_logging_writes_file_and_is_idempotent(tmp_path):
    log_file = tmp_path / "logs" / "app.log"
    logger = logging.getLogger("src")
    before = list(logger.handlers)
    try:
        setup_logging(log_file)
        setup_logging(log_file)  # twice: must not duplicate handlers
        assert len(logger.handlers) == len(before) + 2
        logging.getLogger("src.pipeline").info("hello log file")
        for handler in logger.handlers:
            handler.flush()
        assert "hello log file" in log_file.read_text()
    finally:
        for handler in [h for h in logger.handlers if h not in before]:
            logger.removeHandler(handler)
            handler.close()


# ---- zero-signal retrieval and crash handling ----

def test_query_that_matches_nothing_is_no_match_and_skips_the_model():
    client = ScriptedClient([], parse=json.dumps({}))
    result = RecommendationPipeline(client=client).answer_query("asdfgh qwerty")
    assert result["source"] == "no_match"
    assert result["recommendations"] == []
    assert "couldn't find" in result["summary"]
    assert client.generation_calls == 0
    assert any("no retrieval signal" in e for e in result["guardrail_events"])


def test_unexpected_exception_becomes_error_result():
    pipeline = RecommendationPipeline(client=ScriptedClient([answer(4)]))

    def boom(*args, **kwargs):
        raise RuntimeError("retriever exploded")

    pipeline.retriever.retrieve = boom
    result = pipeline.answer_query("mellow music for studying")
    assert result["source"] == "error"
    assert result["recommendations"] == []
    assert any("retriever exploded" in e for e in result["guardrail_events"])


def test_module_answer_query_survives_pipeline_that_cannot_start(monkeypatch):
    import src.pipeline as pipeline_module

    class CannotStart:
        _finish = staticmethod(pipeline_module.RecommendationPipeline._finish)

        def __init__(self, *args, **kwargs):
            raise FileNotFoundError("songs.csv is missing")

    monkeypatch.setattr(pipeline_module, "_default_pipeline", None)
    monkeypatch.setattr(pipeline_module, "RecommendationPipeline", CannotStart)
    result = pipeline_module.answer_query("anything")
    assert result["source"] == "error"
    assert result["recommendations"] == []
    assert any("songs.csv is missing" in e for e in result["guardrail_events"])
