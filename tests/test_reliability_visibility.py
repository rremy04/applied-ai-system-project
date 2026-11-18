"""Silent fallbacks must be visible, and quota errors must not be retried."""
import json
from types import SimpleNamespace

from src.generator import GenerationError
from src.llm_errors import is_rate_limit, short_reason
from src.pipeline import RecommendationPipeline
from src.query_parser import parse_query


class Quota429(Exception):
    code = 429


def test_rate_limit_detection():
    assert is_rate_limit(Quota429("anything"))
    assert is_rate_limit(RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded"))
    assert not is_rate_limit(TimeoutError("slow"))
    assert not is_rate_limit(ValueError("bad JSON"))


def test_short_reason_hides_the_provider_payload():
    reason = short_reason(RuntimeError("429 RESOURCE_EXHAUSTED " + "x" * 2000))
    assert reason.startswith("rate or daily quota limit") and len(reason) < 80


def test_parse_failure_is_recorded_as_an_event():
    class Broken:
        messages = SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(TimeoutError("slow")))

    events = []
    parsed = parse_query("chill lofi", {"genres": ["lofi"], "moods": ["chill"]}, client=Broken(), events=events)
    assert parsed["genres"] == ["lofi"]  # keyword fallback still worked
    assert len(events) == 1 and events[0].startswith("parse fell back to keyword parser")


def test_no_client_is_expected_and_not_an_event():
    events = []
    parse_query("chill lofi", {"genres": ["lofi"], "moods": ["chill"]}, client=None, events=events)
    assert events == []


class CountingQuota:
    """Every call to the model is refused with a quota error."""

    def __init__(self):
        self.calls = 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls += 1
        if kwargs["system"].startswith("You convert"):
            return SimpleNamespace(content=[SimpleNamespace(text=json.dumps({"moods": ["chill"]}))])
        raise Quota429("429 RESOURCE_EXHAUSTED: daily quota")


def test_quota_error_is_not_retried_and_says_why():
    client = CountingQuota()
    result = RecommendationPipeline(client=client).answer_query("mellow music for studying")
    assert result["source"] == "fallback"
    assert client.calls == 2  # one parse call + ONE generation call: no retry
    assert any("not retrying" in e and "quota" in e for e in result["guardrail_events"])
    assert not any("retrying once" in e for e in result["guardrail_events"])


def test_non_quota_generation_error_still_retries(monkeypatch):
    calls = []

    def fake_generate(*args, **kwargs):
        calls.append(kwargs.get("feedback"))
        err = GenerationError("generation failed: TimeoutError")
        err.rate_limited = False
        raise err

    monkeypatch.setattr("src.pipeline.generate_recommendations", fake_generate)
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: SimpleNamespace(
        content=[SimpleNamespace(text=json.dumps({"moods": ["chill"]}))])))
    result = RecommendationPipeline(client=client).answer_query("mellow music for studying")
    assert len(calls) == 2 and result["source"] == "fallback"
