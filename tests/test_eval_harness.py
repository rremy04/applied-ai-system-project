"""Tests for the evaluation harness itself (not for the recommender's quality)."""
import json
from types import SimpleNamespace

import pytest

from tests import eval_rag
from tests.eval_rag import (
    CountingClient, QUERIES, VARIANT_LABELS, first_hit_rank, is_hit, jaccard,
    render_markdown, run_evaluation, supporting_ok, write_reports,
)

METAL = dict(id=15, genre="metal", mood="angry", energy=0.97, valence=0.2, tempo_bpm=168)
CLASSICAL = dict(id=12, genre="classical", mood="melancholic", energy=0.22, valence=0.25, tempo_bpm=66)
LOFI = dict(id=4, genre="lofi", mood="chill", energy=0.35, valence=0.6, tempo_bpm=72)


def truth(**kw):
    return kw


def test_hit_means_genre_or_mood_is_acceptable():
    t = truth(acceptable={"genre": ["classical"], "mood": ["melancholic"]})
    assert is_hit(CLASSICAL, t) and not is_hit(METAL, t) and not is_hit(LOFI, t)


def test_required_field_blocks_credit_from_mood_alone():
    mood_only = dict(CLASSICAL, genre="folk")  # sad-sounding, but not classical
    t = truth(acceptable={"genre": ["classical"], "mood": ["melancholic"]}, required={"genre": ["classical"]})
    assert is_hit(CLASSICAL, t)
    assert not is_hit(mood_only, t)


def test_supporting_ranges_are_reported_not_required():
    t = truth(acceptable={"mood": ["angry"]}, supporting={"energy": (0.0, 0.4)})
    assert is_hit(METAL, t)                      # still a hit
    assert supporting_ok(METAL, t) is False      # but flagged
    assert supporting_ok(METAL, truth()) is None


def test_first_hit_rank():
    t = truth(acceptable={"mood": ["melancholic"]})
    assert first_hit_rank([METAL, LOFI, CLASSICAL], t) == 3
    assert first_hit_rank([METAL, LOFI], t) is None


def test_zero_score_padding_earns_no_credit():
    # Regression: a title-order padding result (score 0) that happened to match was graded a hit.
    t = truth(acceptable={"mood": ["melancholic"]})
    assert first_hit_rank([CLASSICAL, LOFI], t) == 1                          # no scores: old behaviour
    assert first_hit_rank([CLASSICAL, LOFI], t, scores=[0.0, 0.0]) is None   # padding: no credit
    assert first_hit_rank([LOFI, CLASSICAL], t, scores=[0.4, 0.2]) == 2      # real retrieval still counts


def test_jaccard():
    assert jaccard([1, 2, 3], [2, 3, 4]) == pytest.approx(0.5)
    assert jaccard([], []) == 1.0 and jaccard([1], [2]) == 0.0


def test_query_set_is_well_formed():
    ids = [q["id"] for q in QUERIES]
    assert len(ids) == len(set(ids)) == 14
    assert sum(q["expect"] == "no_match" for q in QUERIES) == 2
    for q in QUERIES:
        if q["expect"] == "hit":
            assert q["acceptable"] and "song_id" not in json.dumps(q)  # attributes, never IDs


def test_offline_run_has_expected_shape_and_needs_no_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-be-ignored-offline")
    monkeypatch.setenv("GEMINI_API_KEY", "also-ignored-offline")
    report = run_evaluation(online=False)
    assert report["mode"] == "offline" and report["n_queries"] == 14
    assert set(report["summary"]) == set(VARIANT_LABELS)
    for s in report["summary"].values():
        assert 0 <= s["top1"] <= s["top3"] <= 1
    assert report["pipeline"]["runs_per_query"] == 1
    assert "api_usage" not in report
    import os
    assert os.environ["ANTHROPIC_API_KEY"] == "should-be-ignored-offline"  # restored afterwards
    assert os.environ["GEMINI_API_KEY"] == "also-ignored-offline"


def test_original_scorer_cannot_refuse_gibberish_but_pipeline_can():
    report = run_evaluation(online=False)
    assert report["summary"]["original_scorer"]["negative_accuracy"] == 0.0
    assert report["summary"]["w=0.50"]["negative_accuracy"] == 1.0
    assert report["pipeline"]["negative_accuracy"] == 1.0


def test_reports_are_written_and_readable(tmp_path):
    report = run_evaluation(online=False)
    md, js = write_reports(report, tmp_path)
    assert "Retrieval: top-1 / top-3 hit rate" in md.read_text()
    assert json.loads(js.read_text())["n_queries"] == 14
    assert "Hybrid misses" in render_markdown(report)


# ---- online path, with a scripted fake client (no network) ----

class FakeLLM:
    """Parser calls get a fixed parse; generation calls recommend scripted IDs."""

    def __init__(self, generation_for):
        self.generation_for = generation_for
        self.messages = SimpleNamespace(create=self._create)
        self.n = 0

    def _create(self, **kwargs):
        usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        if kwargs["system"].startswith("You convert"):
            gibberish = any(word in kwargs["messages"][0]["content"] for word in ("asdfgh", "zxqv"))
            parse = {} if gibberish else {"moods": ["chill"], "numeric_features": {"energy": 0.3}}
            return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(parse))], usage=usage)
        self.n += 1
        ids = self.generation_for(self.n)
        text = json.dumps({"summary": "ok", "recommendations": [{"song_id": i, "reason": "r"} for i in ids]})
        return SimpleNamespace(content=[SimpleNamespace(text=text)], usage=usage)


def test_counting_client_counts_calls_and_tokens():
    counting = CountingClient(FakeLLM(lambda n: [4]))
    counting.messages.create(system="You convert x", messages=[{"role": "user", "content": "x"}], model="m", max_tokens=1)
    counting.messages.create(system="other", messages=[{"role": "user", "content": "x"}], model="m", max_tokens=1)
    assert (counting.calls, counting.input_tokens, counting.output_tokens) == (2, 20, 10)


def test_online_run_measures_consistency_and_guardrails():
    # Always recommends the same retrieved song -> perfectly consistent, passes first try.
    stable = run_evaluation(online=True, client=FakeLLM(lambda n: [6]), runs=3)
    p = stable["pipeline"]
    assert p["top1_consistency"] == 1.0 and p["mean_recommendation_overlap"] == 1.0
    assert p["guardrail_pass_rate"] == 1.0 and p["fallback_rate"] == 0.0
    assert stable["api_usage"]["calls"] > 0 and stable["mode"] == "online"

    # Alternates between two songs -> inconsistent.
    flaky = run_evaluation(online=True, client=FakeLLM(lambda n: [6] if n % 2 else [4]), runs=2)
    assert flaky["pipeline"]["top1_consistency"] < 1.0
    assert flaky["pipeline"]["mean_recommendation_overlap"] < 1.0


def test_online_run_counts_hallucinated_ids_and_fallbacks():
    report = run_evaluation(online=True, client=FakeLLM(lambda n: [999]), runs=1)
    p = report["pipeline"]
    assert p["fallback_rate"] == 1.0 and p["guardrail_pass_rate"] == 0.0
    assert p["negative_accuracy"] == 1.0  # gibberish never reached the model's generation step


def test_online_flag_without_a_key_exits_cleanly(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(eval_rag, "make_client", lambda: None)
    assert eval_rag.main(["--online"]) == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_music_to_fall_asleep_to_is_not_credited_as_a_hit_by_alphabet():
    report = run_evaluation(online=False)
    q12 = next(r for r in report["records"] if r["id"] == "q12")
    for variant in q12["variants"].values():
        assert not any(row["hit"] for row in variant["top"] if not row["has_signal"])


# ---- quota-aware online evaluation ----

class QuotaThenGood:
    """Parse works; generation is refused with a quota error for the first `refuse` calls."""

    def __init__(self, refuse):
        self.refuse = refuse
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        usage = SimpleNamespace(input_tokens=1, output_tokens=1)
        if kwargs["system"].startswith("You convert"):
            return SimpleNamespace(content=[SimpleNamespace(text=json.dumps({"moods": ["chill"]}))], usage=usage)
        if self.refuse > 0:
            self.refuse -= 1
            raise type("Quota429", (Exception,), {"code": 429})("429 RESOURCE_EXHAUSTED quota")
        text = json.dumps({"summary": "ok", "recommendations": [{"song_id": 4, "reason": "r"}]})
        return SimpleNamespace(content=[SimpleNamespace(text=text)], usage=usage)


def test_quota_runs_are_counted_separately_and_excluded_from_rates():
    report = run_evaluation(online=True, client=QuotaThenGood(refuse=3), runs=1)
    p = report["pipeline"]
    assert p["rate_limited_runs"] == 3
    assert p["fallback_rate"] == 0.0          # quota is NOT a fallback problem
    assert p["guardrail_pass_rate"] == 1.0    # the clean runs passed
    assert "INCOMPLETE" in render_markdown(report)


def test_no_quota_means_no_incomplete_warning():
    report = run_evaluation(online=True, client=QuotaThenGood(refuse=0), runs=1)
    assert report["pipeline"]["rate_limited_runs"] == 0
    assert "INCOMPLETE" not in render_markdown(report)


def test_delay_spaces_calls_without_sleeping_on_the_first(monkeypatch):
    from tests import eval_rag as mod
    sleeps = []
    monkeypatch.setattr(mod.time, "sleep", lambda s: sleeps.append(s))
    run_evaluation(online=True, client=QuotaThenGood(refuse=0), runs=1, delay=2.5)
    assert sleeps and set(sleeps) == {2.5} and len(sleeps) == len(QUERIES) - 1
