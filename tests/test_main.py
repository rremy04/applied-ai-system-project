import json
import logging
from types import SimpleNamespace

import pytest

from src.main import CLIENT_FAILED_NOTICE, NO_KEY_NOTICE, SOURCE_LABELS, format_result, main, run_interactive
from src.pipeline import RecommendationPipeline


def answer(*ids):
    return json.dumps({
        "summary": "Here are some picks.",
        "recommendations": [{"song_id": i, "reason": f"reason for {i}"} for i in ids],
    })


class ScriptedClient:
    def __init__(self, generations, parse=None):
        self.generations = list(generations)
        self.parse = parse if parse is not None else {"moods": ["chill"], "numeric_features": {"energy": 0.3}}
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        if kwargs["system"].startswith("You convert"):
            text = json.dumps(self.parse)
        else:
            text = self.generations.pop(0)
        return SimpleNamespace(content=[SimpleNamespace(text=text)])


class Recorder:
    """Collects everything main() prints, and feeds scripted lines to input()."""

    def __init__(self, lines=()):
        self.lines, self.out, self.err, self.prompts = list(lines), [], [], []

    def input(self, prompt=""):
        self.prompts.append(prompt)
        if not self.lines:
            raise EOFError
        line = self.lines.pop(0)
        if isinstance(line, BaseException):
            raise line
        return line

    @property
    def text(self):
        return "\n".join(self.out)


def run_main(argv, client=None, lines=(), pipeline=None):
    rec = Recorder(lines)
    logging_calls = []
    if pipeline is None and client is not None:
        pipeline = RecommendationPipeline(client=client)
    code = main(argv, pipeline=pipeline, input_fn=rec.input, out=rec.out.append, err=rec.err.append,
                configure_logging=lambda **kw: logging_calls.append(kw))
    return code, rec, logging_calls


def make_result(source="llm", recs=True, events=()):
    return {
        "summary": "Short summary.", "source": source, "guardrail_events": list(events),
        "recommendations": [{"song_id": 4, "title": "Library Rain", "artist": "Paper Lanterns",
                             "reason": "Quiet and rainy."}] if recs else [],
        "parsed_query": {"moods": ["chill"]}, "latency_s": 0.12,
        "retrieved": [{"song": {"id": 4, "title": "Library Rain"}, "combined": 0.81,
                       "text_sim": 0.5, "struct_norm": 0.9, "compat_factor": 1.0}],
    }


# ---- formatting ----

def test_format_shows_summary_recommendations_and_source():
    text = format_result(make_result())
    assert "Short summary." in text
    assert "1. Library Rain by Paper Lanterns" in text and "Quiet and rainy." in text
    assert "[answered by the AI model]" in text
    assert "Guardrail" not in text and "parsed query" not in text


@pytest.mark.parametrize("source", list(SOURCE_LABELS))
def test_every_source_has_a_label(source):
    assert f"[{SOURCE_LABELS[source]}]" in format_result(make_result(source=source))


def test_format_lists_guardrail_events():
    text = format_result(make_result("fallback", events=["retry failed: x"]))
    assert "Guardrail events:" in text and "  - retry failed: x" in text


def test_debug_adds_parsed_query_and_scores():
    text = format_result(make_result(), debug=True)
    assert "parsed query: {'moods': ['chill']}" in text
    assert "combined 0.81" in text and "text 0.50" in text and "structured 0.90" in text


def test_no_match_prints_no_numbered_list():
    text = format_result(make_result("no_match", recs=False))
    assert "1." not in text and "[no match]" in text


# ---- interactive loop ----

def test_interactive_answers_then_quits():
    client = ScriptedClient([answer(4), answer(6)])
    code, rec, _ = run_main([], client=client, lines=["mellow studying", "something calm", "quit"])
    assert code == 0
    assert rec.text.count("[answered by the AI model]") == 2
    assert "Library Rain" in rec.text and rec.text.rstrip().endswith("Goodbye.")


@pytest.mark.parametrize("ending", ["", "   ", "quit", "EXIT", "q", EOFError(), KeyboardInterrupt()])
def test_every_way_of_leaving_is_clean(ending):
    code, rec, _ = run_main([], client=ScriptedClient([]), lines=[ending])
    assert code == 0 and rec.text.rstrip().endswith("Goodbye.")


def test_interactive_function_is_injectable():
    rec = Recorder(["asdfgh qwerty", ""])
    pipeline = RecommendationPipeline(client=ScriptedClient([], parse={}))
    run_interactive(pipeline, input_fn=rec.input, out=rec.out.append)
    assert "[no match]" in rec.text


# ---- one-shot, demo, flags ----

def test_one_shot_query_answers_and_exits_without_prompting():
    code, rec, _ = run_main(["--query", "mellow studying"], client=ScriptedClient([answer(4)]))
    assert code == 0 and "Library Rain" in rec.text
    assert rec.prompts == []


def test_one_shot_error_returns_nonzero():
    pipeline = RecommendationPipeline(client=ScriptedClient([]))

    def boom(*a, **k):
        raise RuntimeError("down")

    pipeline.retriever.retrieve = boom
    code, rec, _ = run_main(["-q", "anything"], pipeline=pipeline)
    assert code == 1 and "[error]" in rec.text


def test_demo_mode_runs_original_profiles_without_a_pipeline(capsys):
    code, rec, _ = run_main(["--demo"])
    printed = capsys.readouterr().out
    assert code == 0
    assert "Loaded songs: 18" in printed and "Profile: Chill Lofi" in printed
    assert rec.out == []  # demo prints directly, as it always did


def test_demo_and_query_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        run_main(["--demo", "--query", "x"])


def test_debug_flag_reaches_output():
    _, rec, _ = run_main(["-q", "mellow studying", "--debug"], client=ScriptedClient([answer(4)]))
    assert "parsed query:" in rec.text and "retrieved (combined" in rec.text


def test_logging_is_configured_quietly_by_default_and_verbose_on_request():
    _, _, calls = run_main(["-q", "mellow studying"], client=ScriptedClient([answer(4)]))
    assert calls == [{"console_level": logging.WARNING}]
    _, _, calls = run_main(["-q", "mellow studying", "--verbose"], client=ScriptedClient([answer(4)]))
    assert calls == [{"console_level": logging.INFO}]


# ---- startup problems ----

def test_missing_key_prints_notice_and_still_works(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    pipeline = RecommendationPipeline()  # no client
    code, rec, _ = run_main(["-q", "mellow studying"], pipeline=pipeline)
    assert code == 0
    assert "no API key was found" in rec.text
    assert "[scorer fallback" in rec.text


def test_pipeline_startup_failure_is_a_clean_error(monkeypatch):
    import src.main as main_module

    def cannot_start(*a, **k):
        raise FileNotFoundError("songs.csv is missing")

    monkeypatch.setattr(main_module, "RecommendationPipeline", cannot_start)
    rec = Recorder()
    code = main(["-q", "x"], input_fn=rec.input, out=rec.out.append, err=rec.err.append,
                configure_logging=lambda **kw: None)
    assert code == 1
    assert "could not start" in rec.err[0] and "songs.csv is missing" in rec.err[0]


def test_key_set_but_client_failed_gets_the_right_notice(monkeypatch):
    # Regression: a key existed but the SDK would not import, and the app wrongly said "no key".
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setattr("src.pipeline.make_client", lambda: None)  # simulate the SDK failing to import
    pipeline = RecommendationPipeline()
    code, rec, _ = run_main(["-q", "sad piano"], pipeline=pipeline)
    assert CLIENT_FAILED_NOTICE in rec.text and NO_KEY_NOTICE not in rec.text


def test_no_key_at_all_gets_the_no_key_notice(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code, rec, _ = run_main(["-q", "sad piano"], pipeline=RecommendationPipeline(client=None))
    assert NO_KEY_NOTICE in rec.text and CLIENT_FAILED_NOTICE not in rec.text
