"""Provider selection, config, and the Gemini adapter (no network: the SDK client is faked)."""
import logging
from types import SimpleNamespace

import pytest

from src import config, llm_utils
from src.gemini_client import GeminiClient


# ---- provider / model / key selection ----

def test_no_keys_means_no_provider_and_no_client():
    assert config.get_provider() is None and config.get_api_key() is None
    assert llm_utils.make_client() is None


def test_provider_follows_whichever_key_exists(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert config.get_provider() == "anthropic"
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    assert config.get_provider() == "gemini"          # Gemini preferred when both exist
    assert config.get_api_key() == "g"


def test_explicit_provider_wins_and_bad_values_are_ignored(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert config.get_provider() == "anthropic" and config.get_api_key() == "a"
    monkeypatch.setenv("LLM_PROVIDER", "nonsense")
    assert config.get_provider() == "gemini"


def test_google_api_key_is_accepted_for_gemini(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "g")
    assert config.get_provider() == "gemini" and config.get_api_key() == "g"


def test_blank_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    assert config.get_provider() is None


def test_default_model_depends_on_provider_and_can_be_overridden(monkeypatch):
    assert config.get_model() == config.DEFAULT_MODELS["gemini"]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert config.get_model() == config.DEFAULT_MODELS["anthropic"]
    monkeypatch.setenv("LLM_MODEL", "custom-model")
    assert config.get_model() == "custom-model"


def test_make_client_builds_a_gemini_client(monkeypatch):
    seen = {}
    monkeypatch.setattr("src.gemini_client.GeminiClient.__init__",
                        lambda self, key, timeout, sdk_client=None: seen.update(key=key, timeout=timeout))
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "12")
    assert isinstance(llm_utils.make_client(), GeminiClient)
    assert seen == {"key": "g-key", "timeout": 12.0}


# ---- the Gemini adapter ----

class FakeGeminiSDK:
    """Stands in for google.genai.Client; records what it was asked."""

    def __init__(self, text="hello", fail_first_with=None, usage=(7, 3, None)):
        self.calls = []
        self._fail_first_with = fail_first_with
        self._text = text
        self._usage = usage
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self._fail_first_with and len(self.calls) == 1:
            raise self._fail_first_with
        prompt, cand, thoughts = self._usage
        return SimpleNamespace(
            text=self._text,
            usage_metadata=SimpleNamespace(prompt_token_count=prompt, candidates_token_count=cand,
                                           thoughts_token_count=thoughts),
        )


class HTTP400(Exception):
    code = 400


def make(sdk):
    return GeminiClient(api_key=None, timeout=5, sdk_client=sdk)


def test_adapter_returns_the_anthropic_shaped_response():
    sdk = FakeGeminiSDK(text='{"ok": true}', usage=(10, 4, 6))
    result = make(sdk).messages.create(
        model="gemini-x", max_tokens=300, system="be brief",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert result.content[0].text == '{"ok": true}'
    assert (result.usage.input_tokens, result.usage.output_tokens) == (10, 10)  # thoughts count as output


def test_adapter_maps_messages_system_prompt_and_limits():
    sdk = FakeGeminiSDK()
    make(sdk).messages.create(model="gemini-x", max_tokens=321, system="rules",
                              messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
    call = sdk.calls[0]
    assert call["model"] == "gemini-x"
    assert call["contents"] == [{"role": "user", "parts": [{"text": "q"}]},
                                {"role": "model", "parts": [{"text": "a"}]}]
    assert call["config"].system_instruction == "rules"
    assert call["config"].max_output_tokens == 321
    assert call["config"].thinking_config.thinking_budget == 0  # thinking off: it can eat the budget


def test_adapter_retries_without_thinking_setting_when_the_model_rejects_it(caplog):
    sdk = FakeGeminiSDK(fail_first_with=HTTP400("thinking not supported"))
    with caplog.at_level(logging.INFO):
        result = make(sdk).messages.create(model="m", max_tokens=10, messages=[{"role": "user", "content": "q"}])
    assert len(sdk.calls) == 2
    assert sdk.calls[0]["config"].thinking_config is not None
    assert sdk.calls[1]["config"].thinking_config is None
    assert result.content[0].text == "hello"
    assert "rejected the no-thinking setting" in caplog.text


def test_adapter_does_not_retry_other_errors():
    class HTTP429(Exception):
        code = 429

    sdk = FakeGeminiSDK(fail_first_with=HTTP429("quota"))
    with pytest.raises(HTTP429):
        make(sdk).messages.create(model="m", max_tokens=10, messages=[{"role": "user", "content": "q"}])
    assert len(sdk.calls) == 1


def test_blocked_or_empty_response_becomes_empty_text_not_a_crash():
    class Blocked:
        usage_metadata = None

        @property
        def text(self):
            raise ValueError("response blocked")

    sdk = FakeGeminiSDK()
    sdk.models.generate_content = lambda **kw: Blocked()
    result = make(sdk).messages.create(model="m", max_tokens=10, messages=[{"role": "user", "content": "q"}])
    assert result.content[0].text == "" and result.usage.input_tokens == 0
