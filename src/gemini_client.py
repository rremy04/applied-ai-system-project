"""
Gemini behind the same `client.messages.create(...)` interface the rest of the code uses
with Anthropic's client, so the parser, generator, pipeline and evaluation need no
provider-specific code. Returns an object with `.content[0].text` and `.usage`.

Choices that matter, all found by testing against the real API:
  * API version v1beta: the default (v1) rejects system instructions on some models.
  * Thinking is turned off. Otherwise "thinking" tokens can use up the whole output budget
    and truncate the JSON answer (seen: 480 of 500 tokens spent thinking). If a model
    rejects that setting (HTTP 400), the call is retried once without it.
  * No SDK retries: the pipeline owns the single allowed retry, so latency stays bounded.
"""

import logging
from types import SimpleNamespace
from typing import Optional

logger = logging.getLogger(__name__)


class GeminiClient:
    def __init__(self, api_key: Optional[str], timeout: float, sdk_client=None):
        from google import genai
        from google.genai import types

        self._types = types
        self._client = sdk_client or genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                api_version="v1beta",
                timeout=int(timeout * 1000),  # the SDK takes milliseconds
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        self.messages = SimpleNamespace(create=self._create)

    def _generate(self, model, contents, system, max_tokens, thinking_off: bool):
        types = self._types
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0) if thinking_off else None,
        )
        return self._client.models.generate_content(model=model, contents=contents, config=config)

    def _create(self, *, model, max_tokens, messages, system=None, **_ignored):
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in messages
        ]
        try:
            response = self._generate(model, contents, system, max_tokens, thinking_off=True)
        except Exception as exc:
            if getattr(exc, "code", None) != 400:
                raise
            logger.info("%s rejected the no-thinking setting (%s); retrying without it", model, exc)
            response = self._generate(model, contents, system, max_tokens, thinking_off=False)

        try:
            text = response.text or ""
        except Exception:  # e.g. blocked response: treat as empty so callers fall back
            text = ""
        usage = getattr(response, "usage_metadata", None)
        out_tokens = (getattr(usage, "candidates_token_count", 0) or 0) + (
            getattr(usage, "thoughts_token_count", 0) or 0
        )
        return SimpleNamespace(
            content=[SimpleNamespace(text=text)],
            usage=SimpleNamespace(
                input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                output_tokens=out_tokens,
            ),
        )
