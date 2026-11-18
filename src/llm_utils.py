"""Small helpers shared by every module that talks to an LLM."""

import json
import logging
import re

from .config import get_api_key, get_provider, get_timeout

logger = logging.getLogger(__name__)


def make_client():
    """
    Build a client for the configured provider if a key and the SDK are available, else None.
    Both providers are used through the same `.messages.create(...)` interface. Calls time out
    (config) and are not silently retried by the SDK: the pipeline owns the one allowed retry,
    so latency stays bounded.
    """
    provider, key = get_provider(), get_api_key()
    if not provider or not key:
        return None
    try:
        if provider == "gemini":
            from .gemini_client import GeminiClient
            return GeminiClient(key, get_timeout())
        import anthropic
        return anthropic.Anthropic(api_key=key, timeout=get_timeout(), max_retries=0)
    except Exception as exc:  # package missing or bad config
        logger.warning("Could not create %s client: %s", provider, exc)
        return None


def extract_json(text: str) -> object:
    """Parse JSON from model text, tolerating code fences or surrounding prose."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise
