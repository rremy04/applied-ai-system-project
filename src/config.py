"""
Settings for the LLM-backed parts of the system, read from the environment / `.env`.

The `.env` file is loaded once, when this module is first imported, and never overrides
variables that are already set in the shell.

    LLM_PROVIDER          "gemini" or "anthropic" (default: whichever provider has a key, Gemini first)
    GEMINI_API_KEY        Google Gemini key (GOOGLE_API_KEY also works)
    ANTHROPIC_API_KEY     Anthropic key
    LLM_MODEL             model name (default depends on the provider)
    LLM_TIMEOUT_SECONDS   seconds to wait on each call (default 30)
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

PROVIDERS = ("gemini", "anthropic")  # order = preference when both have a key
KEY_VARS = {"gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"), "anthropic": ("ANTHROPIC_API_KEY",)}
ALL_KEY_VARS = tuple(var for names in KEY_VARS.values() for var in names)
DEFAULT_MODELS = {"gemini": "gemini-3.5-flash", "anthropic": "claude-haiku-5-5"}
DEFAULT_TIMEOUT_SECONDS = 30.0


def _first_env(*names: str) -> Optional[str]:
    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return None


def get_provider() -> Optional[str]:
    """The provider to use, or None when no provider has a key."""
    explicit = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if explicit in PROVIDERS:
        return explicit
    for provider in PROVIDERS:
        if _first_env(*KEY_VARS[provider]):
            return provider
    return None


def get_api_key() -> Optional[str]:
    provider = get_provider()
    return _first_env(*KEY_VARS[provider]) if provider else None


def get_model() -> str:
    return _first_env("LLM_MODEL") or DEFAULT_MODELS[get_provider() or PROVIDERS[0]]


def get_timeout() -> float:
    """Seconds to wait on each call. Bad or non-positive values fall back to the default."""
    try:
        value = float(os.environ.get("LLM_TIMEOUT_SECONDS", ""))
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_TIMEOUT_SECONDS
