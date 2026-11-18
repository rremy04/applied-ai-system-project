"""Keep the test suite hermetic: no real API key from the shell or .env may leak into a test."""
import pytest

from src.config import ALL_KEY_VARS


@pytest.fixture(autouse=True)
def _no_real_api_keys(request, monkeypatch):
    if "integration" in request.keywords:  # these opt in explicitly and need the real key
        return
    for name in ALL_KEY_VARS + ("LLM_PROVIDER", "LLM_MODEL", "LLM_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)
