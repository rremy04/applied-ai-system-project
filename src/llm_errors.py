"""Classify provider errors so the pipeline can react correctly (for example, not retry a quota error)."""

RATE_LIMIT_MARKERS = ("resource_exhausted", "429", "quota", "rate limit", "rate_limit")


def is_rate_limit(exc: BaseException) -> bool:
    """True when the provider refused because of a quota or rate limit (HTTP 429)."""
    if getattr(exc, "code", None) == 429 or getattr(exc, "status_code", None) == 429:
        return True
    text = str(exc).lower()
    return any(marker in text for marker in RATE_LIMIT_MARKERS)


def short_reason(exc: BaseException, limit: int = 140) -> str:
    """One readable line for logs and the console, without the full provider payload."""
    if is_rate_limit(exc):
        return "rate or daily quota limit reached on the AI provider"
    text = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {text[:limit]}"
