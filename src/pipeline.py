"""
The RAG pipeline: parse -> retrieve -> generate -> guard, behind one entry point.

    from src.pipeline import answer_query
    result = answer_query("mellow music for studying on a rainy night")

Guardrail policy (kept deliberately simple):
  * Invalid song IDs in an otherwise good answer are dropped, and the event is logged.
  * If nothing valid remains (or the answer is unusable / the call fails), retry ONCE,
    telling the model what was wrong.
  * If the retry fails too, fall back to the scorer's own top matches and reasons.
  * An empty query, or a query that retrieves nothing (top score 0), is reported as
    "no_match" without calling the model: there is nothing real to ground an answer in.
  * No API client goes straight to the fallback.
  * Any unexpected exception becomes an "error" result instead of a crash.
  * Every model call has a timeout and no hidden SDK retries, so latency is bounded.

Known limitations (deferred on purpose, see the model card):
  * No semantic check that a reason is *true*. A valid ID with a wrong claim ("calm"
    for a metal track) passes. Catching that needs a second model call or a claims check.
  * No minimum retrieval score: the retriever always returns k songs, so weak matches
    still reach the model. To be decided after the evaluation shows whether it is needed.

Result shape (also what the evaluation harness reads):
    {
      "query", "parsed_query", "retrieved",           # what went in / what was found
      "summary", "recommendations",                    # [{song_id, title, artist, reason}]
      "source": "llm" | "llm_retry" | "fallback" | "no_match" | "error",
      "guardrail_events": [str],                       # everything that went wrong
      "latency_s": float,
    }
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from .generator import GenerationError, generate_recommendations
from .llm_utils import make_client
from .query_parser import empty_query, parse_query
from .recommender import load_songs
from .retriever import Retriever

logger = logging.getLogger(__name__)

DEFAULT_SONGS_PATH = Path(__file__).resolve().parent.parent / "data" / "songs.csv"
MAX_RECOMMENDATIONS = 3
FALLBACK_SUMMARY = "The AI explanation was unavailable, so these are the top matches from the scoring engine."
EMPTY_QUERY_SUMMARY = "Describe what you'd like to hear, for example a mood, an activity, or a genre."
NO_MATCH_SUMMARY = "I couldn't find anything in the catalog that matches that. Try a mood, an activity, or a genre."
ERROR_SUMMARY = "Something went wrong while handling that request. Please try again."


class RecommendationPipeline:
    def __init__(
        self,
        songs: Optional[List[Dict]] = None,
        client=None,
        k: int = 5,
        max_recommendations: int = MAX_RECOMMENDATIONS,
    ):
        self.songs = songs if songs is not None else load_songs(str(DEFAULT_SONGS_PATH))
        self.songs_by_id = {s["id"]: s for s in self.songs}
        self.retriever = Retriever(self.songs)
        self.client = client if client is not None else make_client()
        self.k = k
        self.max_recommendations = max_recommendations

    # ------------------------------------------------------------------ public

    def answer_query(self, query: str) -> Dict:
        """Never raises: unexpected failures come back as an "error" result."""
        start = time.perf_counter()
        events: List[str] = []
        try:
            return self._answer(query, events, start)
        except Exception as exc:
            logger.exception("unexpected pipeline failure for %r", query)
            self._event(events, f"unexpected error: {type(exc).__name__}: {exc}")
            return self._finish(query, empty_query(), [], ERROR_SUMMARY, [], "error", events, start)

    def _answer(self, query: str, events: List[str], start: float) -> Dict:
        if not isinstance(query, str) or not query.strip():
            self._event(events, "empty query: nothing to retrieve")
            return self._finish(query, empty_query(), [], EMPTY_QUERY_SUMMARY, [], "no_match", events, start)

        logger.info("query: %r", query)
        vocab = {"genres": self.retriever.genres, "moods": self.retriever.moods}
        parsed = parse_query(query, vocab, client=self.client, events=events)
        retrieved = self.retriever.retrieve(parsed, k=self.k)

        if not retrieved or retrieved[0]["combined"] <= 0:
            self._event(events, "no retrieval signal: nothing in the catalog matched")
            return self._finish(query, parsed, retrieved, NO_MATCH_SUMMARY, [], "no_match", events, start)

        if self.client is None:
            self._event(events, "no API client: using scorer fallback")
            return self._fallback(query, parsed, retrieved, events, start)

        outcome = self._attempt(query, retrieved, None, events)
        if outcome["ok"]:
            return self._finish(query, parsed, retrieved, outcome["summary"], outcome["recs"], "llm", events, start)

        if outcome.get("rate_limited"):
            self._event(events, f"not retrying: {outcome['problem']}")
            return self._fallback(query, parsed, retrieved, events, start)
        self._event(events, f"retrying once: {outcome['problem']}")
        allowed = [r["song"]["id"] for r in retrieved]
        feedback = (
            f"Your previous answer was rejected: {outcome['problem']}. "
            f"Recommend only songs whose ID is in {allowed}, and reply with the required JSON only."
        )
        outcome = self._attempt(query, retrieved, feedback, events)
        if outcome["ok"]:
            return self._finish(query, parsed, retrieved, outcome["summary"], outcome["recs"], "llm_retry", events, start)

        self._event(events, f"retry failed: {outcome['problem']}; using scorer fallback")
        return self._fallback(query, parsed, retrieved, events, start)

    # ------------------------------------------------------------------ internals

    def _attempt(self, query: str, retrieved: List[Dict], feedback: Optional[str], events: List[str]) -> Dict:
        """One generation call. ok=True means we have an answer we can show."""
        try:
            generated = generate_recommendations(
                query, retrieved, client=self.client,
                max_recommendations=self.max_recommendations, feedback=feedback,
            )
        except GenerationError as exc:
            return {"ok": False, "problem": f"generation error ({exc})", "rate_limited": exc.rate_limited}

        allowed = {r["song"]["id"] for r in retrieved}
        valid = [r for r in generated["recommendations"] if r["song_id"] in allowed]
        invalid = [r["song_id"] for r in generated["recommendations"] if r["song_id"] not in allowed]

        if generated["recommendations"] and not valid:
            return {"ok": False, "problem": f"song IDs {invalid} are not in the candidate list"}
        if invalid:
            self._event(events, f"dropped invalid song IDs {invalid}")

        recs = [self._describe(r["song_id"], r["reason"]) for r in valid[: self.max_recommendations]]
        return {"ok": True, "summary": generated["summary"], "recs": recs}

    def _fallback(self, query, parsed, retrieved, events, start) -> Dict:
        recs = [
            self._describe(r["song"]["id"], "; ".join(r["reasons"]) or "closest match in the catalog")
            for r in retrieved[: self.max_recommendations]
        ]
        return self._finish(query, parsed, retrieved, FALLBACK_SUMMARY, recs, "fallback", events, start)

    def _describe(self, song_id: int, reason: str) -> Dict:
        song = self.songs_by_id[song_id]
        return {"song_id": song_id, "title": song["title"], "artist": song["artist"], "reason": reason}

    @staticmethod
    def _event(events: List[str], message: str) -> None:
        events.append(message)
        logger.warning("guardrail: %s", message)

    @staticmethod
    def _finish(query, parsed, retrieved, summary, recs, source, events, start) -> Dict:
        latency = time.perf_counter() - start
        logger.info(
            "answered %r source=%s recs=%s events=%d latency=%.2fs",
            query, source, [r["song_id"] for r in recs], len(events), latency,
        )
        return {
            "query": query,
            "parsed_query": parsed,
            "retrieved": retrieved,
            "summary": summary,
            "recommendations": recs,
            "source": source,
            "guardrail_events": events,
            "latency_s": latency,
        }


_default_pipeline: Optional[RecommendationPipeline] = None


def error_result(query: str, event: str) -> Dict:
    """An "error"-shaped result for failures that happen before a pipeline exists."""
    logger.error("guardrail: %s", event)
    return RecommendationPipeline._finish(
        query, empty_query(), [], ERROR_SUMMARY, [], "error", [event], time.perf_counter()
    )


def answer_query(query: str, pipeline: Optional[RecommendationPipeline] = None) -> Dict:
    """
    Module-level entry point. Builds (and reuses) a default pipeline on the real catalog.
    Never raises: a pipeline that cannot start (e.g. missing songs.csv) yields an "error" result.
    """
    global _default_pipeline
    if pipeline is None:
        if _default_pipeline is None:
            try:
                _default_pipeline = RecommendationPipeline()
            except Exception as exc:
                logger.exception("could not start pipeline")
                return error_result(query, f"could not start pipeline: {type(exc).__name__}: {exc}")
        pipeline = _default_pipeline
    return pipeline.answer_query(query)
